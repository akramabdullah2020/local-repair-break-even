"""Paired (counterfactual) event-level evaluation — the core measurement of the study.

The robot is driven by a neutral executor (it always follows the optimal path on its belief map).
At every blocking event (a newly perceived change invalidates the current path) ALL strategies are
evaluated on exactly the same state, so their costs are directly comparable:

  * G_ASTAR : global A* from the robot to the goal
  * G_DSTAR : D* Lite planning at every event ("shadow" instance kept in sync with every map
              change; its deferred on_map_changed work since the previous plan() is charged to the event)
  * REPAIR_B: bounded local repair (search restricted to the perception window)  -> success? cost?
  * REPAIR_U: unbounded local repair (v1 behaviour: A* between splice points over the whole map)
  * DSTAR_ONFAIL: a second D* Lite shadow that plans only when bounded repair fails — gives the exact
              cost of the policy "repair first, fall back to D* Lite" (and of its oracle)

Work is measured in edge-cost evaluations (hardware independent): A*-based searches examine 8 edges
per expansion; D* Lite counts its own. Wall-clock milliseconds are recorded as a secondary metric.
"""
import time
from src.environment import PerceptionWindow
from src.planners.a_star import AStarPlanner, octile_distance
from src.planners.d_star_lite import DStarLite
from src.planners.risk_adaptive_planner import RiskAdaptivePlanner
from src.validation import check_move, check_path

DIAG = 1.41421356
NAN = float('nan')


def path_cost(p):
  if not p:
    return NAN
  return sum(DIAG if (a[0] != b[0] and a[1] != b[1]) else 1.0 for a, b in zip(p, p[1:]))


def _window_free_frac(known, pos, R):
  x0, x1 = max(0, pos[0] - R), min(known.width - 1, pos[0] + R)
  y0, y1 = max(0, pos[1] - R), min(known.height - 1, pos[1] + R)
  sub = known.grid[y0:y1 + 1, x0:x1 + 1]
  return float((sub == 0).mean())


def _splice_in_window(path, last_bad, pos, R, known):
  for j in range(last_bad + 1, len(path)):
    c = path[j]
    if max(abs(c[0] - pos[0]), abs(c[1] - pos[1])) <= R and known.is_free(*c):
      return 1
  return 0


SHADOW_COLS = {'fail': 'DF', 'ufail': 'DUF', 'risk': 'DR', 'pred': 'DP', 'predtau': 'DT'}


def evaluate_episode(true_map, start, goal, events, R, max_steps=None, predictor=None, risk_theta=0.45):
  """Returns (event_rows, episode_summary). Inputs are copied; the caller's objects are not modified.

  predictor: optional fitted src.policies.RepairPredictor; when given, two extra D* Lite shadows
  measure the exact cost of the predictor policies (break-even rule 'pred', tuned threshold 'predtau').
  """
  pc = time.perf_counter
  true_map = true_map.clone()
  events = [dict(e) for e in events]
  known = true_map.clone()
  win = PerceptionWindow(radius=R)
  win.update_belief(start, true_map, known)
  astar = AStarPlanner()
  path, _, _ = astar.plan(known, start, goal)
  if not path:
    return [], dict(success=0, steps=0, n_events=0, collision=0, no_initial_path=1)
  size = max(true_map.width, true_map.height)
  names = ['every', 'fail', 'ufail', 'risk'] + (['pred', 'predtau'] if predictor is not None else [])
  shadows = {k: DStarLite(known, start, goal) for k in names}
  for d in shadows.values():
    d.plan()                              # initial plan is not charged to any event
  init_edges = {k: d.edge_evals for k, d in shadows.items()}
  pending = {k: [0, 0.0] for k in shadows}  # deferred on_map_changed work: [edges, ms]
  rep = {'U': RiskAdaptivePlanner(bounded_repair=False), 'B': RiskAdaptivePlanner(bounded_repair=True)}

  max_steps = max_steps or 6 * max(true_map.width, true_map.height)
  pos = start; step = 0; rows = []; success = 0; collision = 0; waits = 0
  while step < max_steps and pos != goal:
    step += 1
    for e in events:
      if e.get('done') or max(abs(pos[0] - e['near'][0]), abs(pos[1] - e['near'][1])) > e['dist']:
        continue
      e['done'] = True
      for c in e.get('add', []):
        if c != pos:
          true_map.set_cell(*c, 1)
      for c in e.get('open_alternative', []):
        true_map.set_cell(*c, 0)
    changed = win.update_belief(pos, true_map, known)
    if changed:
      for k, d in shadows.items():
        e0 = d.edge_evals; t = pc()
        d.start = pos; d.on_map_changed(changed)
        pending[k][1] += (pc() - t) * 1000.0; pending[k][0] += d.edge_evals - e0
      ok = check_path(known, path)[0]
      if not ok:
        row = dict(step=step, x=pos[0], y=pos[1], R=R, n_changed=len(changed),
                   n_blocked_changed=sum(1 for _, v in changed if v == 1),
                   dist_to_goal=octile_distance(pos, goal), remaining_len=len(path))
        # ---- risk model (also the predictor's cheap features)
        t = pc()
        risk, ratio, prox, bottle = rep['B'].compute_risk(path, changed, known, R)
        inv = rep['B'].invalid_indices(path, known)
        row.update(risk=risk, risk_ratio=ratio, risk_prox=prox, risk_bottle=bottle,
                   risk_ms=(pc() - t) * 1000.0,
                   n_invalid=len(inv), first_invalid=inv[0] if inv else -1,
                   invalid_span=(inv[-1] - inv[0]) if inv else 0,
                   splice_in_window=_splice_in_window(path, inv[-1], pos, R, known) if inv else 0,
                   window_free=_window_free_frac(known, pos, R))
        # ---- global A*
        t = pc(); pG, nG, _ = astar.plan(known, pos, goal)
        row.update(G_ms=(pc() - t) * 1000.0, G_edges=8 * nG, G_ok=int(bool(pG)), G_cost=path_cost(pG))
        # ---- D* Lite at every event
        d = shadows['every']; e0 = d.edge_evals; t = pc()
        pD, _, _ = d.plan()
        row.update(D_ms=(pc() - t) * 1000.0 + pending['every'][1],
                   D_edges=d.edge_evals - e0 + pending['every'][0], D_cost=path_cost(pD))
        pending['every'] = [0, 0.0]
        # ---- local repair, bounded and unbounded
        for tag, rp in rep.items():
          t = pc(); pL, nL, _ = rp.try_local_repair(pos, path, known, R)
          s = pL is not None
          row.update({f'{tag}_ms': (pc() - t) * 1000.0, f'{tag}_edges': 8 * nL, f'{tag}_ok': int(s),
                      f'{tag}_dL': (path_cost(pL) - path_cost(pG)) if (s and pG) else NAN})
        # ---- policy decisions that need a D* Lite call on this event
        calls = {'fail': not row['B_ok'],                                  # repair_B, then D*
                 'ufail': not row['U_ok'],                                 # repair_U, then D*
                 'risk': row['risk'] >= risk_theta or not row['B_ok']}     # v1 risk gate, then D*
        if predictor is not None:
          t = pc()
          p = float(predictor.p_fail_row(row, size))
          rep_an = predictor.decide_repair_analytic_row(row, size, p)       # break-even rule
          rep_tau = p < predictor.tau                                       # threshold tuned on training
          row.update(pred_p=p, pred_repair=int(rep_an), predtau_repair=int(rep_tau),
                     pred_ms=(pc() - t) * 1000.0)
          calls['pred'] = (not rep_an) or not row['B_ok']
          calls['predtau'] = (not rep_tau) or not row['B_ok']
        # ---- conditional D* Lite shadows: each plans only when its policy calls it, so deferred
        #      work is charged exactly as that policy would experience it
        for k, need in calls.items():
          col = SHADOW_COLS[k]
          if need:
            d = shadows[k]; e0 = d.edge_evals; t = pc()
            d.plan()
            row.update({f'{col}_ms': (pc() - t) * 1000.0 + pending[k][1],
                        f'{col}_edges': d.edge_evals - e0 + pending[k][0]})
            pending[k] = [0, 0.0]
          else:
            row.update({f'{col}_ms': NAN, f'{col}_edges': NAN})
        rows.append(row)
        if pG:
          path = pG
    # ---- executor: follow the optimal path
    if len(path) > 1:
      nxt = path[1]
      if check_move(known, pos, nxt) is not None:
        waits += 1; continue
      path = path[1:]
    else:
      nxt = path[0]
    if check_move(true_map, pos, nxt) is not None:
      collision = 1; break
    pos = nxt
    if pos == goal:
      success = 1
  # A D*-based policy must keep its D* Lite instance updated with EVERY map change, even when it does
  # not plan. Work deferred after the shadow's last plan() call is charged to the episode's last event
  # in *_extra_* columns (all rows carry the columns, 0 elsewhere), so no update work goes uncounted.
  all_cols = {'every': 'D', **SHADOW_COLS}
  for r in rows:
    for k in shadows:
      r[f'{all_cols[k]}_extra_edges'] = 0; r[f'{all_cols[k]}_extra_ms'] = 0.0
  if rows:
    for k in shadows:
      rows[-1][f'{all_cols[k]}_extra_edges'] = pending[k][0]; rows[-1][f'{all_cols[k]}_extra_ms'] = pending[k][1]
  totals = {k: d.edge_evals - init_edges[k] for k, d in shadows.items()}
  return rows, dict(success=success, steps=step, n_events=len(rows), collision=collision,
                    waits=waits, no_initial_path=0,
                    uncharged_dstar_edges=0 if rows else int(sum(p[0] for p in pending.values())),
                    **{f'dstar_total_edges_{k}': v for k, v in totals.items()})
