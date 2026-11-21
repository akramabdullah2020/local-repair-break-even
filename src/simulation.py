import time
from src.environment import PerceptionWindow
from src.planners.a_star import AStarPlanner, octile_distance
from src.planners.d_star_lite import DStarLite
from src.planners.risk_adaptive_planner import RiskAdaptivePlanner
from src.validation import check_move, check_path


class SimulationRunner:
  """Runs one navigation episode.

  Measurement protocol (identical for every planner):
    * the robot knows the initial map; only cells inside the perception window are refreshed each step;
    * a *blocking event* is a step where a newly perceived change makes the current path invalid;
    * the timer wraps everything the planner does in response to a perceived change
      (risk computation, D* Lite vertex updates, repair, fallback), so no planner gets free work;
    * event_ms / event_nodes are averaged over blocking events only (non-blocking changes are
      reported separately in total_change_ms);
    * every executed move is checked by the independent validator in src/validation.py.
  """

  def __init__(self, perception_radius=5, max_steps=200):
    self.R = perception_radius; self.max_steps = max_steps
    self.window = PerceptionWindow(radius=perception_radius)

  def run(self, ptype, true_map, start, goal, events, thr=0.45, log_events=False):
    known = true_map.clone()
    pc = time.perf_counter
    self.window.update_belief(start, true_map, known)
    if ptype == 'A_STAR':
      pl = AStarPlanner(); path, n0, _ = pl.plan(known, start, goal)
    elif ptype == 'D_STAR_LITE':
      pl = DStarLite(known, start, goal); path, n0, _ = pl.plan()
    elif ptype in ('RISK', 'RISK_ADAPTIVE'):
      pl = RiskAdaptivePlanner(risk_threshold=thr); path, n0, _ = pl.a_star.plan(known, start, goal)
    else:
      raise ValueError(f'Unknown planner type: {ptype}')
    if not path: return None
    pos = start; trail = [pos]; step = 0; success = 0; collision = 0; invalid_reason = ''
    reinit = 0; total_ms = 0.0; ev_ms = []; ev_nodes = []; ev_edges = []; rnodes = 0; strat = []; stuck = 0
    ev_log = []
    while step < self.max_steps and pos != goal:
      step += 1
      for e in events:
        if 'near' in e:
          if e.get('done') or max(abs(pos[0]-e['near'][0]), abs(pos[1]-e['near'][1])) > e['dist']: continue
          e['done'] = True
        elif e['step'] != step: continue
        for c in e.get('add', []):
          if c != pos: true_map.set_cell(*c, 1)   # an obstacle cannot appear on the robot itself
        for c in e.get('open_alternative', []): true_map.set_cell(*c, 0)
      changed = self.window.update_belief(pos, true_map, known)
      if changed:
        # path is invalid if a cell is occupied OR a diagonal move now cuts a newly perceived obstacle
        blocked = not check_path(known, path)[0]
        nn_event = 0; s = None; risk = None; d_reinit_now = False
        att0 = getattr(pl, 'local_repair_attempts', 0); fl0 = getattr(pl, 'local_repair_failures', 0)
        e0 = getattr(pl, 'edge_evals', 0)
        ta = pc()
        if ptype == 'A_STAR':
          if blocked:
            newp, nn, _ = pl.plan(known, pos, goal); nn_event += nn
            if newp: path = newp
        elif ptype == 'D_STAR_LITE':
          pl.start = pos; pl.on_map_changed(changed)
          if blocked:
            newp, nn, _ = pl.plan(); nn_event += nn
            if not newp:   # incremental D* returned no path: re-initialise from scratch (counted in time)
              pl = DStarLite(known, pos, goal); newp, nn, _ = pl.plan(); nn_event += nn; reinit += 1
              d_reinit_now = True
            if newp: path = newp
        else:
          newp, s, nn, _, risk = pl.replan(pos, goal, path, changed, known, self.R); nn_event += nn; strat.append(s)
          if newp: path = newp
        dt = (pc() - ta) * 1000.0
        # hardware-independent work: edge-cost evaluations. A*-based planners examine 8 edges per
        # expansion; D* Lite counts its own evaluations (incl. on_map_changed). After a D* re-init
        # the new instance's counter is used.
        if ptype == 'D_STAR_LITE':
          edges_event = pl.edge_evals - (0 if d_reinit_now else e0)
        else:
          edges_event = 8 * nn_event
        total_ms += dt; rnodes += nn_event
        if blocked:
          ev_ms.append(dt); ev_nodes.append(nn_event); ev_edges.append(edges_event)
          if log_events:
            att = getattr(pl, 'local_repair_attempts', 0) - att0
            fl = getattr(pl, 'local_repair_failures', 0) - fl0
            ev_log.append(dict(step=step, x=pos[0], y=pos[1], n_changed=len(changed),
                               strategy=s or ('global_astar' if ptype == 'A_STAR' else 'dstar'),
                               risk=risk, repair_attempted=att, repair_failed=fl,
                               d_reinit=int(d_reinit_now), nodes=nn_event, edges=edges_event, ms=dt))
      if len(path) > 1:
        nxt = path[1]
        if check_move(known, pos, nxt) is not None:  # robot waits instead of making a move it knows is illegal
          stuck += 1; continue
        path = path[1:]
      else:
        nxt = path[0]
      reason = check_move(true_map, pos, nxt)
      if reason is not None:
        collision = 1; invalid_reason = reason; break
      pos = nxt; trail.append(pos)
      if pos == goal: success = 1; break
    plen = sum(octile_distance(trail[i-1], trail[i]) for i in range(1, len(trail)))
    att = getattr(pl, 'local_repair_attempts', 0); fl = getattr(pl, 'local_repair_failures', 0)
    out = dict(success=success, collision=collision, path_length=plen, init_nodes=n0, replan_nodes=rnodes,
      total_change_ms=total_ms, n_events=len(ev_ms), event_ms=(sum(ev_ms)/len(ev_ms) if ev_ms else float('nan')),
      event_nodes=(sum(ev_nodes)/len(ev_nodes) if ev_nodes else float('nan')),
      event_edges=(sum(ev_edges)/len(ev_edges) if ev_edges else float('nan')),
      n_local=strat.count('local_repair'), n_broader=strat.count('broader_replan'), n_reused=strat.count('reused'),
      local_attempts=att, local_fail=fl, stuck_steps=stuck, d_reinit=reinit, steps=step,
      invalid_reason=invalid_reason)
    if log_events:
      out['events'] = ev_log
      out['trail'] = trail
    return out
