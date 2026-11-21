import math
import os
import unittest
import numpy as np
from src.environment import GridMap
from src.planners.a_star import AStarPlanner
from src import movingai, scenarios_v2
from src.event_eval import evaluate_episode
from src.policies import exact_policy_costs, break_even_table
import pandas as pd

HAS_MOVINGAI = os.path.isdir(os.path.join(movingai.MAP_DIR, 'room'))


class TestBoundedAStar(unittest.TestCase):
  def test_bounds_respected_and_can_fail(self):
    m = GridMap(20, 20)
    for y in range(0, 15):
      m.set_cell(10, y, 1)              # wall with a gap only at y >= 15
    p, _, _ = AStarPlanner().plan(m, (8, 5), (12, 5))
    self.assertTrue(p)                  # unbounded: goes around the wall
    p, _, _ = AStarPlanner().plan(m, (8, 5), (12, 5), bounds=(5, 2, 15, 8))
    self.assertEqual(p, [])             # the detour leaves the box
    p, _, _ = AStarPlanner().plan(m, (8, 5), (8, 7), bounds=(5, 2, 15, 8))
    self.assertTrue(all(5 <= x <= 15 and 2 <= y <= 8 for x, y in p))


class TestScenariosV2(unittest.TestCase):
  def test_generated_types_solvable_and_deterministic(self):
    for mt in ['random', 'rooms']:
      for bl in scenarios_v2.BLOCKAGES:
        a = scenarios_v2.build(mt, 60, 0.15, bl, seed=3)
        b = scenarios_v2.build(mt, 60, 0.15, bl, seed=3)
        self.assertIsNotNone(a, (mt, bl))
        gm, s, g, ev, info = a
        self.assertTrue((gm.grid == b[0].grid).all())
        self.assertEqual((s, g, ev), (b[1], b[2], b[3]))
        blocked = gm.clone()
        for e in ev:
          for c in e['add']:
            self.assertNotIn(c, (s, g))
            blocked.set_cell(*c, 1)
        self.assertTrue(AStarPlanner().plan(blocked, s, g)[0], (mt, bl))  # solvable after all blockages
        if bl == 'multi':
          self.assertGreaterEqual(len(ev), 2)

  @unittest.skipUnless(HAS_MOVINGAI, 'Moving AI maps not downloaded')
  def test_movingai_loader_and_crop(self):
    gm = movingai.load_map(movingai.family_files('room16')[0])
    self.assertEqual((gm.width, gm.height), (512, 512))
    self.assertTrue(set(np.unique(gm.grid)) <= {0, 1})
    sub = movingai.crop(gm, 100, np.random.RandomState(0))
    self.assertEqual((sub.width, sub.height), (100, 100))
    for fam in movingai.FAMILIES:
      self.assertIsNotNone(scenarios_v2.build(fam, 80, 0.0, 'wall', seed=1), fam)


class TestEventEvaluation(unittest.TestCase):
  def test_paired_event_invariants(self):
    rows_all = []
    for bl in scenarios_v2.BLOCKAGES:
      gm, s, g, ev, _ = scenarios_v2.build('random', 60, 0.15, bl, seed=11)
      rows, summ = evaluate_episode(gm, s, g, ev, R=5)
      self.assertEqual(summ['collision'], 0)
      for r in rows:
        if r['G_ok']:
          # D* Lite (incremental) must find a path of exactly the optimal cost on the same state
          self.assertAlmostEqual(r['D_cost'], r['G_cost'], places=6)
          if r['B_ok']:
            self.assertGreaterEqual(r['B_dL'], -1e-6)   # a repaired path is never shorter than optimal
        self.assertEqual(math.isnan(r['DF_edges']), bool(r['B_ok']))
      rows_all += rows
    self.assertGreater(len(rows_all), 0)
    df = pd.DataFrame(rows_all).assign(size=60)
    costs = exact_policy_costs(df)
    # the oracle can never be worse than always-repair or always-global
    self.assertTrue((costs['ORACLE_ASTAR'] <= costs['AR_ASTAR'] + 1e-9).all())
    self.assertTrue((costs['ORACLE_ASTAR'] <= costs['G_ASTAR'] + 1e-9).all())
    t = break_even_table(df, 'R')
    self.assertTrue(((t['p_fail'] >= 0) & (t['p_fail'] <= 1)).all())

  def test_dstar_work_fully_charged(self):
    # conservation: per D* shadow, work charged to events (+ end-of-episode leftover) == total work done
    cols = {'every': 'D', 'fail': 'DF', 'ufail': 'DUF', 'risk': 'DR'}
    for mt, bl in [('random', 'multi'), ('rooms', 'barrier'), ('random', 'point')]:
      gm, s, g, ev, _ = scenarios_v2.build(mt, 60, 0.15, bl, seed=4)
      rows, summ = evaluate_episode(gm, s, g, ev, R=5)
      self.assertGreater(len(rows), 0)
      df = pd.DataFrame(rows)
      for k, c in cols.items():
        charged = df[f'{c}_edges'].fillna(0).sum() + df[f'{c}_extra_edges'].sum()
        self.assertEqual(charged, summ[f'dstar_total_edges_{k}'], (mt, bl, k))

  def test_inputs_not_modified(self):
    gm, s, g, ev, _ = scenarios_v2.build('rooms', 50, 0.1, 'wall', seed=2)
    before = gm.grid.copy(); ev_before = [dict(e) for e in ev]
    evaluate_episode(gm, s, g, ev, R=5)
    self.assertTrue((gm.grid == before).all())
    self.assertEqual(ev, ev_before)


if __name__ == '__main__':
  unittest.main()
