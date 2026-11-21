import unittest
from src.environment import GridMap
from src.planners.a_star import AStarPlanner
from src.planners.risk_adaptive_planner import RiskAdaptivePlanner
from src.scenarios import ScenarioBuilder
from src.simulation import SimulationRunner
from src.validation import check_move, check_path

PLANNERS = [('A_STAR', 0.45), ('D_STAR_LITE', 0.45), ('RISK', 0.45), ('RISK', 0.20)]


class TestValidator(unittest.TestCase):
  def setUp(self):
    self.m = GridMap(10, 10)

  def test_legal_moves(self):
    self.assertIsNone(check_move(self.m, (2, 2), (3, 2)))
    self.assertIsNone(check_move(self.m, (2, 2), (3, 3)))
    self.assertIsNone(check_move(self.m, (2, 2), (2, 2)))  # waiting

  def test_rejects_jump(self):
    self.assertEqual(check_move(self.m, (0, 7), (7, 7)), 'non_adjacent')

  def test_rejects_occupied_and_out_of_map(self):
    self.m.set_cell(3, 2, 1)
    self.assertEqual(check_move(self.m, (2, 2), (3, 2)), 'occupied')
    self.assertEqual(check_move(self.m, (0, 0), (-1, 0)), 'occupied')

  def test_rejects_corner_cut(self):
    self.m.set_cell(3, 2, 1)
    self.assertEqual(check_move(self.m, (2, 2), (3, 3)), 'corner_cut')

  def test_check_path(self):
    self.assertEqual(check_path(self.m, [(0, 7), (7, 7)])[2], 'non_adjacent')
    self.assertTrue(check_path(self.m, [(0, 0), (1, 1), (2, 1)])[0])


class TestRiskPlannerPathCheck(unittest.TestCase):
  def test_is_path_valid_rejects_jump(self):
    # counterexample reported in the peer review: this used to return True
    self.assertFalse(RiskAdaptivePlanner().is_path_valid([(0, 7), (7, 7)], GridMap(10, 10)))

  def test_corner_only_blockage_is_not_reused(self):
    # regression: a new obstacle beside a diagonal move leaves every path CELL free,
    # but the move is illegal; v1 treated the path as valid and the robot cut the corner
    m = GridMap(10, 10)
    path = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    m.set_cell(3, 2, 1)  # corner of the move (2,2)->(3,3)
    pl = RiskAdaptivePlanner(risk_threshold=0.45)
    self.assertEqual(pl.invalid_indices(path, m), [2])
    newp, strat, _, _, _ = pl.replan((1, 1), (5, 5), path, [((3, 2), 1)], m, 5)
    self.assertNotEqual(strat, 'reused')
    self.assertTrue(check_path(m, newp)[0])

  def test_is_path_valid_accepts_astar_path(self):
    m = GridMap(15, 15, obstacle_prob=0.15, seed=3); m.set_cell(1, 1, 0); m.set_cell(13, 13, 0)
    p, _, _ = AStarPlanner().plan(m, (1, 1), (13, 13))
    if p:
      self.assertTrue(RiskAdaptivePlanner().is_path_valid(p, m))


class TestSimulationInvariants(unittest.TestCase):
  def _run(self, sc, seed, r, pt, thr, size=30):
    m, s, g, ev = ScenarioBuilder.create_map(sc, size, size, seed)
    return SimulationRunner(r, 200).run(pt, m, s, g, ev, thr=thr, log_events=True)

  def test_every_executed_move_is_legal(self):
    for sc in range(1, 6):
      for seed in [42, 101, 202]:
        for r in [3, 8]:
          for pt, thr in PLANNERS:
            res = self._run(sc, seed, r, pt, thr)
            if res is None:
              continue
            self.assertEqual(res['invalid_reason'], '', (sc, seed, r, pt))
            trail = res['trail']
            for a, b in zip(trail, trail[1:]):
              self.assertEqual(max(abs(a[0]-b[0]), abs(a[1]-b[1])), 1, (sc, seed, r, pt, a, b))

  def test_event_log_consistent_with_summary(self):
    res = self._run(4, 42, 5, 'RISK', 0.45)
    self.assertEqual(len(res['events']), res['n_events'])
    if res['n_events']:
      self.assertAlmostEqual(sum(e['ms'] for e in res['events']) / res['n_events'], res['event_ms'])
      self.assertAlmostEqual(sum(e['nodes'] for e in res['events']) / res['n_events'], res['event_nodes'])

  def test_deterministic_outcomes_reproducible(self):
    keys = ['success', 'collision', 'path_length', 'init_nodes', 'replan_nodes', 'n_events',
            'n_local', 'n_broader', 'local_fail', 'stuck_steps']
    for pt, thr in PLANNERS:
      a = self._run(4, 303, 5, pt, thr); b = self._run(4, 303, 5, pt, thr)
      self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys}, pt)

  def test_unknown_planner_rejected(self):
    with self.assertRaises(ValueError):
      self._run(1, 42, 5, 'NOPE', 0.45)


if __name__ == '__main__':
  unittest.main()
