import unittest
from src.environment import GridMap
from src.planners.a_star import AStarPlanner
from src.planners.risk_adaptive_planner import RiskAdaptivePlanner

class TestPhase5RiskAdaptive(unittest.TestCase):
  def setUp(self):
    self.map = GridMap(20, 20)
    self.start = (2, 2)
    self.goal = (17, 17)
    self.planner = RiskAdaptivePlanner(risk_threshold=0.45)
    self.init_path, _, _ = AStarPlanner().plan(self.map, self.start, self.goal)

  def test_risk_computation_isolated_obstacle(self):
    obs = (5, 5)
    self.map.set_cell(obs[0], obs[1], 1)
    risk, ratio, prox, bottle = self.planner.compute_risk(self.init_path, [(obs, 1)], self.map, 5)
    self.assertGreater(risk, 0.0)
    self.assertLess(risk, self.planner.risk_threshold)

  def test_risk_computation_corridor_blockage(self):
    for y in range(20):
      if y != 10:
        self.map.set_cell(10, y, 1)
    for p in self.init_path:
      if p[0] == 10:
        self.map.set_cell(p[0], p[1], 1)
        risk, _, _, bottle = self.planner.compute_risk(self.init_path, [((p[0], p[1]), 1)], self.map, 5)
        self.assertGreaterEqual(bottle, 0.5)

  def test_local_repair_success(self):
    block_pt = self.init_path[4]
    self.map.set_cell(block_pt[0], block_pt[1], 1)
    repaired, strat, nodes, dt, risk = self.planner.replan(
      self.init_path[2], self.goal, self.init_path[2:], [(block_pt, 1)], self.map, 5
    )
    self.assertEqual(strat, 'local_repair')
    self.assertTrue(len(repaired) > 0)
    self.assertNotIn(block_pt, repaired)
    for p in repaired:
      self.assertTrue(self.map.is_free(*p))

  def test_fallback_to_broader_replan(self):
    for i in range(3, 8):
      pt = self.init_path[i]
      self.map.set_cell(pt[0], pt[1], 1)
    repaired, strat, nodes, dt, risk = self.planner.replan(
      self.init_path[0], self.goal, self.init_path, [(self.init_path[i], 1) for i in range(3, 8)], self.map, 5
    )
    self.assertEqual(strat, 'broader_replan')
    self.assertTrue(len(repaired) > 0)

if __name__ == '__main__':
  unittest.main()

