import unittest
from src.environment import GridMap
from src.planners.d_star_lite import DStarLite

class TestPhase4DStarLite(unittest.TestCase):
  def setUp(self):
    self.map = GridMap(15, 15)
    self.start = (1, 1)
    self.goal = (13, 13)
    self.planner = DStarLite(self.map, self.start, self.goal)

  def test_initial_plan(self):
    path, nodes, dt = self.planner.plan()
    self.assertTrue(len(path) > 0)
    self.assertEqual(path[0], self.start)
    self.assertEqual(path[-1], self.goal)
    self.assertGreater(nodes, 0)
    self.assertGreaterEqual(dt, 0.0)

  def test_incremental_replanning_on_obstacle(self):
    path1, _, _ = self.planner.plan()
    mid_node = path1[len(path1) // 2]
    self.map.set_cell(mid_node[0], mid_node[1], 1)
    self.planner.on_map_changed([(mid_node, 1)])
    path2, nodes, dt = self.planner.plan()
    self.assertTrue(len(path2) > 0)
    self.assertNotIn(mid_node, path2)
    self.assertEqual(path2[0], self.start)
    self.assertEqual(path2[-1], self.goal)
    for p in path2:
      self.assertTrue(self.map.is_free(*p))

  def test_multiple_obstacles_update(self):
    path1, _, _ = self.planner.plan()
    obs = [(path1[2][0], path1[2][1]), (path1[3][0], path1[3][1])]
    for ox, oy in obs:
      self.map.set_cell(ox, oy, 1)
    self.planner.on_map_changed([(o, 1) for o in obs])
    path2, _, _ = self.planner.plan()
    self.assertTrue(len(path2) > 0)
    for o in obs:
      self.assertNotIn(o, path2)

if __name__ == '__main__':
  unittest.main()

