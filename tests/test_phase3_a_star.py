import unittest
from src.environment import GridMap
from src.planners.a_star import AStarPlanner, octile_distance

class TestPhase3AStar(unittest.TestCase):
  def setUp(self):
    self.planner = AStarPlanner(allow_diagonal=True)
    self.map = GridMap(15, 15)

  def test_straight_path(self):
    start = (1, 1)
    goal = (1, 5)
    path, nodes, dt = self.planner.plan(self.map, start, goal)
    self.assertEqual(len(path), 5)
    self.assertEqual(path[0], start)
    self.assertEqual(path[-1], goal)
    self.assertGreater(nodes, 0)
    self.assertGreaterEqual(dt, 0.0)

  def test_obstacle_avoidance(self):
    start = (1, 1)
    goal = (1, 5)
    self.map.set_cell(1, 3, 1)
    path, nodes, dt = self.planner.plan(self.map, start, goal)
    self.assertTrue(len(path) > 0)
    self.assertNotIn((1, 3), path)
    for p in path:
      self.assertTrue(self.map.is_free(*p))

  def test_no_corner_cutting(self):
    start = (2, 2)
    goal = (4, 4)
    self.map.set_cell(2, 3, 1)
    self.map.set_cell(3, 2, 1)
    path, _, _ = self.planner.plan(self.map, start, goal)
    self.assertTrue(len(path) > 0)
    for i in range(1, len(path)):
      prev, curr = path[i - 1], path[i]
      if abs(curr[0] - prev[0]) == 1 and abs(curr[1] - prev[1]) == 1:
        self.assertTrue(self.map.is_free(prev[0], curr[1]))
        self.assertTrue(self.map.is_free(curr[0], prev[1]))

  def test_unreachable_goal(self):
    start = (0, 0)
    goal = (14, 14)
    for y in range(15):
      self.map.set_cell(7, y, 1)
    path, nodes, dt = self.planner.plan(self.map, start, goal)
    self.assertEqual(len(path), 0)

if __name__ == '__main__':
  unittest.main()

