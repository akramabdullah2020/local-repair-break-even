import unittest
import numpy as np
from src.environment import GridMap, PerceptionWindow

class TestPhase2Environment(unittest.TestCase):
  def setUp(self):
    self.width = 20
    self.height = 20
    self.grid_map = GridMap(self.width, self.height)

  def test_grid_initialization_and_bounds(self):
    self.assertEqual(self.grid_map.width, 20)
    self.assertEqual(self.grid_map.height, 20)
    self.assertTrue(self.grid_map.is_valid(0, 0))
    self.assertTrue(self.grid_map.is_valid(19, 19))
    self.assertFalse(self.grid_map.is_valid(20, 20))
    self.assertFalse(self.grid_map.is_valid(-1, 0))

  def test_obstacle_setting(self):
    self.assertTrue(self.grid_map.is_free(5, 5))
    self.grid_map.set_cell(5, 5, 1)
    self.assertFalse(self.grid_map.is_free(5, 5))
    self.grid_map.set_cell(5, 5, 0)
    self.assertTrue(self.grid_map.is_free(5, 5))

  def test_perception_window_coverage(self):
    for r in [3, 5, 8]:
      window = PerceptionWindow(radius=r, metric='chebyshev')
      robot_pos = (10, 10)
      visible = window.get_visible_cells(robot_pos, self.width, self.height)
      expected_count = (2 * r + 1) ** 2
      self.assertEqual(len(visible), expected_count)
      for x, y in visible:
        self.assertTrue(max(abs(x - 10), abs(y - 10)) <= r)

  def test_perception_window_belief_update(self):
    true_map = GridMap(self.width, self.height)
    belief_map = GridMap(self.width, self.height)
    window = PerceptionWindow(radius=5)
    true_map.set_cell(12, 12, 1)
    true_map.set_cell(18, 18, 1)
    robot_pos = (10, 10)
    changed = window.update_belief(robot_pos, true_map, belief_map)
    self.assertIn(((12, 12), 1), changed)
    self.assertEqual(belief_map.grid[12, 12], 1)
    self.assertNotIn(((18, 18), 1), changed)
    self.assertEqual(belief_map.grid[18, 18], 0)

if __name__ == '__main__':
  unittest.main()

