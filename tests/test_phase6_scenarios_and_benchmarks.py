import unittest
from src.scenarios import ScenarioBuilder


class TestPhase6Scenarios(unittest.TestCase):
  def test_scenario_creation_all_five(self):
    for s_id in range(1, 6):
      m, start, goal, events = ScenarioBuilder.create_map(s_id, width=25, height=25, seed=42)
      self.assertEqual(m.width, 25)
      self.assertEqual(m.height, 25)
      self.assertTrue(m.is_free(*start))
      self.assertTrue(m.is_free(*goal))
      if s_id > 1:
        self.assertTrue(len(events) > 0)

  def test_events_identical_for_every_planner(self):
    # same scenario + seed must give the same map and the same events, whoever plans
    for s_id in range(1, 6):
      m1, s1, g1, e1 = ScenarioBuilder.create_map(s_id, 30, 30, seed=101)
      m2, s2, g2, e2 = ScenarioBuilder.create_map(s_id, 30, 30, seed=101)
      self.assertTrue((m1.grid == m2.grid).all())
      self.assertEqual((s1, g1, e1), (s2, g2, e2))

  def test_different_seeds_change_scenario5(self):
    # v1 bug: scenario 5 ignored the seed, so its 'replicates' were identical
    evs = {str(ScenarioBuilder.create_map(5, 30, 30, seed=sd)[3]) for sd in [42, 101, 202, 303, 404]}
    self.assertGreater(len(evs), 1)


if __name__ == '__main__':
  unittest.main()
