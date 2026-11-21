"""Differential test: D* Lite must return a valid path whose cost equals A*'s optimal cost,
after arbitrary sequences of obstacle additions/removals while the robot moves.

Set DSTAR_DIFF_CASES=1000 for the full week-2 acceptance run (default 150 keeps CI fast).
"""
import os
import random
import unittest
from src.environment import GridMap
from src.planners.a_star import AStarPlanner
from src.planners.d_star_lite import DStarLite
from src.validation import check_path

DIAG = 1.41421356


def path_cost(p):
  return sum(DIAG if (a[0] != b[0] and a[1] != b[1]) else 1.0 for a, b in zip(p, p[1:]))


def run_case(seed, verbose=False):
  """Returns None if OK, else a description of the first mismatch."""
  rng = random.Random(seed)
  w = rng.randint(12, 35); h = rng.randint(12, 35)
  m = GridMap(w, h)
  dens = rng.choice([0.0, 0.1, 0.2, 0.3])
  for y in range(h):
    for x in range(w):
      if rng.random() < dens:
        m.set_cell(x, y, 1)
  start = (rng.randrange(w), rng.randrange(h)); goal = (rng.randrange(w), rng.randrange(h))
  m.set_cell(*start, 0); m.set_cell(*goal, 0)
  astar = AStarPlanner()
  d = DStarLite(m, start, goal)
  pos = start
  for it in range(rng.randint(3, 8)):
    dp, _, _ = d.plan(); ap, _, _ = astar.plan(m, pos, goal)
    if bool(dp) != bool(ap):
      return f'seed={seed} it={it}: existence mismatch dstar={bool(dp)} astar={bool(ap)}'
    if ap:
      ok, idx, why = check_path(m, dp)
      if not ok or dp[0] != pos or dp[-1] != goal:
        return f'seed={seed} it={it}: invalid dstar path ({why} at {idx})'
      if abs(path_cost(dp) - path_cost(ap)) > 1e-6:
        return f'seed={seed} it={it}: cost dstar={path_cost(dp):.4f} astar={path_cost(ap):.4f}'
      # move the robot a few steps along the current path
      pos = dp[min(len(dp) - 1, rng.randint(0, 4))]
    # random changes: blocks on/near the path and random far cells, plus some removals
    changed = []
    cand = (dp[1:8] if dp else []) + [(rng.randrange(w), rng.randrange(h)) for _ in range(6)]
    for c in rng.sample(cand, min(len(cand), rng.randint(1, 6))):
      if c in (pos, goal):
        continue
      v = 0 if (m.grid[c[1], c[0]] == 1 and rng.random() < 0.5) else 1
      if m.grid[c[1], c[0]] != v:
        m.set_cell(*c, v); changed.append((c, v))
    d.start = pos
    d.on_map_changed(changed)
  return None


class TestDStarDifferential(unittest.TestCase):
  def test_matches_astar_under_change_sequences(self):
    n = int(os.environ.get('DSTAR_DIFF_CASES', '150'))
    failures = [r for r in (run_case(s) for s in range(n)) if r]
    self.assertEqual(failures, [], f'{len(failures)}/{n} mismatches, first: {failures[:3]}')


if __name__ == '__main__':
  unittest.main()
