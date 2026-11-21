"""Scenario generator v2 (plan §7).

A scenario = (true_map, start, goal, events, info). Blockages are placed on the reference
(optimal) path and appear when the robot first comes within `dist` cells of them, so every
planner faces the same geometry. Every scenario remains solvable after all blockages appear.

Map types : 'random' (uniform clutter), 'rooms' (rooms + doors + clutter),
            or a Moving AI family from src.movingai.FAMILIES ('room8', 'maze8', 'street', ...).
Blockages : 'point'   one cell on the path
            'wall'    3/5/7 cells perpendicular to the path
            'barrier' perpendicular line extended until it meets obstacles (closes a passage),
                      shortened from its ends only if needed to keep the goal reachable
            'multi'   three 3-cell walls at different points of the path
"""
import numpy as np
from src.environment import GridMap
from src.planners.a_star import AStarPlanner, octile_distance
from src import movingai

MAP_TYPES = ['random', 'rooms'] + list(movingai.FAMILIES)
BLOCKAGES = ['point', 'wall', 'barrier', 'multi']


# ---------------------------------------------------------------- maps
def _random_map(size, density, rng):
  gm = GridMap(size, size)
  gm.grid = (rng.rand(size, size) < density).astype(np.int32)
  return gm


def _rooms_map(size, density, rng):
  gm = GridMap(size, size); g = gm.grid
  s = int(rng.randint(8, 16))
  lines = list(range(s, size - 1, s))
  for k in lines:
    g[:, k] = 1; g[k, :] = 1
  bounds = [0] + lines + [size]
  for k in lines:                       # doors (2 cells wide) in every wall segment
    for a, b in zip(bounds, bounds[1:]):
      lo, hi = a + 1, b - 2
      if hi <= lo:
        continue
      d = int(rng.randint(lo, hi))
      g[d:d + 2, k] = 0                 # door in vertical wall x=k
      d = int(rng.randint(lo, hi))
      g[k, d:d + 2] = 0                 # door in horizontal wall y=k
  clutter = (rng.rand(size, size) < density * 0.5) & (g == 0)
  g[clutter] = 1
  return gm


def _base_map(map_type, size, density, rng):
  if map_type == 'random':
    return _random_map(size, density, rng), 'random'
  if map_type == 'rooms':
    return _rooms_map(size, density, rng), 'rooms'
  return movingai.load_family_map(map_type, size, rng)


# ---------------------------------------------------------------- helpers
def _pick_start_goal(gm, rng, min_frac=0.5):
  comp = movingai.largest_component(gm)
  if len(comp) < 20:
    return None
  size = max(gm.width, gm.height)
  for _ in range(30):
    s = comp[int(rng.randint(0, len(comp)))]
    cands = [comp[int(i)] for i in rng.randint(0, len(comp), size=200)]
    g = max(cands, key=lambda c: octile_distance(s, c))
    if octile_distance(s, g) >= min_frac * size:
      return s, g
  return None


def _solvable(gm, start, goal, extra_blocked=()):
  m = gm.clone()
  for c in extra_blocked:
    m.set_cell(*c, 1)
  p, _, _ = AStarPlanner().plan(m, start, goal)
  return bool(p)


def _perp(path, k):
  a = path[max(0, k - 1)]; b = path[min(len(path) - 1, k + 1)]
  dx, dy = b[0] - a[0], b[1] - a[1]
  nx, ny = -dy, dx
  nx = int(np.sign(nx)); ny = int(np.sign(ny))
  return (nx, ny) if (nx, ny) != (0, 0) else (0, 1)


def _line(gm, c, n, half, stop_at_obstacle):
  cells = [c]
  for sgn in (1, -1):
    for i in range(1, half + 1):
      p = (c[0] + sgn * i * n[0], c[1] + sgn * i * n[1])
      if not gm.is_valid(*p):
        break
      if stop_at_obstacle and not gm.is_free(*p):
        break
      cells.append(p)
  return cells


def _blockage_cells(kind, gm, path, k, rng):
  c = path[k]; n = _perp(path, k)
  if kind == 'point':
    return [c]
  if kind == 'wall':
    return _line(gm, c, n, int(rng.choice([1, 2, 3])), stop_at_obstacle=False)
  if kind == 'barrier':
    return _line(gm, c, n, 25, stop_at_obstacle=True)
  raise ValueError(kind)


def _make_solvable(gm, start, goal, cells, placed):
  """Drop cells from the ends of the line (farthest from the path cell first) until solvable."""
  c0 = cells[0]
  cells = sorted(cells, key=lambda p: max(abs(p[0] - c0[0]), abs(p[1] - c0[1])))
  while cells and not _solvable(gm, start, goal, placed + cells):
    if len(cells) == 1:
      return []
    cells = cells[:-1]
  return cells


# ---------------------------------------------------------------- public API
def build(map_type, size, density, blockage, seed, max_tries=20):
  """Returns (true_map, start, goal, events, info) or None if no valid scenario was found."""
  rng = np.random.RandomState(seed)
  for _ in range(max_tries):
    gm, map_name = _base_map(map_type, size, density, rng)
    sg = _pick_start_goal(gm, rng)
    if sg is None:
      continue
    start, goal = sg
    ref, _, _ = AStarPlanner().plan(gm, start, goal)
    if not ref or len(ref) < 12:
      continue
    fracs = [0.3, 0.5, 0.7] if blockage == 'multi' else [float(rng.uniform(0.25, 0.75))]
    events, placed = [], []
    for f in fracs:
      k = int(f * (len(ref) - 1))
      if blockage == 'multi':           # three short (3-cell) walls
        cells = _line(gm, ref[k], _perp(ref, k), 1, stop_at_obstacle=False)
      else:
        cells = _blockage_cells(blockage, gm, ref, k, rng)
      cells = [p for p in cells if p not in (start, goal) and gm.is_free(*p)]
      cells = _make_solvable(gm, start, goal, cells, placed)
      if not cells:
        continue
      placed += cells
      events.append({'near': ref[k], 'dist': int(rng.randint(2, 11)), 'add': cells})
    if not events:
      continue
    info = dict(map_type=map_type, map_name=map_name, size=size, density=density, blockage=blockage,
                seed=seed, ref_len=len(ref), n_block_cells=len(placed),
                free_frac=round(float((gm.grid == 0).mean()), 3))
    return gm, start, goal, events, info
  return None
