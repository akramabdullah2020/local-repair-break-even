"""Independent motion validator.

Deliberately does NOT reuse GridMap.neighbors() or any planner code, so a bug in
the planners or in the neighbour generator cannot hide itself.
"""


def _free(grid_map, x, y):
  return 0 <= x < grid_map.width and 0 <= y < grid_map.height and int(grid_map.grid[y, x]) == 0


def check_move(grid_map, a, b):
  """Return None if the single move a->b is legal on grid_map, else a reason string.

  Legal = target inside the map and free, Chebyshev distance exactly 1 (or 0 for
  waiting in place), and diagonal moves do not cut a blocked corner.
  """
  if a == b:
    return None if _free(grid_map, *b) else 'occupied'
  dx, dy = b[0] - a[0], b[1] - a[1]
  if max(abs(dx), abs(dy)) != 1:
    return 'non_adjacent'
  if not _free(grid_map, *b):
    return 'occupied'
  if dx != 0 and dy != 0:
    if not _free(grid_map, a[0], b[1]) or not _free(grid_map, b[0], a[1]):
      return 'corner_cut'
  return None


def check_path(grid_map, path):
  """Validate a whole path against a static map. Returns (ok, first_bad_index, reason)."""
  if not path:
    return False, 0, 'empty'
  if not _free(grid_map, *path[0]):
    return False, 0, 'occupied'
  for i in range(1, len(path)):
    reason = check_move(grid_map, path[i - 1], path[i])
    if reason is not None:
      return False, i, reason
  return True, None, None
