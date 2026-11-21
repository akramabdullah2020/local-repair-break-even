import numpy as np

class GridMap:
  def __init__(self, width, height, obstacle_prob=0.0, seed=None):
    self.width = width
    self.height = height
    if seed is not None:
      np.random.seed(seed)
    self.grid = (np.random.rand(height, width) < obstacle_prob).astype(np.int32)

  def is_valid(self, x, y):
    return 0 <= x < self.width and 0 <= y < self.height

  def is_free(self, x, y):
    # hot path for every planner: inline bounds check and .item() (returns a Python int)
    return 0 <= x < self.width and 0 <= y < self.height and self.grid.item(y, x) == 0

  def set_cell(self, x, y, value):
    if self.is_valid(x, y):
      self.grid[y, x] = int(value)

  def clone(self):
    m = GridMap(self.width, self.height)
    m.grid = self.grid.copy()
    return m

  def neighbors(self, x, y, allow_diagonal=True):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
      dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in dirs:
      nx, ny = x + dx, y + dy
      if self.is_free(nx, ny):
        if allow_diagonal and abs(dx) == 1 and abs(dy) == 1:
          if not self.is_free(x, ny) or not self.is_free(nx, y):
            continue
        cost = 1.41421356 if abs(dx) == 1 and abs(dy) == 1 else 1.0
        yield (nx, ny), cost

class PerceptionWindow:
  def __init__(self, radius=5, metric='chebyshev'):
    self.radius = radius
    self.metric = metric

  def get_visible_cells(self, robot_pos, width, height):
    rx, ry = robot_pos
    r = self.radius
    x_min = max(0, rx - r)
    x_max = min(width - 1, rx + r)
    y_min = max(0, ry - r)
    y_max = min(height - 1, ry + r)
    cells = []
    for y in range(y_min, y_max + 1):
      for x in range(x_min, x_max + 1):
        if self.metric == 'euclidean':
          if (x - rx) ** 2 + (y - ry) ** 2 <= r ** 2:
            cells.append((x, y))
        else:
          cells.append((x, y))
    return cells

  def update_belief(self, robot_pos, true_map, belief_map):
    visible = self.get_visible_cells(robot_pos, true_map.width, true_map.height)
    changed = []
    for x, y in visible:
      tv = true_map.grid[y, x]
      if belief_map.grid[y, x] != tv:
        belief_map.grid[y, x] = tv
        changed.append(((x, y), tv))
    return changed

