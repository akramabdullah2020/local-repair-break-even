import heapq
import time

def octile_distance(p1, p2):
  dx = abs(p1[0] - p2[0])
  dy = abs(p1[1] - p2[1])
  return (dx + dy) + (1.41421356 - 2.0) * min(dx, dy)

class AStarPlanner:
  def __init__(self, allow_diagonal=True):
    self.allow_diagonal = allow_diagonal

  def plan(self, grid_map, start, goal, bounds=None):
    """bounds=(x0, y0, x1, y1) inclusive restricts the search to a box (used by bounded local repair)."""
    t0 = time.perf_counter()
    if bounds is not None:
      x0, y0, x1, y1 = bounds
      if not (x0 <= start[0] <= x1 and y0 <= start[1] <= y1 and x0 <= goal[0] <= x1 and y0 <= goal[1] <= y1):
        return [], 0, (time.perf_counter() - t0) * 1000.0
    if not grid_map.is_free(*start) or not grid_map.is_free(*goal):
      return [], 0, (time.perf_counter() - t0) * 1000.0
    open_set = []
    heapq.heappush(open_set, (0.0, 0, start))
    came_from = {}
    g_score = {start: 0.0}
    expanded_nodes = 0
    entry_count = 0
    closed = set()
    while open_set:
      _, _, current = heapq.heappop(open_set)
      if current in closed:
        continue
      closed.add(current)
      if current == goal:
        path = []
        curr = current
        while curr in came_from:
          path.append(curr)
          curr = came_from[curr]
        path.append(start)
        path.reverse()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return path, expanded_nodes, dt_ms
      expanded_nodes += 1
      for neighbor, cost in grid_map.neighbors(*current, allow_diagonal=self.allow_diagonal):
        if bounds is not None and not (x0 <= neighbor[0] <= x1 and y0 <= neighbor[1] <= y1):
          continue
        tentative_g = g_score[current] + cost
        if neighbor not in g_score or tentative_g < g_score[neighbor]:
          came_from[neighbor] = current
          g_score[neighbor] = tentative_g
          f_score = tentative_g + octile_distance(neighbor, goal)
          entry_count += 1
          heapq.heappush(open_set, (f_score, entry_count, neighbor))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return [], expanded_nodes, dt_ms

