"""D* Lite — optimized version (Koenig & Likhachev, AAAI 2002, Fig. 4) on an 8-connected grid.

Replaces the v1 implementation, which (a) recomputed rhs over all 8 successors for each of the
8 neighbours of every expanded vertex (~64 edge evaluations per expansion) and (b) returned
sub-optimal or missing paths in 27/1000 randomized change sequences
(tests/test_week2_dstar_differential.py).

Conventions
  * search runs backwards from goal; g/rhs are cost-to-goal; start moves with the robot (km term);
  * edge cost: 1 for straight moves, 1.41421356 for diagonals, inf if either endpoint is occupied
    or a diagonal cuts an occupied corner (identical rule to A* / GridMap.neighbors);
  * the graph is undirected, so Pred(u) == Succ(u);
  * a cell change alters the outgoing edge costs of the cell itself and of its 8 neighbours only
    (incident edges + diagonals that use it as a corner), so on_map_changed recomputes rhs there.

Floating-point safety: every cost is a sum of 1.0 and 1.41421356 (8 decimals), so g, rhs, km and keys
are rounded to 8 decimals. Without this, mathematically equal keys (e.g. 7.242640680000001 vs
7.24264068) made the termination test stop before an under-consistent vertex was processed —
the root cause of the 27/1000 wrong results, shared by the v1 implementation.

Work counters (hardware-independent): `expanded_nodes` per plan() call (returned) and
`last_edge_evals` (edge-cost evaluations in the last plan() call, incl. preceding map updates).
"""
import heapq
import time

INF = float('inf')
DIAG = 1.41421356
_DIRS8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, DIAG), (-1, 1, DIAG), (1, -1, DIAG), (1, 1, DIAG)]


def _r(x):
  return round(x, 8)  # round(inf) stays inf


def octile_dist(p1, p2):
  dx = abs(p1[0] - p2[0])
  dy = abs(p1[1] - p2[1])
  return _r((dx + dy) + (DIAG - 2.0) * min(dx, dy))


class DStarLite:
  def __init__(self, grid_map, start, goal, allow_diagonal=True):
    self.grid_map = grid_map
    self.start = start
    self.goal = goal
    self.last_node = start
    self.dirs = _DIRS8 if allow_diagonal else _DIRS8[:4]
    self.km = 0.0
    self.g = {}
    self.rhs = {goal: 0.0}
    self.U = []          # heap of (key, tiebreak, vertex); stale entries skipped lazily
    self.in_u = {}       # vertex -> current key
    self._tie = 0
    self.expanded_nodes = 0
    self.edge_evals = 0
    self.last_edge_evals = 0
    self._push(goal, self.calculate_key(goal))

  # ---------- basic helpers ----------
  def get_g(self, u):
    return self.g.get(u, INF)

  def get_rhs(self, u):
    return self.rhs.get(u, INF)

  def calculate_key(self, s):
    m = min(self.g.get(s, INF), self.rhs.get(s, INF))
    return (_r(m + octile_dist(self.start, s) + self.km), m)

  def _push(self, u, k):
    self._tie += 1
    heapq.heappush(self.U, (k, self._tie, u))
    self.in_u[u] = k

  def neighbors(self, u):
    x, y = u; w = self.grid_map.width; h = self.grid_map.height
    for dx, dy, _ in self.dirs:
      nx, ny = x + dx, y + dy
      if 0 <= nx < w and 0 <= ny < h:
        yield (nx, ny)

  def edge_cost(self, u, v):
    self.edge_evals += 1
    free = self.grid_map.is_free
    if not free(*u) or not free(*v):
      return INF
    dx = v[0] - u[0]; dy = v[1] - u[1]
    if dx != 0 and dy != 0:
      if not free(u[0], v[1]) or not free(v[0], u[1]):
        return INF
      return DIAG
    return 1.0

  def _recompute_rhs(self, u):
    if u == self.goal:
      return
    best = INF
    for s in self.neighbors(u):
      gs = self.g.get(s, INF)
      if gs == INF:
        continue
      c = self.edge_cost(u, s)
      if c + gs < best:
        best = c + gs
    self.rhs[u] = _r(best)

  def update_vertex(self, u):
    """Queue maintenance only (Fig. 4 UpdateVertex); rhs is set by the caller."""
    if self.g.get(u, INF) != self.rhs.get(u, INF):
      k = self.calculate_key(u)
      if self.in_u.get(u) != k:
        self._push(u, k)
    else:
      self.in_u.pop(u, None)

  # ---------- main loop ----------
  def _top(self):
    U, in_u = self.U, self.in_u
    while U and in_u.get(U[0][2]) != U[0][0]:
      heapq.heappop(U)
    return U[0] if U else None

  def compute_shortest_path(self):
    expanded = 0
    while True:
      top = self._top()
      if top is None:
        break
      k_old, _, u = top
      start = self.start
      if not (k_old < self.calculate_key(start) or self.get_rhs(start) != self.get_g(start)):
        break
      k_new = self.calculate_key(u)
      g_u = self.g.get(u, INF); rhs_u = self.rhs.get(u, INF)
      if k_old < k_new:
        self._push(u, k_new)                      # key was outdated (km grew): reinsert
      elif g_u > rhs_u:                           # over-consistent: lower g, relax predecessors
        heapq.heappop(self.U); self.in_u.pop(u, None)
        self.g[u] = rhs_u; expanded += 1
        for s in self.neighbors(u):
          if s != self.goal:
            c = self.edge_cost(s, u)
            v = _r(c + rhs_u)
            if v < self.rhs.get(s, INF):
              self.rhs[s] = v
          self.update_vertex(s)
      else:                                       # under-consistent: raise g, repair dependants
        heapq.heappop(self.U); self.in_u.pop(u, None)
        g_old = g_u
        self.g[u] = INF; expanded += 1
        for s in self.neighbors(u):
          if s != self.goal:
            c = self.edge_cost(s, u)
            r = self.rhs.get(s, INF)
            if r != INF and (c == INF or r == _r(c + g_old)):
              self._recompute_rhs(s)
          self.update_vertex(s)
        self._recompute_rhs(u)
        self.update_vertex(u)
    self.expanded_nodes += expanded
    return expanded

  def on_map_changed(self, changed_cells):
    """Call after the grid was updated; `self.start` must already be the robot's current cell."""
    self.km = _r(self.km + octile_dist(self.last_node, self.start))
    self.last_node = self.start
    affected = set()
    for (x, y), _ in changed_cells:
      affected.add((x, y))
      affected.update(self.neighbors((x, y)))
    for u in affected:
      self._recompute_rhs(u)
      self.update_vertex(u)

  def extract_path(self):
    if self.get_g(self.start) == INF:
      return []
    path = [self.start]; curr = self.start; visited = {curr}
    for _ in range(self.grid_map.width * self.grid_map.height):
      if curr == self.goal:
        return path
      best, best_cost = None, INF
      for s in self.neighbors(curr):
        gs = self.g.get(s, INF)
        if gs == INF and s != self.goal:
          continue
        c = self.edge_cost(curr, s) + (0.0 if s == self.goal else gs)
        if c < best_cost:
          best_cost, best = c, s
      if best is None or best in visited or best_cost == INF:
        return []
      curr = best; path.append(curr); visited.add(curr)
    return []

  def plan(self):
    t0 = time.perf_counter()
    e0 = self.edge_evals
    nodes = self.compute_shortest_path()
    path = self.extract_path()
    self.last_edge_evals = self.edge_evals - e0
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return path, nodes, dt_ms
