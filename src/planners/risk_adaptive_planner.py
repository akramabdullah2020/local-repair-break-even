import time
from src.planners.a_star import AStarPlanner, octile_distance

class RiskAdaptivePlanner:
  def __init__(self, risk_threshold=0.5, w_ratio=0.5, w_prox=0.25, w_bottle=0.25, bounded_repair=False):
    # bounded_repair=True restricts the repair A* to the perception window around the robot
    # (cost bounded by the window, may fail if the detour leaves it); False = v1 behaviour
    # (A* between the splice points over the whole known map).
    self.bounded_repair = bounded_repair
    self.risk_threshold = risk_threshold
    self.w_ratio = w_ratio
    self.w_prox = w_prox
    self.w_bottle = w_bottle
    self.a_star = AStarPlanner()
    self.local_repair_attempts = 0
    self.local_repair_failures = 0

  def compute_risk(self, remaining_path, changed_cells, belief_map, perception_radius):
    if not remaining_path or not changed_cells:
      return 0.0, 0.0, 0.0, 0.0
    path_set = set(remaining_path)
    blocked_count = sum(1 for node in remaining_path if not belief_map.is_free(*node))
    ratio = blocked_count / max(1, len(remaining_path))
    prox_count = 0
    for (cx, cy), val in changed_cells:
      if val == 1:
        min_d = min(octile_distance((cx, cy), p) for p in remaining_path)
        if min_d <= 2.0:
          prox_count += 1
    prox_score = min(1.0, prox_count / max(1, perception_radius * 2))
    bottle_score = 0.0
    for (cx, cy), val in changed_cells:
      if val == 1 and (cx, cy) in path_set:
        adj_blocked = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
          nx, ny = cx + dx, cy + dy
          if not belief_map.is_valid(nx, ny) or not belief_map.is_free(nx, ny):
            adj_blocked += 1
        if adj_blocked >= 2:
          bottle_score = 1.0
          break
        bottle_score = max(bottle_score, adj_blocked / 4.0)
    risk = self.w_ratio * ratio + self.w_prox * prox_score + self.w_bottle * bottle_score
    return risk, ratio, prox_score, bottle_score

  def is_path_valid(self, path, belief_map):
    if not path:
      return False
    for i, pt in enumerate(path):
      if not belief_map.is_free(*pt):
        return False
      if i > 0:
        prev = path[i - 1]
        dx = abs(pt[0] - prev[0])
        dy = abs(pt[1] - prev[1])
        if max(dx, dy) != 1:  # consecutive points must be 8-neighbours (rejects jumps and duplicates)
          return False
        if dx == 1 and dy == 1:
          if not belief_map.is_free(prev[0], pt[1]) or not belief_map.is_free(pt[0], prev[1]):
            return False
    return True

  @staticmethod
  def invalid_indices(path, belief_map):
    """Indices i where cell path[i] is occupied, or the move path[i-1]->path[i] cuts an occupied corner."""
    bad = []
    for i, pt in enumerate(path):
      if not belief_map.is_free(*pt):
        bad.append(i); continue
      if i > 0:
        prev = path[i - 1]
        if abs(pt[0] - prev[0]) == 1 and abs(pt[1] - prev[1]) == 1:
          if not belief_map.is_free(prev[0], pt[1]) or not belief_map.is_free(pt[0], prev[1]):
            bad.append(i)
    return bad

  def try_local_repair(self, current_pos, current_path, belief_map, perception_radius):
    self.local_repair_attempts += 1
    t0 = time.perf_counter()
    if not current_path:
      self.local_repair_failures += 1
      return None, 0, (time.perf_counter() - t0) * 1000.0
    blocked_indices = self.invalid_indices(current_path, belief_map)
    if not blocked_indices:
      dt = (time.perf_counter() - t0) * 1000.0
      return current_path, 0, dt
    first_b = blocked_indices[0]
    last_b = blocked_indices[-1]
    splice_start_idx = max(0, first_b - 1)
    splice_start = current_path[splice_start_idx]
    if not belief_map.is_free(*splice_start):
      splice_start = current_pos
      splice_start_idx = 0
    splice_end_idx = None
    for j in range(last_b + 1, len(current_path)):
      cand = current_path[j]
      dist_to_robot = max(abs(cand[0] - current_pos[0]), abs(cand[1] - current_pos[1]))
      if dist_to_robot <= perception_radius and belief_map.is_free(*cand):
        splice_end_idx = j
        break
    if splice_end_idx is None:
      self.local_repair_failures += 1
      dt = (time.perf_counter() - t0) * 1000.0
      return None, 0, dt
    splice_end = current_path[splice_end_idx]
    bounds = None
    if self.bounded_repair:
      r = perception_radius
      bounds = (current_pos[0] - r, current_pos[1] - r, current_pos[0] + r, current_pos[1] + r)
    local_path, nodes, _ = self.a_star.plan(belief_map, splice_start, splice_end, bounds=bounds)
    if not local_path:
      self.local_repair_failures += 1
      dt = (time.perf_counter() - t0) * 1000.0
      return None, nodes, dt
    reconstructed = current_path[:splice_start_idx] + local_path + current_path[splice_end_idx + 1:]
    if not self.is_path_valid(reconstructed, belief_map):
      self.local_repair_failures += 1
      dt = (time.perf_counter() - t0) * 1000.0
      return None, nodes, dt
    dt = (time.perf_counter() - t0) * 1000.0
    return reconstructed, nodes, dt

  def replan(self, current_pos, goal, current_path, changed_cells, belief_map, perception_radius):
    risk, ratio, prox, bottle = self.compute_risk(current_path, changed_cells, belief_map, perception_radius)
    nodes = 0
    dt_ms = 0.0
    path_blocked = bool(self.invalid_indices(current_path, belief_map))
    if not path_blocked and risk < self.risk_threshold:
      return current_path, 'reused', 0, 0.0, risk
    if risk < self.risk_threshold:
      repaired_path, r_nodes, r_dt = self.try_local_repair(current_pos, current_path, belief_map, perception_radius)
      nodes += r_nodes
      dt_ms += r_dt
      if repaired_path is not None:
        return repaired_path, 'local_repair', nodes, dt_ms, risk
    t0 = time.perf_counter()
    broader_path, b_nodes, _ = self.a_star.plan(belief_map, current_pos, goal)
    dt_ms += (time.perf_counter() - t0) * 1000.0
    nodes += b_nodes
    return broader_path, 'broader_replan', nodes, dt_ms, risk
