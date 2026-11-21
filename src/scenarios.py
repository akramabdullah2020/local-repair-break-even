import numpy as np
from src.environment import GridMap
from src.planners.a_star import AStarPlanner

class ScenarioBuilder:
  @staticmethod
  def create_map(sid, width=30, height=30, seed=42):
    np.random.seed(seed)
    gm = GridMap(width, height)
    start, goal = (2, 2), (width-3, height-3)
    ev = []
    dens = {1:0.15, 2:0.12, 3:0.10, 4:0.12}
    if sid in dens:
      for _ in range(int(width*height*dens[sid])):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        if (x, y) not in [start, goal]:
          gm.set_cell(x, y, 1)
    gm.set_cell(*start, 0); gm.set_cell(*goal, 0)
    if sid in (3, 4):
      ref, _, _ = AStarPlanner().plan(gm, start, goal)
      sched = [(5, 4)] if sid == 3 else [(3, 3), (7, 3), (12, 4)]
      for step, off in sched:   # fixed cells taken from the reference path -> identical for all planners
        idx = min(len(ref)-2, (step-1)+off)
        ev.append({'step': step, 'add': [ref[idx]]})
    elif sid == 2:
      ev = [{'step': 3, 'add': [(width-5, 2), (width-6, 3)]}, {'step': 6, 'add': [(2, height-5), (3, height-6)]}]
    elif sid == 5:
      rng = np.random.RandomState(seed)
      w1, w2 = width//3, (2*width)//3
      for y in range(height):
        gm.set_cell(w1, y, 1); gm.set_cell(w2, y, 1)
      d1, d2 = int(rng.randint(10, 21)), int(rng.randint(10, 21))
      gm.set_cell(w1, d1, 0); gm.set_cell(w2, d2, 0)
      off = int(rng.randint(3, 10)); sign = 1 if rng.rand() < 0.5 else -1
      alt = d1 + sign*off
      if not (1 <= alt <= height-2): alt = d1 - sign*off
      # position-triggered: door closes when the robot is within 3 cells of it (same geometry for every planner);
      # the alternative opening appears `off` cells away from the door, visible only if R is large enough
      ev = [{'near': (w1, d1), 'dist': 3, 'add': [(w1, d1)], 'open_alternative': [(w1, alt)]}]
    return gm, start, goal, ev
