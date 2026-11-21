"""Reader for Moving AI Lab grid benchmark maps (Sturtevant, IEEE TCIAIG 2012).

Format: 'type octile' / 'height H' / 'width W' / 'map' followed by H rows of W characters.
Passable: '.', 'G', 'S'. Everything else ('@', 'O', 'T', 'W') is treated as an obstacle.

Maps are stored under data/movingai/{room,maze,street}/ (downloaded from
https://movingai.com/benchmarks/grids.html). Large maps are cropped to a size x size window.
"""
import os
from collections import deque
import numpy as np
from src.environment import GridMap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(BASE_DIR, 'data', 'movingai')
PASSABLE = set('.GS')

# map families used in the study: (family, sub-folder, file prefix)
FAMILIES = {
  'room8': ('room', '8room_'), 'room16': ('room', '16room_'), 'room32': ('room', '32room_'),
  'maze4': ('maze', 'maze512-4-'), 'maze8': ('maze', 'maze512-8-'), 'maze16': ('maze', 'maze512-16-'),
  'street': ('street', None),   # *_256.map city maps
}


def load_map(path):
  with open(path, 'r', encoding='ascii') as f:
    lines = f.read().split('\n')
  if not lines[0].startswith('type'):
    raise ValueError(f'not a Moving AI map: {path}')
  h = int(lines[1].split()[1]); w = int(lines[2].split()[1])
  rows = lines[4:4 + h]
  grid = np.ones((h, w), dtype=np.int32)
  for y, row in enumerate(rows):
    for x, ch in enumerate(row[:w]):
      if ch in PASSABLE:
        grid[y, x] = 0
  gm = GridMap(w, h); gm.grid = grid
  return gm


def family_files(family):
  sub, prefix = FAMILIES[family]
  d = os.path.join(MAP_DIR, sub)
  if not os.path.isdir(d):
    raise FileNotFoundError(f'{d} missing: download the Moving AI {sub} maps first (see data/movingai/README.md)')
  files = sorted(f for f in os.listdir(d) if f.endswith('.map'))
  if prefix:
    files = [f for f in files if f.startswith(prefix)]
  else:
    files = [f for f in files if f.endswith('_256.map')]
  return [os.path.join(d, f) for f in files]


def largest_component(gm):
  """Cells of the largest 4-connected free component (4-connectivity is conservative w.r.t. corner rule)."""
  h, w = gm.height, gm.width
  seen = np.zeros((h, w), dtype=bool); best = []
  g = gm.grid
  for sy in range(h):
    for sx in range(w):
      if g[sy, sx] or seen[sy, sx]:
        continue
      comp = []; q = deque([(sx, sy)]); seen[sy, sx] = True
      while q:
        x, y = q.popleft(); comp.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
          if 0 <= nx < w and 0 <= ny < h and not g[ny, nx] and not seen[ny, nx]:
            seen[ny, nx] = True; q.append((nx, ny))
      if len(comp) > len(best):
        best = comp
  return best


def crop(gm, size, rng, min_free=0.3, tries=50):
  """Random size x size window whose largest free component covers >= min_free of the window."""
  if gm.width <= size and gm.height <= size:
    return gm
  best = None
  for _ in range(tries):
    x0 = int(rng.randint(0, gm.width - size + 1)); y0 = int(rng.randint(0, gm.height - size + 1))
    sub = GridMap(size, size); sub.grid = gm.grid[y0:y0 + size, x0:x0 + size].copy()
    frac = (sub.grid == 0).mean()
    if frac >= min_free:
      lcc = len(largest_component(sub))
      if lcc >= min_free * size * size:
        return sub
      if best is None or lcc > best[0]:
        best = (lcc, sub)
  return best[1] if best else sub


def load_family_map(family, size, rng):
  files = family_files(family)
  path = files[int(rng.randint(0, len(files)))]
  return crop(load_map(path), size, rng), os.path.basename(path)
