"""Runtime and memory of each strategy on this computer (read-only add-on; core code is not modified).

Two modes, run on the same episodes (default: evaluation seed 3, all 832 configurations):

  --mode memory   tracemalloc is started in every worker and the planner entry points are wrapped:
                  global A* (AStarPlanner.plan), D* Lite (DStarLite.plan, DStarLite.on_map_changed),
                  local repair (RiskAdaptivePlanner.try_local_repair, bounded / unbounded).
                  For each call we record the peak memory allocated above the level at call entry.
                  Nested calls (the A* inside a repair) are attributed to the outer call only.
                  We also record the memory a D* Lite instance retains after its initial plan.
  --mode runtime  no tracing (tracemalloc slows Python down and would distort timings). Records the
                  per-event milliseconds of each strategy, the per-episode wall time, the total wall
                  time and the peak working set of each worker process.

Both modes run the full paired evaluation of run_events.py (with the fitted predictor, so the same six
D* Lite shadows are maintained as in the main experiment).

Usage:
  python scripts/measure_resources.py --mode memory  --out results/v2_resources
  python scripts/measure_resources.py --mode runtime --out results/v2_resources
"""
import argparse, json, os, platform, sys, time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import run_events  # noqa: E402

PREDICTOR = os.path.join(BASE, 'results', 'v2_events', 'predictor.pkl')

# ------------------------------------------------------------------ per-process state
_REC = {}        # label -> list of peak bytes (per call)
_DEPTH = [0]


def _peak_working_set():
  """Peak working set of this process in bytes (Windows), or max RSS elsewhere."""
  if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
      _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                  ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                  ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                  ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                  ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return int(c.PeakWorkingSetSize)
  import resource
  return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _wrap(cls, name, label_fn):
  import tracemalloc
  orig = getattr(cls, name)

  def wrapper(self, *a, **k):
    if _DEPTH[0] > 0:
      return orig(self, *a, **k)
    _DEPTH[0] += 1
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
      return orig(self, *a, **k)
    finally:
      peak = tracemalloc.get_traced_memory()[1]
      _REC.setdefault(label_fn(self), []).append(max(0, peak - base))
      _DEPTH[0] -= 1

  setattr(cls, name, wrapper)


def _init_worker(mode):
  run_events._init_worker(PREDICTOR)
  if mode != 'memory':
    return
  import tracemalloc
  from src.planners.a_star import AStarPlanner
  from src.planners.d_star_lite import DStarLite
  from src.planners.risk_adaptive_planner import RiskAdaptivePlanner
  from src import scenarios_v2
  orig_build = scenarios_v2.build

  def build_untraced(*a, **k):        # scenario generation (reference-path A*) is not a strategy call
    _DEPTH[0] += 1
    try:
      return orig_build(*a, **k)
    finally:
      _DEPTH[0] -= 1

  scenarios_v2.build = build_untraced

  def dstar_label(s):                 # first plan() of an instance is a full search, not an event call
    if getattr(s, '_measured_initial', False):
      return 'dstar_plan'
    s._measured_initial = True
    return 'dstar_initial_plan'

  tracemalloc.start()
  _wrap(AStarPlanner, 'plan', lambda s: 'astar')
  _wrap(DStarLite, 'plan', dstar_label)
  _wrap(DStarLite, 'on_map_changed', lambda s: 'dstar_update')
  _wrap(RiskAdaptivePlanner, 'try_local_repair',
        lambda s: 'repair_bounded' if s.bounded_repair else 'repair_unbounded')


def _dstar_state_bytes(cfg):
  """Memory retained by one D* Lite instance after its initial plan on the episode's initial map."""
  import tracemalloc
  from src import scenarios_v2
  from src.planners.d_star_lite import DStarLite
  sc = scenarios_v2.build(cfg['map_type'], cfg['size'], cfg['density'], cfg['blockage'], cfg['seed'])
  if sc is None:
    return np.nan
  gm, s, g, _, _ = sc
  known = gm.clone()
  _DEPTH[0] += 1                      # keep the wrappers out of this measurement
  try:
    before = tracemalloc.get_traced_memory()[0]
    d = DStarLite(known, s, g); d.plan()
    after = tracemalloc.get_traced_memory()[0]
  finally:
    _DEPTH[0] -= 1
  del d
  return after - before


def work(args):
  cfg, mode = args
  _REC.clear()
  rows, summ = run_events.run_one(cfg)
  out = dict(cfg=cfg, built=summ.get('built', 0), n_events=len(rows), runtime_s=summ.get('runtime_s'),
             pid=os.getpid())
  if mode == 'memory':
    out['calls'] = {k: list(v) for k, v in _REC.items()}
    out['dstar_state'] = _dstar_state_bytes(cfg) if summ.get('built') else np.nan
  else:
    out['ms'] = {c: [r[c] for r in rows] for c in ('G_ms', 'D_ms', 'B_ms', 'U_ms', 'DF_ms', 'DUF_ms', 'pred_ms')
                 if rows and c in rows[0]}
    out['peak_ws'] = _peak_working_set()
  return out


def q(v, p):
  v = np.asarray([x for x in v if x == x], float)
  return float(np.percentile(v, p)) if v.size else float('nan')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--mode', choices=['memory', 'runtime'], required=True)
  ap.add_argument('--seeds', type=int, nargs='+', default=[3])
  ap.add_argument('--out', default=os.path.join(BASE, 'results', 'v2_resources'))
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
  ap.add_argument('--limit', type=int, default=None, help='smoke test: every n-th configuration only')
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  cfgs = list(run_events.configs(a.seeds, run_events.MOVINGAI))
  if a.limit:
    cfgs = cfgs[::a.limit]
  print(f'{a.mode}: {len(cfgs)} episodes on {a.workers} workers', flush=True)
  t0 = time.time(); res = []
  with Pool(a.workers, initializer=_init_worker, initargs=(a.mode,)) as pool:
    for i, r in enumerate(pool.imap_unordered(work, [(c, a.mode) for c in cfgs], chunksize=2)):
      res.append(r)
      if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{len(cfgs)}  {time.time() - t0:.0f}s', flush=True)
  wall = time.time() - t0
  res = [r for r in res if r['built']]

  summary = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), mode=a.mode,
                 seeds=a.seeds, n_episodes=len(res), n_events=int(sum(r['n_events'] for r in res)),
                 workers=a.workers, wall_s=round(wall, 1), python=platform.python_version(),
                 platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count())
  KiB = 1024.0
  if a.mode == 'memory':
    labels = sorted({k for r in res for k in r['calls']})
    stats = {}
    for lab in labels:
      v = [x for r in res for x in r['calls'].get(lab, [])]
      stats[lab] = dict(n_calls=len(v), median_kib=q(v, 50) / KiB, p95_kib=q(v, 95) / KiB, max_kib=max(v) / KiB)
    st = [r['dstar_state'] for r in res]
    stats['dstar_state_after_initial_plan'] = dict(n=len(st), median_kib=q(st, 50) / KiB, p95_kib=q(st, 95) / KiB,
                                                   max_kib=float(np.nanmax(st)) / KiB)
    summary['peak_alloc_per_call'] = stats
    by = []
    for r in res:
      row = {k: r['cfg'][k] for k in ('map_type', 'size', 'R', 'blockage')}
      for lab in labels:
        v = r['calls'].get(lab, [])
        row[f'{lab}_max_kib'] = (max(v) / KiB) if v else np.nan
      row['dstar_state_kib'] = r['dstar_state'] / KiB
      by.append(row)
    df = pd.DataFrame(by)
    df.to_csv(os.path.join(a.out, 'memory_episodes.csv'), index=False)
    summary['max_kib_by_size'] = df.groupby('size')[[c for c in df.columns if c.endswith('_kib')]].max().round(1).to_dict()
  else:
    cols = sorted({k for r in res for k in r['ms']})
    summary['ms_per_event'] = {c: dict(median=q([x for r in res for x in r['ms'].get(c, [])], 50),
                                       p95=q([x for r in res for x in r['ms'].get(c, [])], 95)) for c in cols}
    ep = [r['runtime_s'] for r in res]
    summary['episode_runtime_s'] = dict(median=q(ep, 50), p95=q(ep, 95), max=max(ep), total=float(sum(ep)))
    ws = {}
    for r in res:
      ws[r['pid']] = max(ws.get(r['pid'], 0), r['peak_ws'])
    summary['peak_working_set_mib_per_worker'] = dict(max=max(ws.values()) / 2**20,
                                                      median=float(np.median(list(ws.values()))) / 2**20)
    pd.DataFrame([dict(r['cfg'], n_events=r['n_events'], runtime_s=r['runtime_s']) for r in res]).to_csv(
        os.path.join(a.out, 'runtime_episodes.csv'), index=False)
  with open(os.path.join(a.out, f'summary_{a.mode}.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=1)
  print(json.dumps(summary, indent=1))


if __name__ == '__main__':
  main()
