"""Benchmark runner (v2).

Default settings reproduce the configuration of results/fixed_results.csv
(5 scenarios x 20 seeds x 6 radii x 4 planner configs = 2400 runs, 30x30 maps).

Usage:
  python run.py                         # full default benchmark -> results/v2/
  python run.py --quick                 # 2 seeds x 2 radii smoke test
  python run.py --out results/v2_test --seeds 42 101 --radii 5 --size 50
"""
import argparse, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from src.scenarios import ScenarioBuilder
from src.simulation import SimulationRunner

DEFAULT_SEEDS = [42, 101, 202, 303, 404] + list(range(500, 800, 20))
DEFAULT_RADII = [3, 4, 5, 6, 8, 10]
PLANNER_CFG = [('A_STAR', 'A_STAR', None), ('D_STAR_LITE', 'D_STAR_LITE', None),
               ('RISK_0.45', 'RISK', 0.45), ('RISK_0.20', 'RISK', 0.20)]


def code_fingerprint():
  h = {}
  for root, _, files in os.walk(os.path.join(BASE_DIR, 'src')):
    for f in sorted(files):
      if f.endswith('.py'):
        p = os.path.join(root, f)
        with open(p, 'rb') as fh:
          h[os.path.relpath(p, BASE_DIR).replace('\\', '/')] = hashlib.sha256(fh.read()).hexdigest()[:16]
  return dict(sorted(h.items()))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(BASE_DIR, 'results', 'v2'))
  ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
  ap.add_argument('--radii', type=int, nargs='+', default=DEFAULT_RADII)
  ap.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3, 4, 5])
  ap.add_argument('--size', type=int, default=30)
  ap.add_argument('--max-steps', type=int, default=200)
  ap.add_argument('--quick', action='store_true')
  a = ap.parse_args()
  if a.quick:
    a.seeds, a.radii = a.seeds[:2], [3, 5]
  os.makedirs(a.out, exist_ok=True)

  rows, events = [], []; t = time.time(); skipped = 0
  for sc in a.scenarios:
    for sd in a.seeds:
      for r in a.radii:
        for name, pt, th in PLANNER_CFG:
          m, s, g, ev = ScenarioBuilder.create_map(sc, a.size, a.size, sd)
          res = SimulationRunner(r, a.max_steps).run(pt, m, s, g, ev, thr=th or 0.45, log_events=True)
          if res is None:
            skipped += 1; continue
          key = dict(scenario=sc, seed=sd, radius=r, size=a.size, planner=name)
          for e in res.pop('events'):
            events.append({**key, **e})
          res.pop('trail')
          res.update(key); rows.append(res)

  runs = pd.DataFrame(rows); evs = pd.DataFrame(events)
  runs.to_csv(os.path.join(a.out, 'runs.csv'), index=False)
  evs.to_csv(os.path.join(a.out, 'events.csv'), index=False)
  meta = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
              python=platform.python_version(), platform=platform.platform(),
              args={**vars(a), 'out': os.path.relpath(a.out, BASE_DIR).replace(os.sep, '/')},
              n_runs=len(runs), n_events=len(evs), skipped_no_initial_path=skipped,
              runtime_s=round(time.time() - t, 1), code_sha256=code_fingerprint())
  with open(os.path.join(a.out, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
  print(f"{len(runs)} runs, {len(evs)} blocking events, {skipped} skipped, {meta['runtime_s']} s -> {a.out}")
  bad = runs[runs.invalid_reason.fillna('') != '']
  if len(bad):
    print('WARNING invalid moves:', bad.invalid_reason.value_counts().to_dict())


if __name__ == '__main__':
  main()
