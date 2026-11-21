"""Paired event-level experiment runner (plan §7), parallel over CPU cores.

  python run_events.py --pilot                 # 10% pilot (checkpoint 2)
  python run_events.py --seeds 0 1 2 ... --out results/v2_events/full

Outputs in --out: events.csv (one row per blocking event, all strategies on the same state),
episodes.csv (one row per episode), metadata.json (settings + code fingerprint).
Seeds with seed % 10 < 3 form the TRAINING split (threshold tuning, predictor fitting);
all other seeds are the held-out EVALUATION split.
"""
import argparse, itertools, json, os, platform, sys, time
from datetime import datetime, timezone
from multiprocessing import Pool
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

SIZES = [50, 100, 150, 200]
DENSITIES = [0.05, 0.15, 0.25]          # used by 'random' and 'rooms'; Moving AI maps have their own
RADII = [3, 5, 8, 12]
BLOCKAGES = ['point', 'wall', 'barrier', 'multi']
GENERATED = ['random', 'rooms']
MOVINGAI = ['room8', 'room16', 'room32', 'maze4', 'maze8', 'maze16', 'street']


def configs(seeds, families):
  for mt in GENERATED:
    for size, dens, R, bl, sd in itertools.product(SIZES, DENSITIES, RADII, BLOCKAGES, seeds):
      yield dict(map_type=mt, size=size, density=dens, R=R, blockage=bl, seed=sd)
  for mt in families:
    for size, R, bl, sd in itertools.product(SIZES, RADII, BLOCKAGES, seeds):
      yield dict(map_type=mt, size=size, density=0.0, R=R, blockage=bl, seed=sd)


_PREDICTOR = None


def _init_worker(predictor_path):
  global _PREDICTOR
  if predictor_path:
    import pickle
    with open(predictor_path, 'rb') as f:
      _PREDICTOR = pickle.load(f)


def split_of(seed):
  return 'train' if seed % 10 < 3 else 'eval'


def run_one(cfg):
  from src import scenarios_v2
  from src.event_eval import evaluate_episode
  t = time.perf_counter()
  sc = scenarios_v2.build(cfg['map_type'], cfg['size'], cfg['density'], cfg['blockage'], cfg['seed'])
  if sc is None:
    return [], dict(cfg, built=0)
  gm, s, g, ev, info = sc
  rows, summ = evaluate_episode(gm, s, g, ev, cfg['R'], predictor=_PREDICTOR)
  key = dict(cfg, map_name=info['map_name'], ref_len=info['ref_len'], n_block_cells=info['n_block_cells'],
             split=split_of(cfg['seed']))
  for i, r in enumerate(rows):
    r.update(key, event_idx=i)
  summ.update(key, built=1, runtime_s=round(time.perf_counter() - t, 3))
  return rows, summ


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--pilot', action='store_true', help='3 seeds (0,1,2) on all configurations')
  ap.add_argument('--seeds', type=int, nargs='+', default=None)
  ap.add_argument('--families', nargs='+', default=MOVINGAI)
  ap.add_argument('--out', default=None)
  ap.add_argument('--split', choices=['train', 'eval', 'all'], default='all',
                  help='keep only seeds of this split (seed %% 10 < 3 -> train)')
  ap.add_argument('--predictor', default=None, help='pickled RepairPredictor (fitted on train split)')
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
  a = ap.parse_args()
  seeds = a.seeds or ([0, 1, 2] if a.pilot else list(range(30)))
  if a.split != 'all':
    seeds = [sd for sd in seeds if split_of(sd) == a.split]
  out = a.out or os.path.join(BASE_DIR, 'results', 'v2_events', 'pilot' if a.pilot else 'full')
  os.makedirs(out, exist_ok=True)
  cfgs = list(configs(seeds, a.families))
  print(f'{len(cfgs)} episodes on {a.workers} workers -> {out}', flush=True)
  t = time.time(); rows, eps = [], []
  with Pool(a.workers, initializer=_init_worker, initargs=(a.predictor,)) as pool:
    for i, (r, s) in enumerate(pool.imap_unordered(run_one, cfgs, chunksize=4)):
      rows += r; eps.append(s)
      if (i + 1) % 200 == 0:
        print(f'  {i + 1}/{len(cfgs)} episodes, {len(rows)} events, {time.time() - t:.0f}s', flush=True)
  pd.DataFrame(rows).to_csv(os.path.join(out, 'events.csv'), index=False)
  pd.DataFrame(eps).to_csv(os.path.join(out, 'episodes.csv'), index=False)
  from run import code_fingerprint
  meta = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
              python=platform.python_version(), platform=platform.platform(),
              seeds=seeds, split=a.split, predictor=a.predictor, families=a.families, sizes=SIZES, densities=DENSITIES, radii=RADII,
              blockages=BLOCKAGES, n_episodes=len(eps), n_events=len(rows),
              runtime_s=round(time.time() - t, 1), code_sha256=code_fingerprint())
  with open(os.path.join(out, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
  print(f'done: {len(eps)} episodes, {len(rows)} events, {meta["runtime_s"]} s')


if __name__ == '__main__':
  main()
