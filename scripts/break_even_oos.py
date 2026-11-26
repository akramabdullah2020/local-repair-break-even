"""Out-of-sample validation of the break-even model (read-only add-on analysis).

The in-sample check in analyze_v2.py estimates p_fail, C_L and C_G from the same held-out
events on which the winner is measured. Here the three quantities are estimated per cell on
the TRAINING split only, and the prediction is scored against the winner measured on the
HELD-OUT split. Existing event tables are only read; outputs go to a new directory.

Cells: map_type x size x R x blockage (as in the paper). A cell is kept when it has at least
`--min-n` events in both splits.

Also reported:
  * in-sample agreement recomputed here (sanity check: must match the paper, 97.6% / 90.8%);
  * a model-free baseline: predict the held-out winner by the winner measured in training;
  * the cost of acting on the out-of-sample prediction (per-cell choice between repair-first
    and the fallback) on held-out episodes, relative to D* Lite at every event.

Usage:
  python scripts/break_even_oos.py --train results/v2_events/full_train \
      --eval results/v2_events/full_eval --out results/v2_oos
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.policies import exact_policy_costs  # noqa: E402

CELL = ['map_type', 'size', 'R', 'blockage']
EPISODE = ['map_type', 'size', 'density', 'R', 'blockage', 'seed']


def wilson(k, n, z=1.959964):
  if n == 0:
    return (float('nan'), float('nan'))
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (float(c - h), float(c + h))


def cell_table(ev):
  """Per-cell break-even inputs and measured mean costs (same definitions as analyze_v2.py)."""
  c = exact_policy_costs(ev)
  e = pd.concat([ev[CELL + ['B_ok', 'B_edges', 'G_edges', 'D_edges']],
                 c[['AR_ASTAR', 'G_ASTAR', 'AR_DSTAR', 'G_DSTAR']]], axis=1)
  g = e.groupby(CELL)
  t = pd.DataFrame({'n': g.size(), 'p_fail': 1 - g['B_ok'].mean(), 'C_L': g['B_edges'].mean(),
                    'C_A': g['G_edges'].mean(), 'C_D': g['D_edges'].mean(),
                    'AR_A': g['AR_ASTAR'].mean(), 'G_A': g['G_ASTAR'].mean(),
                    'AR_D': g['AR_DSTAR'].mean(), 'G_D': g['G_DSTAR'].mean()})
  for fb, C in [('astar', 'C_A'), ('dstar', 'C_D')]:
    t[f'p_star_{fb}'] = 1 - t['C_L'] / t[C]
    t[f'pred_{fb}'] = t['p_fail'] < t[f'p_star_{fb}']
  t['true_astar'] = t['AR_A'] < t['G_A']
  t['true_dstar'] = t['AR_D'] < t['G_D']
  return t


def agreement(pred, true, w):
  ok = (pred.values == true.values)
  k, n = int(ok.sum()), int(len(ok))
  lo, hi = wilson(k, n)
  return dict(cells=n, correct=k, agreement=k / n if n else float('nan'), ci95_wilson=[lo, hi],
              weighted_by_events=float(np.average(ok, weights=w)) if n else float('nan'))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--train', default='results/v2_events/full_train')
  ap.add_argument('--eval', default='results/v2_events/full_eval')
  ap.add_argument('--out', default='results/v2_oos')
  ap.add_argument('--min-n', type=int, default=5)
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)

  tr = pd.read_csv(os.path.join(a.train, 'events.csv'))
  ev = pd.read_csv(os.path.join(a.eval, 'events.csv'))
  assert set(tr['seed']).isdisjoint(set(ev['seed'])), 'train and eval seeds overlap'

  T, E = cell_table(tr), cell_table(ev)
  E_ins = E[E['n'] >= a.min_n]  # in-sample set used in the paper
  both = T[T['n'] >= a.min_n].join(E[E['n'] >= a.min_n], lsuffix='_tr', rsuffix='_ev', how='inner')

  S = {'min_n': a.min_n, 'n_train_events': int(len(tr)), 'n_eval_events': int(len(ev)),
       'train_seeds': sorted(int(s) for s in tr['seed'].unique()),
       'eval_seeds': sorted(int(s) for s in ev['seed'].unique()),
       'cells_eval': int(len(E_ins)), 'cells_both_splits': int(len(both))}

  for fb in ['astar', 'dstar']:
    S[f'in_sample_{fb}'] = agreement(E_ins[f'pred_{fb}'], E_ins[f'true_{fb}'], E_ins['n'])
    S[f'out_of_sample_{fb}'] = agreement(both[f'pred_{fb}_tr'], both[f'true_{fb}_ev'], both['n_ev'])
    S[f'train_winner_baseline_{fb}'] = agreement(both[f'true_{fb}_tr'], both[f'true_{fb}_ev'], both['n_ev'])
    wrong = both[f'pred_{fb}_tr'] != both[f'true_{fb}_ev']
    S[f'out_of_sample_{fb}']['errors_pred_repair_but_fallback_cheaper'] = int((wrong & both[f'pred_{fb}_tr']).sum())
    S[f'out_of_sample_{fb}']['errors_pred_fallback_but_repair_cheaper'] = int((wrong & ~both[f'pred_{fb}_tr']).sum())
    # how much the inputs move between splits (stability of the cell estimates)
    S[f'abs_diff_p_star_{fb}_median'] = float((both[f'p_star_{fb}_tr'] - both[f'p_star_{fb}_ev']).abs().median())
  S['abs_diff_p_fail_median'] = float((both['p_fail_tr'] - both['p_fail_ev']).abs().median())

  # ---- acting on the out-of-sample prediction: per-cell choice, scored on held-out episodes
  c = exact_policy_costs(ev)
  e = pd.concat([ev[EPISODE], c[['G_DSTAR', 'AR_DSTAR', 'G_ASTAR', 'AR_ASTAR']]], axis=1)
  choice = both[['pred_dstar_tr', 'pred_astar_tr', 'true_dstar_ev', 'true_astar_ev']].reset_index()
  e = e.merge(choice, on=CELL, how='inner')
  rows = {}
  for fb, G, AR in [('dstar', 'G_DSTAR', 'AR_DSTAR'), ('astar', 'G_ASTAR', 'AR_ASTAR')]:
    ref = e['G_DSTAR'].sum()
    rows[fb] = {
      'events_scored': int(len(e)),
      'always_fallback': float(e[G].sum() / ref),
      'always_repair_first': float(e[AR].sum() / ref),
      'break_even_oos_choice': float(np.where(e[f'pred_{fb}_tr'], e[AR], e[G]).sum() / ref),
      'cell_oracle_eval': float(np.where(e[f'true_{fb}_ev'], e[AR], e[G]).sum() / ref),
    }
  # model-free baseline choice (winner measured on training cells)
  base = both[['true_dstar_tr', 'true_astar_tr']].reset_index()
  e = e.merge(base, on=CELL, how='left')
  for fb, G, AR in [('dstar', 'G_DSTAR', 'AR_DSTAR'), ('astar', 'G_ASTAR', 'AR_ASTAR')]:
    rows[fb]['train_winner_choice'] = float(np.where(e[f'true_{fb}_tr'], e[AR], e[G]).sum() / e['G_DSTAR'].sum())
  S['policy_cost_vs_dstar_every_event'] = rows

  both.reset_index().to_csv(os.path.join(a.out, 'break_even_oos_cells.csv'), index=False)
  with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(S, f, indent=1)
  print(json.dumps(S, indent=1))


if __name__ == '__main__':
  main()
