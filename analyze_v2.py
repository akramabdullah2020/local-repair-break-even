"""Statistical analysis and figures for the v2 study (plan §7, research questions Q1–Q4).

  python analyze_v2.py --eval results/v2_events/full_eval --train results/v2_events/full_train

All hypothesis tests use the held-out EVALUATION split. The unit of analysis for policy comparisons
is the episode (events inside an episode are not independent): per-episode summed cost.
Outputs: results/v2_analysis/{tables/*.csv, figures/*.png, summary.json}
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

from src.policies import exact_policy_costs, break_even_table, FEATURES

# ---- palette (dataviz reference palette, light mode; validated: 4 slots pass CVD/normal-vision,
#      slots 3-4 need direct labels -> every line is direct-labelled)
SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
DIV = LinearSegmentedColormap.from_list('div', ['#184f95', '#6da7ec', '#f0efec', '#ec8a89', '#a83232'])

plt.rcParams.update({
  'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
  'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
  'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
  'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
  'axes.titleweight': 'bold', 'lines.linewidth': 2, 'legend.frameon': False, 'axes.axisbelow': True,
})

POLICY_ORDER = ['G_ASTAR', 'G_DSTAR', 'AR_ASTAR', 'AR_DSTAR', 'ARU_ASTAR', 'ARU_DSTAR', 'RISK_DSTAR',
                'PRED_DSTAR', 'PREDTAU_DSTAR', 'ORACLE_DSTAR', 'ORACLE_U_DSTAR']
POLICY_LABEL = {
  'G_ASTAR': 'A* every event', 'G_DSTAR': 'D* Lite every event (baseline)',
  'AR_ASTAR': 'Repair first → A*', 'AR_DSTAR': 'Repair first → D* Lite',
  'ARU_ASTAR': 'Unbounded repair → A*', 'ARU_DSTAR': 'Unbounded repair → D* Lite',
  'RISK_DSTAR': 'Risk gate (v1) → D* Lite', 'PRED_DSTAR': 'Predictor + break-even rule → D* Lite',
  'PREDTAU_DSTAR': 'Predictor + tuned threshold → D* Lite', 'ORACLE_DSTAR': 'Oracle, bounded repair → D* Lite',
  'ORACLE_U_DSTAR': 'Oracle, unbounded repair → D* Lite',
}


# ------------------------------------------------------------------ statistics helpers
def holm(pvals):
  p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
  for i, k in enumerate(order):
    run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
  return adj


def rank_biserial(x, y):
  """Matched-pairs rank-biserial correlation (effect size for Wilcoxon signed-rank); >0 means x > y."""
  d = np.asarray(x) - np.asarray(y); d = d[d != 0]
  if len(d) == 0:
    return 0.0
  r = stats.rankdata(np.abs(d))
  return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cluster_bootstrap_ratio(ep, num, den, n_boot=2000, seed=0):
  rng = np.random.default_rng(seed); a = ep[num].values; b = ep[den].values; n = len(a)
  idx = rng.integers(0, n, size=(n_boot, n))
  ratios = a[idx].sum(1) / b[idx].sum(1)
  return float(a.sum() / b.sum()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def logit_with_se(X, y, names):
  """Unpenalised logistic regression with Wald standard errors (no statsmodels dependency)."""
  X = np.c_[np.ones(len(X)), X]; names = ['intercept'] + names
  def nll(b):
    z = X @ b; return np.sum(np.logaddexp(0, z) - y * z)
  def grad(b):
    p = 1 / (1 + np.exp(-(X @ b))); return X.T @ (p - y)
  b = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method='BFGS', options={'maxiter': 2000}).x
  p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
  cov = np.linalg.pinv((X * W[:, None]).T @ X); se = np.sqrt(np.diag(cov))
  zv = b / se; pv = 2 * stats.norm.sf(np.abs(zv))
  return pd.DataFrame({'term': names, 'coef': b, 'se': se, 'odds_ratio': np.exp(b),
                       'or_ci_low': np.exp(b - 1.96 * se), 'or_ci_high': np.exp(b + 1.96 * se), 'p': pv})


# ------------------------------------------------------------------ analysis
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--eval', default='results/v2_events/full_eval')
  ap.add_argument('--train', default='results/v2_events/full_train')
  ap.add_argument('--out', default='results/v2_analysis')
  a = ap.parse_args()
  T = os.path.join(a.out, 'tables'); F = os.path.join(a.out, 'figures')
  os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)

  ev = pd.read_csv(os.path.join(a.eval, 'events.csv'))
  eps = pd.read_csv(os.path.join(a.eval, 'episodes.csv'))
  tr = pd.read_csv(os.path.join(a.train, 'events.csv')) if os.path.exists(os.path.join(a.train, 'events.csv')) else None
  assert (ev['split'] == 'eval').all()
  ev['family'] = np.where(ev['map_type'].isin(['random', 'rooms']), 'generated', 'movingai')
  S = {}
  S['n_eval_episodes'] = int(len(eps)); S['n_eval_events'] = int(len(ev))
  S['eval_success_rate'] = float(eps['success'].mean()); S['eval_collisions'] = int(eps['collision'].sum())
  S['dstar_cost_mismatch_events'] = int((np.abs(ev['D_cost'] - ev['G_cost']) > 1e-6).sum())

  costs = exact_policy_costs(ev)
  pols = [p for p in POLICY_ORDER if p in costs]
  ev_c = pd.concat([ev, costs.add_prefix('cost_')], axis=1)
  ep_key = ['map_type', 'size', 'density', 'R', 'blockage', 'seed']
  ep = ev_c.groupby(ep_key)[[f'cost_{p}' for p in pols]].sum().reset_index()
  ep.columns = ep_key + pols

  # ---- Q3: policy comparison (episode-level, held-out)
  rows = []
  for p in pols:
    r, lo, hi = cluster_bootstrap_ratio(ep, p, 'G_DSTAR')
    rows.append(dict(policy=p, label=POLICY_LABEL[p], ratio_vs_dstar=r, ci_low=lo, ci_high=hi,
                     total_edges=float(ep[p].sum())))
  pol_tab = pd.DataFrame(rows); pol_tab.to_csv(os.path.join(T, 'policy_costs_vs_dstar.csv'), index=False)

  tests = [('AR_DSTAR', 'G_DSTAR'), ('PRED_DSTAR', 'G_DSTAR'), ('PREDTAU_DSTAR', 'G_DSTAR'),
           ('ARU_DSTAR', 'G_DSTAR'), ('RISK_DSTAR', 'AR_DSTAR'), ('PRED_DSTAR', 'AR_DSTAR'),
           ('PRED_DSTAR', 'ORACLE_DSTAR'), ('G_DSTAR', 'G_ASTAR'), ('ARU_DSTAR', 'AR_DSTAR'),
           ('ARU_DSTAR', 'PRED_DSTAR')]
  tests = [t for t in tests if t[0] in ep and t[1] in ep]
  tr_rows = []
  for x, y in tests:
    d = ep[x] - ep[y]
    w = stats.wilcoxon(ep[x], ep[y], zero_method='wilcox') if (d != 0).any() else None
    tr_rows.append(dict(policy=x, versus=y, n_episodes=len(ep), median_diff=float(d.median()),
                        share_policy_cheaper=float((d < 0).mean()), share_equal=float((d == 0).mean()),
                        rank_biserial=rank_biserial(ep[x], ep[y]), p=float(w.pvalue) if w else 1.0))
  tt = pd.DataFrame(tr_rows); tt['p_holm'] = holm(tt['p']); tt.to_csv(os.path.join(T, 'policy_tests.csv'), index=False)

  # ---- Q1: what drives repair failure (logistic regression on design factors, eval split)
  X = pd.get_dummies(ev[['blockage', 'family']], drop_first=True).astype(float)
  X['log2_size'] = np.log2(ev['size']); X['R'] = ev['R']
  lg = logit_with_se(X.values, (1 - ev['B_ok']).values.astype(float), list(X.columns))
  lg.to_csv(os.path.join(T, 'repair_failure_logit.csv'), index=False)

  # ---- break-even tables by single factor + full cells
  for by in ['blockage', 'map_type', 'size', 'R']:
    break_even_table(ev, by).to_csv(os.path.join(T, f'break_even_by_{by}.csv'))

  # ---- Q2: does the break-even model predict where repair-first pays? (cells: map_type x size x R x blockage)
  cell_key = ['map_type', 'size', 'R', 'blockage']
  cg = ev_c.groupby(cell_key)
  cells = pd.DataFrame({'n': cg.size(), 'p_fail': 1 - cg['B_ok'].mean(), 'C_L': cg['B_edges'].mean(),
                        'C_A': cg['G_edges'].mean(), 'C_D': cg['D_edges'].mean(),
                        'AR_A': cg['cost_AR_ASTAR'].mean(), 'G_A': cg['cost_G_ASTAR'].mean(),
                        'AR_D': cg['cost_AR_DSTAR'].mean(), 'G_D': cg['cost_G_DSTAR'].mean()}).reset_index()
  cells = cells[cells['n'] >= 5]
  for fb, C in [('astar', 'C_A'), ('dstar', 'C_D')]:
    cells[f'p_star_{fb}'] = 1 - cells['C_L'] / cells[C]
    cells[f'pred_pays_{fb}'] = cells['p_fail'] < cells[f'p_star_{fb}']
  cells['true_pays_astar'] = cells['AR_A'] < cells['G_A']
  cells['true_pays_dstar'] = cells['AR_D'] < cells['G_D']
  cells.to_csv(os.path.join(T, 'break_even_cells.csv'), index=False)
  for fb in ['astar', 'dstar']:
    agree = (cells[f'pred_pays_{fb}'] == cells[f'true_pays_{fb}'])
    S[f'break_even_agreement_{fb}'] = dict(
      cells=int(len(cells)), agreement=float(agree.mean()),
      weighted_by_events=float(np.average(agree, weights=cells['n'])),
      share_cells_repair_pays=float(cells[f'true_pays_{fb}'].mean()))

  # ---- predictor quality (eval split)
  if 'pred_p' in ev:
    yf = 1 - ev['B_ok']
    S['predictor'] = dict(auc=float(roc_auc_score(yf, ev['pred_p'])), brier=float(brier_score_loss(yf, ev['pred_p'])),
                          auc_risk_v1=float(roc_auc_score(yf, ev['risk'])),
                          share_events_risk_over_045=float((ev['risk'] >= 0.45).mean()),
                          base_rate_fail=float(yf.mean()),
                          pred_ms_median=float(ev['pred_ms'].median()), risk_ms_median=float(ev['risk_ms'].median()))

  # ---- Q4: path quality of successful bounded repair
  for tag in ['B', 'U']:
    okb = ev[(ev[f'{tag}_ok'] == 1) & (ev['G_ok'] == 1)]
    rel = okb[f'{tag}_dL'] / okb['G_cost'].clip(lower=1e-9) * 100
    S[f'path_increase_pct_{tag}'] = dict(n=int(len(okb)), mean=float(rel.mean()), median=float(rel.median()),
                                         p90=float(rel.quantile(0.9)), share_zero=float((okb[f'{tag}_dL'] < 1e-6).mean()))
  fu = ev['U_ok'] == 0
  S['unbounded_repair'] = dict(fail_rate=float(fu.mean()), mean_edges_when_fails=float(ev.loc[fu, 'U_edges'].mean()),
                               mean_edges_when_ok=float(ev.loc[~fu, 'U_edges'].mean()))
  S['episodes_without_events'] = int((eps['n_events'] == 0).sum())
  S['uncharged_dstar_edges_total'] = int(eps.get('uncharged_dstar_edges', pd.Series([0])).sum())
  S['ms_median'] = {k: float(ev[f'{k}_ms'].median()) for k in ['G', 'D', 'B', 'U']}

  # ---- generalisation: per map family
  fam = ep.groupby('map_type')[pols].sum()
  fam_ratio = fam.div(fam['G_DSTAR'], axis=0)
  fam_ratio.to_csv(os.path.join(T, 'policy_ratio_by_map_type.csv'))
  ratio_size_R = ep.groupby(['blockage', 'size', 'R'])[['AR_DSTAR', 'G_DSTAR']].sum()
  ratio_size_R['ratio'] = ratio_size_R['AR_DSTAR'] / ratio_size_R['G_DSTAR']
  ratio_size_R.to_csv(os.path.join(T, 'phase_ratio_AR_vs_DSTAR.csv'))

  with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(S, f, indent=2)

  figures(ev, ep, pol_tab, cells, ratio_size_R, fam_ratio, F)
  print(json.dumps(S, indent=2))
  print(pol_tab[['policy', 'ratio_vs_dstar', 'ci_low', 'ci_high']].round(3).to_string(index=False))
  print(tt.round(4).to_string(index=False))


# ------------------------------------------------------------------ figures
def figures(ev, ep, pol_tab, cells, ratio_size_R, fam_ratio, F):
  # Fig 1 — phase diagram: log2(cost repair-first→D* / cost D* every event), small multiples per blockage
  blocks = ['point', 'wall', 'multi', 'barrier']
  sizes = sorted(ev['size'].unique()); radii = sorted(ev['R'].unique())
  fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
  lim = 2.0; norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
  for ax, bl in zip(axes, blocks):
    M = np.full((len(radii), len(sizes)), np.nan)
    for i, R in enumerate(radii):
      for j, s in enumerate(sizes):
        try:
          M[i, j] = np.log2(ratio_size_R.loc[(bl, s, R), 'ratio'])
        except KeyError:
          pass
    im = ax.imshow(np.clip(M, -lim, lim), cmap=DIV, norm=norm, origin='lower', aspect='auto')
    for i in range(len(radii)):
      for j in range(len(sizes)):
        if not np.isnan(M[i, j]):
          ax.text(j, i, f'{2 ** M[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                  color=INK if abs(M[i, j]) < 1.2 else SURFACE)
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes)
    ax.set_yticks(range(len(radii))); ax.set_yticklabels(radii)
    ax.set_title(bl); ax.set_xlabel('map size (cells)'); ax.grid(False)
  axes[0].set_ylabel('perception radius R')
  cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
  cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['×0.25', '×0.5', '×1', '×2', '×4'])
  cb.set_label('cost ratio: repair-first→D* Lite ÷ D* Lite\n(blue = repair pays, red = D* Lite cheaper)', color=INK2)
  fig.suptitle('Where local repair pays off (held-out episodes, edge evaluations)', x=0.01, y=1.04, ha='left',
               fontweight='bold', color=INK)
  fig.savefig(os.path.join(F, 'fig1_phase_diagram.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

  # Fig 2 — policy cost relative to D* Lite with 95% cluster-bootstrap CI (single series, one colour)
  pt = pol_tab.set_index('policy').loc[[p for p in POLICY_ORDER if p in pol_tab['policy'].values]]
  pt = pt.iloc[::-1]
  fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(pt) + 1.4))
  y = np.arange(len(pt))
  ax.barh(y, pt['ratio_vs_dstar'], height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
  ax.errorbar(pt['ratio_vs_dstar'], y, xerr=[pt['ratio_vs_dstar'] - pt['ci_low'], pt['ci_high'] - pt['ratio_vs_dstar']],
              fmt='none', ecolor=INK, elinewidth=1.2, capsize=3)
  ax.axvline(1.0, color=INK2, linewidth=1, linestyle='--')
  ax.set_yticks(y); ax.set_yticklabels(pt['label']); ax.set_xscale('log')
  from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
  ticks = [t for t in [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8] if pt['ci_low'].min() * 0.9 <= t <= pt['ci_high'].max() * 1.6]
  ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
  ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
  ax.set_xlabel('total work relative to D* Lite (log scale; lower is better)')
  for yi, v, hi in zip(y, pt['ratio_vs_dstar'], pt['ci_high']):
    ax.text(hi * 1.05, yi, f'{v:.3f}', va='center', fontsize=8.5, color=INK2)
  ax.set_xlim(right=pt['ci_high'].max() * 1.35)
  ax.grid(axis='y', visible=False)
  ax.set_title('Decision policies on held-out episodes (95% cluster-bootstrap CI)', loc='left')
  fig.savefig(os.path.join(F, 'fig2_policy_costs.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

  # Fig 3 — repair failure probability vs R per blockage (4 series, direct labels)
  fig, ax = plt.subplots(figsize=(6.5, 4))
  g = ev.groupby(['blockage', 'R'])['B_ok'].mean().unstack(0)
  for k, bl in enumerate(blocks):
    vals = 1 - g[bl]
    ax.plot(g.index, vals, color=SERIES[k], marker='o', markersize=6, markeredgecolor=SURFACE, label=bl)
    ax.text(g.index[-1] + 0.3, vals.iloc[-1], bl, color=INK, va='center', fontsize=9)
  ax.set_xticks(g.index); ax.set_xlim(g.index[0] - 0.5, g.index[-1] + 2.5); ax.set_ylim(0, 1.02)
  ax.set_xlabel('perception radius R (cells)'); ax.set_ylabel('P(bounded repair fails)')
  ax.legend(loc='lower left', ncol=4, fontsize=8.5)
  ax.set_title('Repair failure falls with perception radius', loc='left')
  fig.savefig(os.path.join(F, 'fig3_failure_vs_R.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

  # Fig 4 — break-even model validation (cells): measured p_fail vs predicted threshold p*
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
  for ax, fb, name in [(axes[0], 'astar', 'A*'), (axes[1], 'dstar', 'D* Lite')]:
    for k, (val, lab) in enumerate([(True, 'repair-first measured cheaper'), (False, f'{name} measured cheaper')]):
      sub = cells[cells[f'true_pays_{fb}'] == val]
      ax.scatter(sub[f'p_star_{fb}'], sub['p_fail'], s=14, color=SERIES[k], alpha=0.75, label=lab,
                 edgecolors=SURFACE, linewidths=0.6)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle='--')
    ax.text(0.03, 0.93, 'above line: model says\nfallback is cheaper', fontsize=8, color=INK2, va='top')
    ax.set_xlim(-0.05, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f'break-even threshold p* = 1 − C(repair) / C({name})'); ax.set_title(f'fallback: {name}', loc='left')
    ax.legend(loc='lower left', fontsize=8)
  axes[0].set_ylabel('measured P(repair fails)')
  fig.savefig(os.path.join(F, 'fig4_break_even_validation.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

  # Fig 5 — ROC: learned predictor vs v1 risk score (2 series, direct labels)
  if 'pred_p' in ev:
    yf = 1 - ev['B_ok']
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for k, (col, lab) in enumerate([('pred_p', 'learned predictor'), ('risk', 'v1 risk score')]):
      fpr, tpr, _ = roc_curve(yf, ev[col]); auc = roc_auc_score(yf, ev[col])
      ax.plot(fpr, tpr, color=SERIES[k], label=f'{lab} (AUC {auc:.2f})')
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
    ax.legend(loc='lower right', fontsize=8.5); ax.set_title('Predicting repair failure', loc='left')
    fig.savefig(os.path.join(F, 'fig5_roc.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

  # Fig 6 — generalisation across map families: repair-first and predictor vs D* Lite
  cols = [c for c in ['AR_DSTAR', 'PRED_DSTAR', 'ARU_DSTAR'] if c in fam_ratio]
  fr = fam_ratio[cols].sort_values(cols[0])
  fig, ax = plt.subplots(figsize=(8, 4))
  x = np.arange(len(fr)); w = 0.26
  for k, c in enumerate(cols):
    ax.bar(x + (k - 1) * w, fr[c], width=w - 0.03, color=SERIES[k], label=POLICY_LABEL[c])
  ax.axhline(1.0, color=INK2, linestyle='--', linewidth=1)
  ax.set_xticks(x); ax.set_xticklabels(fr.index, rotation=30, ha='right')
  ax.set_ylabel('work relative to D* Lite'); ax.grid(axis='x', visible=False)
  ax.legend(fontsize=8, loc='upper left'); ax.set_title('Generalisation across map families', loc='left')
  fig.savefig(os.path.join(F, 'fig6_map_families.png'), dpi=200, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
  main()
