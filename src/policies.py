"""Offline evaluation of decision policies on the paired event table (plan §6).

Cost unit = edge-cost evaluations per blocking event (hardware independent).
Exact policies (each event's cost is measured, not simulated):
  G_ASTAR       global A* at every event
  G_DSTAR       D* Lite at every event
  AR_ASTAR      always try bounded repair first, fall back to A* on failure
  AR_DSTAR      always try bounded repair first, fall back to D* Lite on failure
  ARU_ASTAR     always try UNbounded repair first (v1 behaviour), fall back to A*
  ORACLE_ASTAR  knows whether bounded repair will succeed: repair if yes, A* if no
  ORACLE_DSTAR  same with D* Lite fallback
  ARU_DSTAR     unbounded repair first, D* Lite fallback          (own D* shadow)
  RISK_DSTAR    v1 risk gate (theta=0.45), D* Lite fallback       (own D* shadow)
  PRED_DSTAR    predictor + break-even rule, D* Lite fallback     (own D* shadow, eval split only)
  PREDTAU_DSTAR predictor + threshold tuned on training           (own D* shadow, eval split only)
Decision policies with A* fallback (exact as well):
  RISK_<theta>  v1 risk model: risk >= theta -> A* directly, else repair first
  PRED          logistic model of repair failure + break-even rule p_fail < 1 - C_L/C_G
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['risk_ratio', 'risk_prox', 'risk_bottle', 'n_invalid', 'invalid_span', 'splice_in_window',
            'window_free', 'R', 'first_invalid', 'n_blocked_changed', 'n_changed']


def exact_policy_costs(df):
  out = pd.DataFrame(index=df.index)
  fail_b = 1 - df['B_ok']; fail_u = 1 - df['U_ok']
  df_edges = df['DF_edges'].fillna(0.0)

  def extra(col):  # D* Lite update work left after the shadow's last plan() (charged on the last event)
    c = f'{col}_extra_edges'
    return df[c].fillna(0.0) if c in df else 0.0

  out['G_ASTAR'] = df['G_edges']
  out['G_DSTAR'] = df['D_edges'] + extra('D')
  out['AR_ASTAR'] = df['B_edges'] + fail_b * df['G_edges']
  out['AR_DSTAR'] = df['B_edges'] + fail_b * df_edges + extra('DF')
  out['ARU_ASTAR'] = df['U_edges'] + fail_u * df['G_edges']
  out['ORACLE_ASTAR'] = np.where(df['B_ok'] == 1, df['B_edges'], df['G_edges'])
  out['ORACLE_DSTAR'] = np.where(df['B_ok'] == 1, df['B_edges'], df_edges) + extra('DF')
  # policies measured with their own D* Lite shadow (exact)
  if 'DUF_edges' in df:
    out['ARU_DSTAR'] = df['U_edges'] + fail_u * df['DUF_edges'].fillna(0.0) + extra('DUF')
    # oracle for UNbounded repair: skips repair attempts that will fail (D* shadow 'ufail' is called on
    # exactly those events, so this is exact)
    out['ORACLE_U_DSTAR'] = np.where(df['U_ok'] == 1, df['U_edges'], df['DUF_edges'].fillna(0.0)) + extra('DUF')
  if 'DR_edges' in df:
    gate = df['risk'] >= 0.45
    out['RISK_DSTAR'] = np.where(gate, df['DR_edges'], df['B_edges'] + fail_b * df['DR_edges'].fillna(0.0)) + extra('DR')
  for flag, col, name in [('pred_repair', 'DP', 'PRED_DSTAR'), ('predtau_repair', 'DT', 'PREDTAU_DSTAR')]:
    if flag in df:
      rep = df[flag] == 1
      out[name] = np.where(rep, df['B_edges'] + fail_b * df[f'{col}_edges'].fillna(0.0), df[f'{col}_edges']) + extra(col)
  return out


def risk_policy_cost(df, theta):
  ar = df['B_edges'] + (1 - df['B_ok']) * df['G_edges']
  return np.where(df['risk'] >= theta, df['G_edges'], ar)


class RepairPredictor:
  """p(repair fails | cheap features) and estimates of C_L, C_G (A*) and C_D (D* Lite),
  all fitted on TRAINING events only. Two decision rules for the D* Lite fallback:
    analytic : repair iff p_fail < p* = 1 - C_L_hat / C_D_hat   (break-even model, §4)
    tau      : repair iff p_fail < tau, tau tuned on training events
  """

  TAUS = np.round(np.arange(0.30, 1.0001, 0.025), 3)

  def fit(self, train):
    X = train[FEATURES].values; y = 1 - train['B_ok'].values
    self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    self.clf.fit(X, y)
    Xc = self._cost_X(train['dist_to_goal'].values, train['size'].values, train['R'].values)
    self.reg_g = LinearRegression().fit(Xc, np.log1p(train['G_edges']))   # A* cost
    self.reg_d = LinearRegression().fit(Xc, np.log1p(train['D_edges']))   # D* Lite cost
    self.c_l = train.groupby('R')['B_edges'].mean().to_dict()             # repair cost per radius
    self.c_l_default = float(train['B_edges'].mean())
    # tune tau on training (D* cost of every-event shadow approximates the D* call cost)
    p = self.p_fail(train)
    ar = train['B_edges'] + (1 - train['B_ok']) * train['DF_edges'].fillna(0.0)
    costs = [np.where(p < t, ar, train['D_edges']).sum() for t in self.TAUS]
    self.tau = float(self.TAUS[int(np.argmin(costs))])
    return self

  @staticmethod
  def _cost_X(dist, size, R):
    return np.c_[np.log1p(dist), np.log(size), R]

  def p_fail(self, df):
    return self.clf.predict_proba(df[FEATURES].values)[:, 1]

  def p_fail_row(self, row, size):
    x = np.array([[row[f] for f in FEATURES]], dtype=float)
    return self.clf.predict_proba(x)[0, 1]

  def decide_repair_analytic_row(self, row, size, p):
    c_d = float(np.expm1(self.reg_d.predict(self._cost_X([row['dist_to_goal']], [size], [row['R']]))[0]))
    c_l = self.c_l.get(row['R'], self.c_l_default)
    return p < 1.0 - c_l / max(c_d, 1.0)

  def c_g_hat(self, df):
    return np.expm1(self.reg_g.predict(self._cost_X(df['dist_to_goal'].values, df['size'].values, df['R'].values)))

  def decide_repair(self, df):
    c_l = df['R'].map(self.c_l).fillna(np.median(list(self.c_l.values()))).values
    p_star = 1.0 - c_l / np.maximum(self.c_g_hat(df), 1.0)
    return self.p_fail(df) < p_star

  def cost(self, df):
    ar = df['B_edges'] + (1 - df['B_ok']) * df['G_edges']
    return np.where(self.decide_repair(df), ar, df['G_edges'])


def all_policy_costs(df, predictor=None, thetas=(0.2, 0.3, 0.45)):
  out = exact_policy_costs(df)
  for th in thetas:
    out[f'RISK_{th}'] = risk_policy_cost(df, th)
  if predictor is not None:
    out['PRED'] = predictor.cost(df)
  return out


def break_even_table(df, by):
  """Per group: measured failure rate of bounded repair vs the break-even threshold p* = 1 - C_L/C_G
  (C_L, C_G = mean costs in the group); repair-first pays off on average iff p_fail < p*."""
  g = df.groupby(by)
  t = pd.DataFrame({
    'n_events': g.size(),
    'p_fail': 1 - g['B_ok'].mean(),
    'C_L': g['B_edges'].mean(),
    'C_G_astar': g['G_edges'].mean(),
    'C_G_dstar': g['D_edges'].mean(),
  })
  t['p_star_astar'] = 1 - t['C_L'] / t['C_G_astar']
  t['p_star_dstar'] = 1 - t['C_L'] / t['C_G_dstar']
  t['repair_pays_vs_astar'] = t['p_fail'] < t['p_star_astar']
  t['repair_pays_vs_dstar'] = t['p_fail'] < t['p_star_dstar']
  return t
