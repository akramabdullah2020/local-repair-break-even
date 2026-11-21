"""Stage 1 -> 2: fit the repair-failure predictor on TRAINING events only and pickle it.

  python fit_predictor.py results/v2_events/full_train/events.csv results/v2_events/predictor.pkl
"""
import json, pickle, sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.policies import RepairPredictor, FEATURES

src, dst = sys.argv[1], sys.argv[2]
tr = pd.read_csv(src)
assert (tr['split'] == 'train').all(), 'predictor must be fitted on training seeds only'
pred = RepairPredictor().fit(tr)
with open(dst, 'wb') as f:
  pickle.dump(pred, f)
info = dict(n_train_events=len(tr), tau=pred.tau, train_auc=round(roc_auc_score(1 - tr.B_ok, pred.p_fail(tr)), 4),
            features=FEATURES, c_l_by_R={int(k): round(v, 1) for k, v in pred.c_l.items()})
with open(dst.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
  json.dump(info, f, indent=2)
print(info)
