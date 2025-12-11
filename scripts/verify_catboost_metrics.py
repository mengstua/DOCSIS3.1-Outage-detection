import pickle
import sys
from pathlib import Path

p = Path('logs/results_full_run.pkl')
if not p.exists():
    print('results_full_run.pkl not found at', p.resolve())
    sys.exit(2)

with p.open('rb') as f:
    data = pickle.load(f)

print('Keys in pickle:', list(data.keys()))

model = None
for k in ['cat', 'catboost', 'cat_model', 'catboost_model']:
    if k in data:
        model = data[k]
        model_key = k
        break

if model is None:
    # try to find first model-like object
    for k, v in data.items():
        if hasattr(v, 'predict_proba'):
            model = v
            model_key = k
            break

if model is None:
    print('No model with predict_proba found in pickle. Keys:', list(data.keys()))
    sys.exit(3)

print('Found model under key:', model_key, type(model))

# get X_test and y_test
X_test = data.get('X_test') or data.get('X_test_transformed') or data.get('X_test_scaled')
y_test = data.get('y_test')

if X_test is None or y_test is None:
    print('X_test or y_test not found in pickle. Keys available:', list(data.keys()))
    sys.exit(4)

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# Convert if pandas DataFrame
try:
    import pandas as pd
    if isinstance(X_test, pd.DataFrame):
        X_arr = X_test.values
    else:
        X_arr = np.asarray(X_test)
    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_arr = np.asarray(y_test).ravel()
    else:
        y_arr = np.asarray(y_test).ravel()
except Exception:
    X_arr = np.asarray(X_test)
    y_arr = np.asarray(y_test).ravel()

# predict_proba
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X_arr)
    # binary: take second column if shape (n,2)
    if probs.ndim == 2 and probs.shape[1] > 1:
        pos_probs = probs[:, 1]
    else:
        pos_probs = probs.ravel()
else:
    # try decision_function
    if hasattr(model, 'decision_function'):
        dec = model.decision_function(X_arr)
        # map to 0-1 via minmax
        from sklearn.preprocessing import minmax_scale
        pos_probs = minmax_scale(dec)
    else:
        print('Model has no predict_proba or decision_function')
        sys.exit(5)

roc = roc_auc_score(y_arr, pos_probs)
pr = average_precision_score(y_arr, pos_probs)

# derive threshold used in metrics CSV if available inside data
threshold = data.get('best_threshold') or data.get('threshold')

# compute precision/recall/f1 at threshold ~0.5 or provided threshold
th = float(threshold) if threshold is not None else 0.5
preds = (pos_probs >= th).astype(int)
prec, recall, f1, _ = precision_recall_fscore_support(y_arr, preds, average='binary')

print('\nRecomputed metrics for model key:', model_key)
print('roc_auc:', roc)
print('pr_auc (avg precision):', pr)
print('threshold used:', th)
print('precision:', prec)
print('recall:', recall)
print('f1:', f1)

# Also print sample of pos_probs
print('\nSample positive probabilities (first 10):', pos_probs[:10])
