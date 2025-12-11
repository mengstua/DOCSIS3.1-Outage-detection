"""
Original archived: check_metrics_dashboard.py
"""
import sys
print('This helper has been archived. See archive/scripts/check_metrics_dashboard.py')
sys.exit(0)
import pandas as pd
import os

LOGS = 'logs'
path = os.path.join(LOGS, 'models_metrics.csv')
print('Checking', path)
if not os.path.exists(path):
    print('metrics CSV not found')
    raise SystemExit(1)

metrics = pd.read_csv(path, index_col=0)
print('\nmetrics head:')
print(metrics.head())


def safe_metric(metrics_df, model_key, metric_name, alt_names=None, default=None):
    if metrics_df is None:
        return default
    candidates = [metric_name]
    if alt_names:
        candidates = candidates + list(alt_names)
    for m in candidates:
        if model_key in metrics_df.index and m in metrics_df.columns:
            try:
                return float(metrics_df.loc[model_key, m])
            except Exception:
                pass
    for m in candidates:
        if m in metrics_df.index and model_key in metrics_df.columns:
            try:
                return float(metrics_df.loc[m, model_key])
            except Exception:
                pass
    for m in candidates:
        flat = f"{model_key}_{m}"
        if flat in metrics_df.columns:
            try:
                return float(metrics_df[flat].iloc[0])
            except Exception:
                pass
    return default

print('\nResolved metrics (xgb):')
print('roc_auc ->', safe_metric(metrics, 'xgb', 'roc_auc', alt_names=['roc_auc']))
print('f1 ->', safe_metric(metrics, 'xgb', 'best_f1', alt_names=['f1']))
print('threshold ->', safe_metric(metrics, 'xgb', 'best_threshold', alt_names=['threshold']))
print('cv_std ->', safe_metric(metrics, 'xgb', 'cv_std', alt_names=['cv_std','std']))
