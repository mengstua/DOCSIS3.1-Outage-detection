"""
Original archived: clean_selected_pareto.py
"""
import os
import pandas as pd
import datetime
import re

LOGS = 'logs'
FNAME = 'selected_pareto_80pct_vars.csv'
path = os.path.join(LOGS, FNAME)

if not os.path.exists(path):
    print(f'File not found: {path}')
    raise SystemExit(1)

try:
    df = pd.read_csv(path, header=0)
except Exception:
    df = pd.read_csv(path, header=None)

feat_col = df.columns[0]
features = df[feat_col].astype(str).str.strip().tolist()

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
backup = os.path.join(LOGS, f'selected_pareto_80pct_vars.backup.{ts}.csv')
pd.DataFrame(features, columns=[feat_col]).to_csv(backup, index=False)
print(f'Backup written to: {backup}')

def is_engineered_or_target(f):
    lf = f.lower()
    if lf.startswith('call_flag') or lf.startswith('target') or lf == 'call_flag' or lf == 'y':
        return True
    if re.search(r'(_lag\d*$)|(_lag$)|(_roll)|rollmean|rollstd|_delta|_diff', lf):
        return True
    if re.search(r'_p\d{1,3}$', lf):
        return True
    if 'call_flag' in lf:
        return True
    return False

cleaned = [f for f in features if not is_engineered_or_target(f)]
removed = [f for f in features if f not in cleaned]
print(f'Removed {len(removed)} engineered/target-derived rows.')
if removed:
    for r in removed:
        print(' -', r)

out_df = pd.DataFrame(cleaned, columns=[feat_col])
out_df.to_csv(path, index=False)
print(f'Cleaned CSV written to: {path}')
print('Done.')
