import pickle, os, pandas as pd, numpy as np
LOGS='logs'
res_path = os.path.join(LOGS,'results_full_run.pkl')
print('results exists?', os.path.exists(res_path))
results = None
if os.path.exists(res_path):
    try:
        with open(res_path,'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print('failed load results:',e)

# build a tiny test input
input_feats = None
if results and 'top_features' in results:
    input_feats = results['top_features'][:10]
else:
    input_feats = ['snr_ds_avg','snr_us_avg','p95_rtt','jitter','fec31_uncor_sum']

# helper from dashboard (simplified)
def get_range(fname):
    fallback = {
        'snr_ds_avg': (0.0, 50.0, 35.0),
        'snr_us_avg': (0.0, 50.0, 38.0),
        'p95_rtt': (0.0, 1000.0, 25.0),
        'jitter': (0.0, 100.0, 5.0),
        'fec31_uncor_sum': (0.0, 1e6, 100.0),
        'fec_ofdma_uncor_sum': (0.0, 1e6, 50.0),
        'download_speed_mbps': (0.0, 1000.0, 500.0)
    }
    if results and isinstance(results.get('X_train'),pd.DataFrame) and fname in results['X_train'].columns:
        col = results['X_train'][fname].dropna().astype(float)
        if not col.empty:
            lo = float(col.min()); hi = float(col.max()); mid = float(col.median())
            span = max(1.0, (hi-lo)*0.05)
            return (max(0.0, lo-span), hi+span, mid)
    return fallback.get(fname,(0.0,100.0,0.0))

# assemble user_vals
user_vals = {}
for f in input_feats:
    lo,hi,mid = get_range(f)
    user_vals[f] = mid

# attempt model prediction
model_xgb = results.get('xgb') if results else None
scaler = results.get('scaler') if results else None

print('using top_features:', input_feats)
print('model present?', model_xgb is not None)

# build X_in
X_in = pd.DataFrame([user_vals])
if model_xgb is not None and results and 'top_features' in results:
    top = results['top_features']
    missing = [c for c in top if c not in X_in.columns]
    print('missing for model:', missing[:5])
    if not missing:
        X_proc = X_in[top]
        try:
            if scaler is not None and hasattr(scaler,'transform'):
                X_proc = pd.DataFrame(scaler.transform(X_proc), columns=top)
        except Exception as e:
            print('scaler transform failed:', e)
        try:
            proba = float(model_xgb.predict_proba(X_proc)[:,1][0])
            print('pred proba model:', proba)
        except Exception as e:
            print('model predict failed:', e)
else:
    # heuristic
    score=0.0
    score += (40 - user_vals.get('snr_ds_avg',35)) * 0.02
    score += (40 - user_vals.get('snr_us_avg',38)) * 0.015
    score += user_vals.get('p95_rtt',25) * 0.003
    score += user_vals.get('jitter',5) * 0.008
    score += user_vals.get('fec31_uncor_sum',100) * 0.0002
    score += (500 - user_vals.get('download_speed_mbps',500)) * 0.0001
    probability = 1/(1+np.exp(-score))
    print('heuristic proba', probability)
