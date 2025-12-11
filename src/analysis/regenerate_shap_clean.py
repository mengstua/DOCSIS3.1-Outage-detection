"""Regenerate a clean SHAP top-10 image and CSV (relocated to src/analysis).

This is a direct copy of `scripts/regenerate_shap_clean.py` for the new project layout.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGS = os.path.join(ROOT, 'logs')
OUT_CSV = os.path.join(LOGS, 'shap_top10_clean.csv')
OUT_PNG = os.path.join(LOGS, 'shap_top10.png')


def read_shap_series(path):
    try:
        df = pd.read_csv(path, header=0)
    except Exception:
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            return None

    if df.shape[1] == 1:
        col = df.columns[0]
        if pd.to_numeric(df[col], errors='coerce').notna().all():
            return None
        return pd.Series(np.arange(len(df)), index=df[col].astype(str)).astype(float)

    numeric_score = {}
    for c in df.columns:
        numeric_score[c] = pd.to_numeric(df[c], errors='coerce').notna().mean()

    value_col = max(numeric_score, key=lambda k: numeric_score[k])
    feature_col = min(numeric_score, key=lambda k: numeric_score[k])

    if pd.to_numeric(df[feature_col], errors='coerce').notna().mean() > 0.5:
        for c in df.columns:
            if str(c).lower().startswith('feature') or 'feature' in str(c).lower():
                feature_col = c
                break

    try:
        feats = df[feature_col].astype(str).values
        vals = pd.to_numeric(df[value_col], errors='coerce').astype(float).values
        return pd.Series(vals, index=feats)
    except Exception:
        return None


def load_feature_map():
    fmap = os.path.join(LOGS, 'feature_name_map.csv')
    if os.path.exists(fmap):
        try:
            df = pd.read_csv(fmap)
            if 'engineered' in df.columns and 'base' in df.columns:
                return dict(zip(df['engineered'].astype(str), df['base'].astype(str)))
        except Exception:
            return {}
    return {}


def base_name(feat, fmap):
    if feat in fmap:
        return fmap[feat]
    s = str(feat)
    s = re.sub(r'(_lag\d+|_rollmean\d+|_rollstd\d+|_delta\d+|_p\d+|_ma\d+|_diff)$', '', s)
    s = re.sub(r'_p\d+$', '', s)
    s = re.sub(r'__+', '_', s).strip('_')
    return s


def is_engineered_or_target(name):
    lx = str(name).lower()
    if lx in ('call_flag', 'target', 'y') or 'call_flag' in lx:
        return True
    if any(p in lx for p in ['_lag', 'rollmean', 'rollstd', 'roll', '_p', '_ma', '_delta', '_diff']):
        return True
    return False


def main():
    cand = [os.path.join(LOGS, f) for f in os.listdir(LOGS) if f.startswith('shap_') and f.endswith('.csv')]
    if not cand:
        print('No shap CSV files found in logs/. Run SHAP generation first.', file=sys.stderr)
        sys.exit(1)

    fmap = load_feature_map()

    frames = []
    for p in cand:
        s = read_shap_series(p)
        if s is None or s.empty:
            continue
        s.index = [str(i) for i in s.index]
        frames.append(s)

    if not frames:
        print('No usable SHAP series found in shap CSVs.', file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(frames, axis=1)
    df_all = df_all.apply(pd.to_numeric, errors='coerce')
    mean_abs = df_all.mean(axis=1).abs()

    mapped = {}
    for feat, val in mean_abs.items():
        b = base_name(feat, fmap)
        mapped.setdefault(b, 0.0)
        mapped[b] += float(val)

    agg = pd.Series(mapped).sort_values(ascending=False)
    agg = agg.loc[~agg.index.to_series().apply(is_engineered_or_target)]

    if agg.empty:
        print('After filtering targets/engineered features, no features remain.', file=sys.stderr)
        sys.exit(1)

    canonical_allowed = [
        'deregistration_count',
        'latency_jitter_ms',
        'keepalive_miss_quarters',
        'offline_count_quarters',
        'snr_downstream_db',
        'rx_power_upstream_dbmv',
        'profile_measurements_count',
        'avg_node_quality_index',
        'rtt_ms_p95',
        'rtt_ms_p50'
    ]

    top = []
    for f in canonical_allowed:
        if f in agg.index and len(top) < 10:
            top.append(f)

    for f in agg.index:
        if len(top) >= 10:
            break
        if f not in top:
            top.append(f)

    top_series = agg.loc[top]

    os.makedirs(LOGS, exist_ok=True)
    top_series.to_csv(OUT_CSV, header=['mean_abs_shap'])
    print('Saved cleaned SHAP top list to:', OUT_CSV)

    plt.figure(figsize=(8, 6))
    top_series.sort_values().plot(kind='barh', color='purple')
    plt.xlabel('Aggregated mean |SHAP|')
    plt.title('SHAP top features (clean, base-level)')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print('Saved cleaned SHAP image to:', OUT_PNG)

    print('\nTop features:')
    for i, (f, v) in enumerate(top_series.items(), 1):
        print(f'{i}. {f} ({v:.6f})')


if __name__ == '__main__':
    main()
