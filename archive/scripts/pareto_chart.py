"""
Original archived: pareto_chart.py
"""
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGS = os.path.join(ROOT, 'logs')
DATA = os.path.join(ROOT, 'data')

def load_pareto_from_logs():
    candidates = [
        os.path.join(LOGS, 'pareto_feature_importance_with_base.csv'),
        os.path.join(LOGS, 'pareto_base_top50.csv'),
        os.path.join(LOGS, 'pareto_feature_importance.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None

def compute_from_model():
    model_paths = [
        os.path.join(ROOT, 'xgb_model.pkl'),
        os.path.join(LOGS, 'xgb_model.pkl'),
        os.path.join(LOGS, 'model_xgb.pkl'),
    ]
    model = None
    for mp in model_paths:
        if os.path.exists(mp):
            try:
                model = joblib.load(mp)
                break
            except Exception:
                continue
    if model is None:
        return None

    data_candidates = [
        os.path.join(DATA, 'ml_master_table_1M.csv'),
        os.path.join(DATA, 'ml_master_table_6HK.csv'),
    ]
    df = None
    for dp in data_candidates:
        if os.path.exists(dp):
            try:
                df = pd.read_csv(dp, nrows=5)
                data_path = dp
                break
            except Exception:
                continue

    if df is None:
        print('No suitable data CSV found to infer feature names; aborting model-based importance.', file=sys.stderr)
        return None

    try:
        df_full = pd.read_csv(data_path, nrows=10)
    except Exception:
        df_full = df

    possible_targets = ['call_flag', 'target', 'y']
    cols = list(df_full.columns)
    drop_cols = [c for c in ['cpe_hk_hashed', 'snapshot_date'] if c in cols]
    feat_cols = [c for c in cols if c not in drop_cols and c not in possible_targets]

    if hasattr(model, 'feature_importances_'):
        importances = np.array(model.feature_importances_)
        if importances.shape[0] != len(feat_cols):
            try:
                booster = model.get_booster()
                fmap = booster.feature_names
                if fmap:
                    feat_cols = list(fmap)
                    importances = importances[:len(feat_cols)]
            except Exception:
                pass

        df_imp = pd.DataFrame({'feature': feat_cols, 'importance': importances})
        return df_imp.sort_values('importance', ascending=False)

    return None

def make_pareto(df_imp, out_png, save_selected_csv=True):
    df = df_imp.copy()
    df['importance'] = df['importance'].abs()
    fmap_path = os.path.join(LOGS, 'feature_name_map.csv')
    feature_to_base = {}
    if os.path.exists(fmap_path):
        try:
            fmap_df = pd.read_csv(fmap_path)
            if 'engineered' in fmap_df.columns and 'base' in fmap_df.columns:
                feature_to_base = dict(zip(fmap_df['engineered'].astype(str), fmap_df['base'].astype(str)))
        except Exception:
            feature_to_base = {}

    def infer_base(name):
        if name in feature_to_base:
            return feature_to_base[name]
        s = str(name)
        s = re.sub(r"(_lag|_rollmean|_rollstd|_p|_ma|_delta|_diff)\d*", "", s)
        s = re.sub(r"_p\d+", "", s)
        s = re.sub(r"__+", "_", s).strip('_')
        return s

    try:
        if 'feature' in df.columns:
            feat_series = df['feature']
            if isinstance(feat_series, pd.DataFrame):
                feat_series = df.iloc[:, 0]
        else:
            feat_series = df.iloc[:, 0]

        features = feat_series.astype(str).tolist()
        bases = []
        for f in features:
            try:
                bases.append(infer_base(f))
            except Exception:
                bases.append(str(f))
        df['base_feature'] = bases
    except Exception:
        df['base_feature'] = df['feature'].astype(str)

    agg = df.groupby('base_feature', as_index=False)['importance'].sum()
    agg = agg.rename(columns={'base_feature': 'feature'})
    df = agg.sort_values('importance', ascending=False).reset_index(drop=True)
    df['percent'] = 100.0 * df['importance'] / df['importance'].sum()
    df['cum_percent'] = df['percent'].cumsum()

    try:
        plt.style.use('seaborn')
    except Exception:
        pass
    plt.figure(figsize=(12, 6))
    try:
        labels = [str(v) for v in df['feature'].values]
    except Exception:
        labels = [f'feat_{i}' for i in range(len(df))]

    plt.bar(labels, df['percent'], color='royalblue')
    plt.plot(labels, df['cum_percent'], color='red', marker='o')
    plt.axhline(80, color='green', linestyle='--', label='80% Threshold')
    plt.xticks(rotation=90)
    plt.ylabel('Percent Contribution (%)')
    plt.title('Pareto Chart of Feature Importance (80/20 Rule)')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f'Saved pareto chart to: {out_png}')

    if save_selected_csv:
        mask = (df['cum_percent'].values <= 80)
        sel_vals = df['feature'].values
        sel = [str(v) for (v, m) in zip(sel_vals, mask) if m]
        def is_engineered_or_target(x):
            lx = str(x).lower()
            if 'call_flag' in lx or lx.startswith('target') or lx in ('call_flag', 'target', 'y'):
                return True
            if any(s in lx for s in ['_lag', 'rollmean', 'rollstd', 'roll', '_p', '_ma', '_delta', '_diff']):
                return True
            return False

        cleaned_sel = [s for s in sel if not is_engineered_or_target(s)]
        final_sel = cleaned_sel if len(cleaned_sel) > 0 else sel
        sel_path = os.path.join(LOGS, 'selected_pareto_80pct_vars.csv')
        pd.DataFrame({'feature': final_sel}).to_csv(sel_path, index=False)
        print(f'Saved selected 80% features to: {sel_path}')

def main():
    df_imp = load_pareto_from_logs()
    if df_imp is not None:
        if 'feature' not in df_imp.columns or 'importance' not in df_imp.columns:
            if 'pct' in df_imp.columns and 'base' in df_imp.columns:
                df_imp = df_imp.rename(columns={'base': 'feature', 'pct': 'importance'})
            elif 'feature' in df_imp.columns and 'pct' in df_imp.columns:
                df_imp = df_imp.rename(columns={'pct': 'importance'})
            else:
                df_imp = df_imp.iloc[:, :2]
                df_imp.columns = ['feature', 'importance']
    else:
        df_imp = compute_from_model()

    if df_imp is None or df_imp.empty:
        print('Could not locate feature importances to build Pareto chart.', file=sys.stderr)
        sys.exit(1)

    out_png = os.path.join(LOGS, 'pareto_chart.png')
    make_pareto(df_imp, out_png)

if __name__ == '__main__':
    main()
