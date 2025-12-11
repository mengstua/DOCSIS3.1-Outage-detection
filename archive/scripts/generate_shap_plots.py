"""Generate SHAP visualizations (beeswarm, bar, dependence, waterfall) and save PNGs to `logs/`.

Usage:
    .venv\Scripts\python scripts\generate_shap_plots.py --model xgb

If `results_full_run.pkl` lacks X_test, the script will rebuild features from `data/ml_master_table_1M.csv` (may be slow).
"""
import os
import sys
import argparse
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS = os.path.join(ROOT, 'logs')
RESULTS_PKL = os.path.join(LOGS, 'results_full_run.pkl')

OUT_BEE = os.path.join(LOGS, 'shap_beeswarm.png')
OUT_BAR = os.path.join(LOGS, 'shap_bar_mean.png')
OUT_DEP_FMT = os.path.join(LOGS, 'shap_dependence_{feat}.png')
OUT_WF_FMT = os.path.join(LOGS, 'shap_waterfall_{idx}.png')


def base_feature_name(feat):
    return re.sub(r'(_lag\d+|_rollmean\d+|_rollstd\d+|_delta\d+|_p\d+)$', '', feat)


def rebuild_X_from_csv(results):
    print('Rebuilding X from CSV (this may take a while).')
    df = pd.read_csv(os.path.join(ROOT,'data','ml_master_table_1M.csv'), low_memory=False)
    # minimal cleaning similar to model_compare_and_shap
    from sklearn.preprocessing import LabelEncoder
    df['snapshot_date'] = pd.to_datetime(df.get('snapshot_date', pd.Series(pd.NaT)), errors='coerce')
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d\.\-eE+]+", "", regex=True), errors='coerce')
            if converted.notna().mean() > 0.7:
                df[col] = converted
            else:
                try:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                except Exception:
                    df[col] = df[col].astype(str)
    # basic time features (lightweight)
    # drop target
    y = df.get('call_flag', pd.Series(0)).fillna(0).astype(int)
    X = df.drop(columns=['call_flag','cpe_hk_hashed','snapshot_date'], errors='ignore')
    # select top features
    top_features = results.get('top_features')
    if top_features is None:
        raise RuntimeError('top_features missing in results; cannot select features')

    # Some top_features may be engineered (lags/rollmean) which won't exist in raw CSV.
    # Attempt to map engineered->base and find best matching raw column. If a direct engineered
    # column is missing, fall back to its base name or a closest match.
    avail_cols = set(X.columns.astype(str).tolist())
    resolved = []
    for feat in top_features:
        if feat in avail_cols:
            resolved.append(feat)
            continue
        base = base_feature_name(feat)
        if base in avail_cols:
            resolved.append(base)
            continue
        # try to find any column that startswith base or contains base token
        candidates = [c for c in avail_cols if c.startswith(base) or base in c]
        if candidates:
            resolved.append(candidates[0])
            continue
        # last resort: skip this feature (we'll fill with zeros later)
        # but record placeholder so order aligns with top_features
        resolved.append(None)

    # Build selection DataFrame, filling missing columns with zeros
    # ensure sel_cols are unique (avoid duplicate column names which cause pandas to
    # return DataFrame slices when selecting by label)
    sel_cols = [r for r in resolved if r is not None]
    sel_cols = list(dict.fromkeys(sel_cols))
    X_sel = X.reindex(columns=sel_cols)
    # For any missing resolved entries, create zero columns with the engineered name
    for orig, res in zip(top_features, resolved):
        if res is None:
            X_sel[orig] = 0.0
        elif res != orig:
            # if we mapped to a different raw column, also ensure engineered name exists by copying
            src = X_sel[res]
            # if selecting src yielded multiple duplicate columns, take the first
            if isinstance(src, pd.DataFrame):
                src = src.iloc[:, 0]
            X_sel[orig] = src
    # coerce numeric
    for c in X_sel.columns:
        X_sel[c] = pd.to_numeric(X_sel[c], errors='coerce').fillna(0)
    return X_sel, y


def load_results():
    if not os.path.exists(RESULTS_PKL):
        raise FileNotFoundError(RESULTS_PKL)
    with open(RESULTS_PKL,'rb') as f:
        res = pickle.load(f)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='model name to explain (xgb, cat, rf, svc). Defaults to best available')
    args = p.parse_args()

    results = load_results()

    X_test = results.get('X_test')
    X_train = results.get('X_train')
    if X_test is None:
        X_all, y = rebuild_X_from_csv(results)
        # We'll need to match the model's expected feature names exactly to avoid XGBoost shape errors
        # so defer sampling until after we know the chosen model and its feature_names_in_
        X_shap = None
    else:
        # use top_features ordering
        top = results.get('top_features')
        if top is None:
            top = list(X_test.columns)
        X_shap = X_test[top].sample(n=min(1000, len(X_test)), random_state=42)

    # choose model
    models = {}
    for name in ['xgb','cat','rf','svc']:
        if name in results:
            models[name] = results[name]
    if not models:
        raise RuntimeError('No models found in results_full_run.pkl')
    chosen = args.model if args.model in models else (args.model if args.model is None else None)
    if chosen is None:
        # pick xgb > cat > rf > svc by preference
        for cand in ['xgb','cat','rf','svc']:
            if cand in models:
                chosen = cand
                break
    model = models[chosen]
    print('Using model for SHAP:', chosen)

    # compute explainer and shap values
    # If X_shap is None (because X_test was not present), construct X_shap matching model feature names
    if X_shap is None:
        # try to get expected feature names from the model
        feat_names = None
        if hasattr(model, 'feature_names_in_') and getattr(model, 'feature_names_in_') is not None:
            try:
                feat_names = list(model.feature_names_in_)
            except Exception:
                feat_names = None
        # fallback to results top_features
        if feat_names is None:
            feat_names = results.get('top_features')
        if feat_names is None:
            feat_names = list(X_all.columns)

        # Build X_shap by resolving each expected feature name to a column in X_all or a base mapping
        cols = []
        for fname in feat_names:
            if fname in X_all.columns:
                cols.append(fname)
            else:
                # try base name
                b = base_feature_name(fname)
                if b in X_all.columns:
                    cols.append(b)
                else:
                    # try any column that contains the base token
                    candidates = [c for c in X_all.columns if b in str(c)]
                    if candidates:
                        cols.append(candidates[0])
                    else:
                        # will create zero column later
                        cols.append(None)

        # create DataFrame with same number of rows as X_all but only selected columns (or zeros)
        df_sel = pd.DataFrame(index=X_all.index)
        for orig, mapped in zip(feat_names, cols):
            if mapped is None:
                df_sel[orig] = 0.0
            else:
                df_sel[orig] = pd.to_numeric(X_all[mapped], errors='coerce').fillna(0)

        # sample for SHAP speed; take up to 1000 rows
        if len(df_sel) == 0:
            raise RuntimeError('Could not construct feature matrix for SHAP from source data')
        X_shap = df_sel.sample(n=min(1000, len(df_sel)), random_state=42)

    try:
        if chosen in ('xgb','rf','cat'):
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X_shap)
            # handle list or array shapes
            if isinstance(shap_values, list):
                # multiclass -> average
                shap_arr = np.array(shap_values).mean(axis=0)
            elif hasattr(shap_values, 'values'):
                # new API: shap.Explanation object
                shap_arr = np.array(shap_values.values)
            else:
                shap_arr = np.array(shap_values)
        else:
            expl = shap.LinearExplainer(model, X_shap, feature_perturbation='interventional')
            shap_values = expl.shap_values(X_shap)
            shap_arr = np.array(shap_values)
    except Exception as e:
        print('SHAP explainer failed:', e)
        raise

    # Create a shap.Explanation object when possible for plotting convenience
    try:
        exp = shap.Explanation(values=shap_arr, base_values=np.zeros(shap_arr.shape[0]) if shap_arr.ndim==2 else None, data=X_shap.values, feature_names=X_shap.columns.tolist())
    except Exception:
        exp = None

    os.makedirs(LOGS, exist_ok=True)

    # 1) Beeswarm / summary (shap.plots.beeswarm)
    try:
        plt.figure(figsize=(10,6))
        # shap.plots.beeswarm accepts shap_values or Explanation
        if exp is not None:
            shap.plots.beeswarm(exp, show=False)
        else:
            shap.plots.beeswarm(shap_arr, feature_names=X_shap.columns.tolist(), show=False)
        plt.title('SHAP Summary (beeswarm)')
        plt.tight_layout()
        plt.savefig(OUT_BEE, dpi=300)
        plt.close()
        print('Saved beeswarm to', OUT_BEE)
    except Exception as e:
        print('Beeswarm plot failed:', e)

    # 2) Bar plot of mean |SHAP|
    try:
        mean_abs = np.abs(shap_arr).mean(axis=0)
        s = pd.Series(mean_abs, index=X_shap.columns).sort_values(ascending=False)
        plt.figure(figsize=(10,6))
        s.plot(kind='bar', color='purple')
        plt.title('Mean |SHAP| per feature')
        plt.ylabel('Mean |SHAP|')
        plt.tight_layout()
        plt.savefig(OUT_BAR, dpi=300)
        plt.close()
        print('Saved bar plot to', OUT_BAR)
    except Exception as e:
        print('Bar plot failed:', e)

    # 3) Dependence plot for top feature
    try:
        top_feat = s.index[0]
        fig = plt.figure(figsize=(8,6))
        # shap.dependence_plot will draw into current figure
        shap.dependence_plot(top_feat, shap_arr, X_shap, show=False)
        plt.title(f'Dependence: {top_feat}')
        plt.tight_layout()
        out_dep = OUT_DEP_FMT.format(feat=top_feat)
        plt.savefig(out_dep, dpi=300)
        plt.close()
        print('Saved dependence plot to', out_dep)
    except Exception as e:
        print('Dependence plot failed:', e)

    # 4) Waterfall (force-like) for a representative sample: choose sample with largest absolute sum of SHAP
    try:
        sample_idx = int(np.argmax(np.abs(shap_arr).sum(axis=1)))
        # shap.plots.waterfall expects an Explanation for single sample; construct
        try:
            single_exp = shap.Explanation(values=shap_arr[sample_idx], base_values=expl.expected_value if hasattr(expl,'expected_value') else None, data=X_shap.iloc[sample_idx].values, feature_names=X_shap.columns.tolist())
            plt.figure(figsize=(8,6))
            shap.plots.waterfall(single_exp, show=False)
            out_wf = OUT_WF_FMT.format(idx=sample_idx)
            plt.tight_layout()
            plt.savefig(out_wf, dpi=300)
            plt.close()
            print('Saved waterfall (force-like) to', out_wf)
        except Exception as e:
            print('Waterfall plotting via Explanation failed:', e)
    except Exception as e:
        print('Waterfall selection/plot failed:', e)

    print('Done. SHAP images in logs/')

if __name__ == '__main__':
    main()
