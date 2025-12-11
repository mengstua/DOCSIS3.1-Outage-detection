import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

RESULTS_PKL = os.path.join('logs','results_full_run.pkl')
OUT_DIR = 'logs'
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(RESULTS_PKL):
    raise FileNotFoundError(RESULTS_PKL)

with open(RESULTS_PKL,'rb') as f:
    results = pickle.load(f)

print('Loaded results keys:', list(results.keys()))

# If X_test is missing, reconstruct features from CSV using the same cleaning and TS functions
# We'll inline minimal versions of the cleaning + TS feature functions used in the notebook.
import numpy as np

# --- Inlineed helpers (copied/adapted from notebook) --------------------------------
from sklearn.preprocessing import LabelEncoder

def auto_clean_dataframe(df, time_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    sentinel_ts = pd.Timestamp("1970-01-01")
    df[time_col] = df[time_col].fillna(sentinel_ts)

    for col in df.columns:
        try:
            if df[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                df[col] = df[col].astype(str)
        except Exception:
            pass

    for col in df.columns:
        if col == time_col:
            continue
        if df[col].dtype == "object":
            df[col] = (df[col].astype(str).str.replace(r"[^\d\.\-eE+]+", "", regex=True))
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.7:
                df[col] = converted
            else:
                try:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                except Exception:
                    df[col] = df[col].astype(str)
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].astype(np.float32)
    return df


def add_time_features(df, id_col="cpe_hk_hashed", time_col="snapshot_date", feature_cols=None, lags=(1,3,6), windows=(3,6,12)):
    df = df.copy()
    df = df.sort_values([id_col, time_col])
    non_ts_cols = {"cmts_id", "cpe_model", "node_id", "freq_plan_id", "docsis_version"}

    if feature_cols is None:
        numeric_cols = []
        for col in df.columns:
            if col in {id_col, time_col}:
                continue
            converted = pd.to_numeric(df[col], errors='coerce')
            valid_fraction = 1 - converted.isna().mean()
            if valid_fraction >= 0.90:
                numeric_cols.append(col)
        feature_cols = [col for col in numeric_cols if col not in non_ts_cols]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().any():
            df[c] = df[c].fillna(0).astype(np.float32)

    g = df.groupby(id_col, group_keys=False)
    for col in feature_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        for w in windows:
            roll = g[col].shift(1).rolling(w, min_periods=1)
            df[f"{col}_rollmean{w}"] = roll.mean()
            df[f"{col}_rollstd{w}"] = roll.std()
        try:
            df[f"{col}_delta1"] = g[col] - g[col].shift(1)
        except Exception:
            s = pd.to_numeric(df[col].astype(str).str.replace(r'[^\\d\\.\\-eE+]','', regex=True), errors='coerce').fillna(0).astype(np.float32)
            df[f"{col}_delta1"] = s - s.groupby(df[id_col]).shift(1)
    return df

# ------------------------------------------------------------------------------------

# Try to use X_test if in results, else rebuild from data
X_test_f = None
X_train_f = None
if 'X_test' in results:
    X_test_f = results['X_test']
if 'X_train' in results:
    X_train_f = results['X_train']

if X_test_f is None:
    # Rebuild using CSV — may be slow
    print('X_test not found in results; rebuilding features from CSV (this can take time)...')
    df = pd.read_csv(os.path.join('data','ml_master_table_1M.csv'), low_memory=False)
    df = auto_clean_dataframe(df, 'snapshot_date')
    df_ts = add_time_features(df)
    # Ensure target exists and fill
    df_ts['call_flag'] = df_ts.get('call_flag', 0).fillna(0).astype(int)
    y = df_ts['call_flag'].astype(int)
    X = df_ts.drop(columns=['call_flag','cpe_hk_hashed','snapshot_date'], errors='ignore')

    # replicate conditional stratify logic
    test_size = 0.2
    strat = None
    try:
        vc = y.value_counts()
        expected_test_per_class = (vc * test_size).astype(int)
        if expected_test_per_class.min() >= 1:
            strat = y
    except Exception:
        strat = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)

    # Load results top_features and scaler if present
    top_features = results.get('top_features')
    scaler = results.get('scaler')
    if top_features is None:
        raise RuntimeError('top_features not found in results; cannot select features')

    # select features
    X_train_f = X_train[top_features]
    X_test_f = X_test[top_features]

    # scale using saved scaler if available, else fit new
    if scaler is not None:
        try:
            # If scaler stores feature names, ensure the order matches
            feat_names = getattr(scaler, 'feature_names_in_', None)
            if feat_names is not None:
                missing = [f for f in feat_names if f not in X_train_f.columns]
                if missing:
                    raise ValueError(f"Scaler expects features {len(feat_names)}, missing {len(missing)}: {missing[:10]}")
                X_train_f = pd.DataFrame(scaler.transform(X_train_f[feat_names]), columns=feat_names)
                X_test_f = pd.DataFrame(scaler.transform(X_test_f[feat_names]), columns=feat_names)
            else:
                X_train_f = pd.DataFrame(scaler.transform(X_train_f), columns=X_train_f.columns)
                X_test_f = pd.DataFrame(scaler.transform(X_test_f), columns=X_test_f.columns)
        except Exception as e:
            print('Saved scaler incompatible or transform failed, fitting a new scaler. Error:', e)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train_f = pd.DataFrame(sc.fit_transform(X_train_f), columns=X_train_f.columns)
            X_test_f = pd.DataFrame(sc.transform(X_test_f), columns=X_test_f.columns)
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train_f = pd.DataFrame(sc.fit_transform(X_train_f), columns=X_train_f.columns)
        X_test_f = pd.DataFrame(sc.transform(X_test_f), columns=X_test_f.columns)

    y_test_series = y_test
else:
    # If X_test present, use results' objects
    y_test_series = results.get('y_test')

print('Shapes: X_train_f', getattr(X_train_f, 'shape', None), 'X_test_f', getattr(X_test_f, 'shape', None), 'y_test', getattr(y_test_series, 'shape', None))

# Load models
models = {}
for name in ['xgb','cat','rf','svc']:
    if name in results:
        models[name] = results[name]

# Predictions and metrics
metrics_list = []
plt.figure(figsize=(8,6))
for name, model in models.items():
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test_f)[:,1]
        else:
            # fallback to decision_function scaled
            raw = model.decision_function(X_test_f)
            proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    except Exception as e:
        print(f'Failed to get probabilities for {name}:', e)
        continue
    fpr, tpr, _ = roc_curve(y_test_series, proba)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_test_series, proba)
    # compute optimal threshold by F0.5 similar to notebook
    precision, recall, thresholds = precision_recall_curve(y_test_series, proba)
    precision = precision[:-1]; recall = recall[:-1]
    beta = 0.5
    beta2 = beta**2
    f = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + 1e-9)
    idx = np.argmax(f)
    thr = thresholds[idx]
    pred = (proba >= thr).astype(int)
    prec = precision_score(y_test_series, pred)
    rec = recall_score(y_test_series, pred)
    f1 = f1_score(y_test_series, pred)
    metrics_list.append({'model':name,'roc_auc':roc_auc,'pr_auc':ap,'threshold':float(thr),'precision':float(prec),'recall':float(rec),'f1':float(f1)})
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'models_roc.png'))
print('Saved ROC plot to logs/models_roc.png')

metrics_df = pd.DataFrame(metrics_list).set_index('model')
metrics_df.to_csv(os.path.join(OUT_DIR,'models_metrics.csv'))
print('Saved model metrics to logs/models_metrics.csv')

# Pareto of feature importances
# For each model, get feature importances or coef_ absolute values
feat_imp = pd.DataFrame(index=results['top_features'])
for name, model in models.items():
    vals = None
    if hasattr(model, 'feature_importances_'):
        try:
            vals = np.abs(model.feature_importances_)
        except Exception:
            vals = None
    if vals is None and hasattr(model, 'coef_'):
        try:
            coef = np.array(model.coef_).ravel()
            vals = np.abs(coef)
        except Exception:
            vals = None
    if vals is None:
        # last resort: uniform
        vals = np.ones(len(results['top_features']))
    # normalize to percentage
    vals = np.array(vals)
    vals = vals / (vals.sum() + 1e-12) * 100.0
    feat_imp[name] = vals

# Average importance across models
feat_imp['mean_pct'] = feat_imp.mean(axis=1)
feat_imp_sorted = feat_imp['mean_pct'].sort_values(ascending=False)
pareto = feat_imp_sorted.cumsum()
pareto_df = pd.DataFrame({'feature':feat_imp_sorted.index, 'pct':feat_imp_sorted.values, 'cum_pct':pareto.values})
pareto_df.to_csv(os.path.join(OUT_DIR,'pareto_feature_importance.csv'), index=False)

# Pareto plot
plt.figure(figsize=(10,6))
plt.bar(range(len(pareto_df)), pareto_df['pct'], label='Feature %')
plt.plot(range(len(pareto_df)), pareto_df['cum_pct'], color='red', marker='o', label='Cumulative %')
plt.axhline(80, color='green', linestyle='--', label='80%')
plt.xticks(range(len(pareto_df)), pareto_df['feature'], rotation=90)
plt.ylabel('Percentage')
plt.title('Pareto of Mean Feature Importance (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'pareto.png'))
print('Saved pareto plot to logs/pareto.png')

# Determine features needed to reach 80%
n = np.searchsorted(pareto_df['cum_pct'].values, 80) + 1
print(f'{n} features required to reach >=80% cumulative mean importance')

# Aggregate engineered features back to original base variables for Pareto
import re
TARGET = 'call_flag'
ID_COLS = {'cpe_hk_hashed', 'snapshot_date'}

def base_feature_name(feat):
    # remove common engineered suffixes: _lag#, _rollmean#, _rollstd#, _delta#
    return re.sub(r'(_lag\d+|_rollmean\d+|_rollstd\d+|_delta\d+)$', '', feat)

base_map = [base_feature_name(f) for f in pareto_df['feature'].tolist()]
base_df = pd.DataFrame({'engineered': pareto_df['feature'], 'base': base_map, 'pct': pareto_df['pct'], 'cum_pct': pareto_df['cum_pct']})

# Exclude target and id/time base features
base_df = base_df[~base_df['base'].isin({TARGET})]
base_df = base_df[~base_df['base'].isin(ID_COLS)]

# aggregate by base
agg = base_df.groupby('base', as_index=False)['pct'].sum()
agg = agg.sort_values('pct', ascending=False).reset_index(drop=True)
agg['cum_pct'] = agg['pct'].cumsum()
agg.to_csv(os.path.join(OUT_DIR,'pareto_base_feature_importance.csv'), index=False)
print('Saved aggregated Pareto by base features to logs/pareto_base_feature_importance.csv')

# Save engineered->base mapping for dashboard / notebook convenience
base_df.to_csv(os.path.join(OUT_DIR, 'feature_name_map.csv'), index=False)
print('Saved engineered->base feature mapping to logs/feature_name_map.csv')

# Also save a full pareto CSV that includes the base name alongside engineered features
pareto_with_base = pareto_df.copy()
pareto_with_base['base'] = base_map
pareto_with_base.to_csv(os.path.join(OUT_DIR, 'pareto_feature_importance_with_base.csv'), index=False)
print('Saved pareto with base names to logs/pareto_feature_importance_with_base.csv')

# Save list of all considered base variables (after filtering)
considered_base = agg['base'].tolist()
pd.Series(considered_base, name='base_feature').to_csv(os.path.join(OUT_DIR,'considered_base_features.csv'), index=False)
print('Saved list of considered base variables to logs/considered_base_features.csv')

# Create Pareto for top ~50 base features (or fewer if not available)
top_n = min(50, len(agg))
top50 = agg.head(top_n).copy()
top50.to_csv(os.path.join(OUT_DIR, 'pareto_base_top50.csv'), index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.bar(range(len(top50)), top50['pct'], label='Feature %')
plt.plot(range(len(top50)), top50['cum_pct'], color='red', marker='o', label='Cumulative %')
plt.axhline(80, color='green', linestyle='--', label='80%')
plt.xticks(range(len(top50)), top50['base'], rotation=90)
plt.ylabel('Percentage')
plt.title(f'Pareto (Top {top_n} base features)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pareto_base_top50.png'))
print(f'Saved Pareto top {top_n} base features to logs/pareto_base_top50.png and CSV')

# Determine minimal set within top50 that reaches >=80% cumulative
idx80 = np.searchsorted(top50['cum_pct'].values, 80)
k80 = idx80 + 1
selected_80 = top50.head(k80)['base'].tolist()
# Filter selected_80 to remove engineered/target-derived names just in case
def is_engineered_or_target(x):
    lx = str(x).lower()
    if 'call_flag' in lx or lx.startswith('target') or lx in ('call_flag', 'target', 'y'):
        return True
    if any(s in lx for s in ['_lag', 'rollmean', 'rollstd', 'roll', '_p', '_ma', '_delta', '_diff']):
        return True
    return False

cleaned_selected_80 = [s for s in selected_80 if not is_engineered_or_target(s)]
final_selected_80 = cleaned_selected_80 if len(cleaned_selected_80) > 0 else selected_80
pd.Series(final_selected_80, name='selected_80pct_vars').to_csv(os.path.join(OUT_DIR,'selected_pareto_80pct_vars.csv'), index=False)
print(f'{len(final_selected_80)} base features (within top{top_n}) required to reach >=80% cumulative mean importance')

# Also compute top-20% of top50 (rounded)
top20pct_count = max(1, int(round(0.2 * top_n)))
top20pct = top50.head(top20pct_count)['base'].tolist()
pd.Series(top20pct, name='top_20pct_of_top50').to_csv(os.path.join(OUT_DIR,'selected_top20pct_of_top50.csv'), index=False)
coverage_top20pct = top50.head(top20pct_count)['pct'].sum()
print(f'Top {top20pct_count} features (20% of {top_n}) cover {coverage_top20pct:.2f}% of mean importance')

# SHAP on top 20 variables (use sample up to 1000 rows from X_test_f)
top20 = results['top_features'][:20]
X_shap = X_test_f[top20].sample(n=min(1000, len(X_test_f)), random_state=42)
shap_out = {}
for name, model in models.items():
    try:
        print('\nComputing SHAP for', name)
        if name in ['xgb','rf','cat']:
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X_shap)
            # shap_values can be (n_samples,n_features) or list for multiclass
            if isinstance(sv, list):
                sv = np.array(sv).mean(axis=0)
            elif sv.ndim == 3:
                sv = sv.mean(axis=0)
        else:
            # linear model
            expl = shap.LinearExplainer(model, X_shap, feature_perturbation='interventional')
            sv = expl.shap_values(X_shap)
        mean_abs = np.abs(sv).mean(axis=0)
        # Map engineered feature names to base names and aggregate SHAP importances
        # so saved SHAP CSVs contain base feature names only (no engineered suffixes)
        def base_feature_name(feat):
            return re.sub(r'(_lag\d+|_rollmean\d+|_rollstd\d+|_delta\d+|_p\d+)$', '', feat)

        feat_bases = [base_feature_name(f) for f in top20]
        shp_series = pd.Series(mean_abs, index=feat_bases)
        # aggregate by base feature name (sum mean-abs across engineered variants)
        shp_agg = shp_series.groupby(shp_series.index).sum().sort_values(ascending=False)
        shp_agg.to_csv(os.path.join(OUT_DIR,f'shap_{name}_top20.csv'))
        shp_df = shp_agg
        shap_out[name] = shp_df
        print(f'Saved SHAP importances for {name} to logs/shap_{name}_top20.csv')
    except Exception as e:
        print('SHAP failed for', name, e)

# Combined SHAP bar plot for top features (mean across models that succeeded)
if shap_out:
    combined = pd.DataFrame(shap_out).fillna(0)
    combined_mean = combined.mean(axis=1).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    combined_mean.plot(kind='bar')
    plt.title('Mean |SHAP| for Top20 features (average across models)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'shap_top20_mean.png'))
    print('Saved combined SHAP plot to logs/shap_top20_mean.png')


# Optionally run the advanced SHAP generator script to create beeswarm, dependence, and waterfall plots
# Use `--no-shap` on the command line to skip this step.
import sys
import subprocess

if '--no-shap' not in sys.argv:
    try:
        # pick best model by F1 if available, else fallback to xgb if present
        best_model_name = None
        try:
            if 'metrics_df' in globals() and not metrics_df.empty:
                best_model_name = metrics_df['f1'].idxmax()
        except Exception:
            best_model_name = None
        if not best_model_name or best_model_name not in models:
            best_model_name = 'xgb' if 'xgb' in models else next(iter(models.keys()), None)

        gen_script = os.path.join('scripts', 'generate_shap_plots.py')
        cmd = [sys.executable, gen_script]
        if best_model_name:
            cmd += ['--model', best_model_name]

        print('Invoking SHAP generator:', ' '.join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            print('SHAP generator completed successfully.')
            if proc.stdout:
                print(proc.stdout)
        else:
            print('SHAP generator exited with code', proc.returncode)
            if proc.stdout:
                print('stdout:\n', proc.stdout)
            if proc.stderr:
                print('stderr:\n', proc.stderr)
    except Exception as e:
        print('Failed to run SHAP generator:', e)

print('\nAll done. Outputs in logs/: models_metrics.csv, models_roc.png, pareto_feature_importance.csv, pareto.png, shap_* files')
