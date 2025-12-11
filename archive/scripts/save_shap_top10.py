"""
Original archived: save_shap_top10.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGS = 'logs'
OUT_PNG = os.path.join(LOGS, 'shap_top10.png')

def try_load_results():
    path = os.path.join(LOGS, 'results_full_run.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def try_load_joblib(fname):
    import joblib
    if os.path.exists(fname):
        try:
            return joblib.load(fname)
        except Exception:
            return None
    return None

def get_model_and_data():
    res = try_load_results()
    if res is not None:
        model = None
        for k in ['xgb', 'xgb_model', 'xgb_model_final', 'xgb_model_bal', 'cat', 'cat_model', 'cat_model_final']:
            if k in res and res.get(k) is not None:
                model = res.get(k)
                break
        if model is None:
            for k, v in res.items():
                if hasattr(v, 'predict_proba'):
                    model = v
                    break

        X_train = res.get('X_train') if isinstance(res.get('X_train'), pd.DataFrame) else None
        return model, X_train

    for fname in ['xgb_model.pkl', 'xgb_model_final.pkl', 'xgb_model_bal.pkl', 'xgb_model_bal.joblib']:
        m = try_load_joblib(fname)
        if m is not None:
            res2 = try_load_results()
            X_train = res2.get('X_train') if res2 and isinstance(res2.get('X_train'), pd.DataFrame) else None
            return m, X_train

    return None, None

def main():
    model, X_train = get_model_and_data()
    if model is None:
        print('No model found in `results_full_run.pkl` or common pkl files. Please run training or supply model artifacts.')
        sys.exit(1)

    if X_train is None:
        possible = os.path.join(LOGS, 'X_train_sample.csv')
        if os.path.exists(possible):
            try:
                X_train = pd.read_csv(possible)
            except Exception:
                X_train = None

    if X_train is None:
        cols = None
        if hasattr(model, 'feature_names_in_'):
            try:
                cols = list(model.feature_names_in_)
            except Exception:
                cols = None
        if cols is not None:
            X_train = pd.DataFrame(np.zeros((50, len(cols))), columns=cols)
        else:
            print('Unable to determine feature names for X. Aborting.')
            sys.exit(1)

    n_samples = min(1000, max(50, int(len(X_train) * 0.1)))
    try:
        sample_X = X_train.sample(n_samples, random_state=42)
    except Exception:
        sample_X = X_train.iloc[:n_samples]

    try:
        import shap
    except Exception:
        print('shap package not installed. Please pip install shap')
        sys.exit(1)

    try:
        explainer = None
        est = model
        if hasattr(model, 'named_steps'):
            try:
                est = model.named_steps.get('clf', model.named_steps[list(model.named_steps.keys())[-1]])
            except Exception:
                est = model

        try:
            explainer = shap.TreeExplainer(est)
        except Exception:
            try:
                explainer = shap.Explainer(est)
            except Exception as e:
                print('Failed to create SHAP explainer:', e)
                sys.exit(1)

        try:
            shap_out = explainer(sample_X)
            shap_values = shap_out.values
        except Exception:
            shap_values = explainer.shap_values(sample_X)

    except Exception as e:
        print('Error computing SHAP values:', e)
        sys.exit(1)

    if isinstance(shap_values, list) or isinstance(shap_values, tuple):
        shap_vals_arr = shap_values[0]
    else:
        shap_vals_arr = shap_values

    mean_abs = np.abs(shap_vals_arr).mean(axis=0)
    feat_names = list(sample_X.columns)
    imp_df = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': mean_abs})
    imp_df = imp_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    top10 = imp_df.head(10)['feature'].tolist()
    print('Top-10 features by mean |SHAP|:')
    for i, f in enumerate(top10, 1):
        print(f'{i}. {f}')

    try:
        idxs = [feat_names.index(f) for f in top10]
        shap_subset = shap_vals_arr[:, idxs]
        X_subset = sample_X[top10]

        plt.figure(figsize=(8, 6))
        try:
            shap.summary_plot(shap_subset, X_subset, feature_names=top10, show=False)
            plt.tight_layout()
            os.makedirs(LOGS, exist_ok=True)
            plt.savefig(OUT_PNG, dpi=300)
            print('Saved SHAP summary (top10) to:', OUT_PNG)
        except Exception as e:
            print('shap.summary_plot failed, falling back to bar plot:', e)
            plt.clf()
            plt.barh(imp_df.head(10)['feature'][::-1], imp_df.head(10)['mean_abs_shap'][::-1], color='purple')
            plt.xlabel('Mean |SHAP value|')
            plt.title('Top 10 Feature Importance (mean |SHAP|)')
            plt.tight_layout()
            plt.savefig(OUT_PNG, dpi=300)
            print('Saved fallback SHAP bar chart to:', OUT_PNG)

    except Exception as e:
        print('Failed to create/save SHAP plot:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
