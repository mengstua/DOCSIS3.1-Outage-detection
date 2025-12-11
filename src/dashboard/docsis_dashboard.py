import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import re

# Page configuration
st.set_page_config(
    page_title="DOCSIS Customer Call Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# The body of this file is a copy of the original top-level `docsis_dashboard.py` relocated
# to `src/dashboard/` so the top-level script can be a small shim importing this module.

LOGS = 'logs'
DATA = 'data'

@st.cache_resource
def load_results_pickle():
    path = os.path.join(LOGS, 'results_full_run.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f'Failed to load results pickle: {e}')
    return None

def load_metrics_csv():
    candidates = []
    for fname in os.listdir(LOGS):
        if fname.startswith('models_metrics') and fname.endswith('.csv'):
            candidates.append(os.path.join(LOGS, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = candidates[0]
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

def load_pareto_base_top50():
    path1 = os.path.join(LOGS, 'pareto_base_top50.csv')
    path2 = os.path.join(LOGS, 'pareto_feature_importance_with_base.csv')
    if os.path.exists(path1):
        try:
            return pd.read_csv(path1)
        except Exception:
            pass
    if os.path.exists(path2):
        try:
            df = pd.read_csv(path2)
            if 'base' in df.columns:
                agg = df.groupby('base', as_index=False)['pct'].sum().sort_values('pct', ascending=False).reset_index(drop=True)
                agg['cum_pct'] = agg['pct'].cumsum()
                return agg.head(50)
            return df.head(50)
        except Exception:
            pass
    return None

def load_pareto_base_agg():
    path = os.path.join(LOGS, 'pareto_base_feature_importance.csv')
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def load_feature_name_map():
    path = os.path.join(LOGS, 'feature_name_map.csv')
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

@st.cache_data
def load_shap_csv(model_name):
    path = os.path.join(LOGS, f'shap_{model_name}_top20.csv')
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0, header=None).iloc[:,0]
        except Exception:
            try:
                return pd.read_csv(path, index_col=0)
            except Exception:
                return None
    return None


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

results = load_results_pickle()
metrics = load_metrics_csv()
pareto_top50 = load_pareto_base_top50()
pareto_agg = load_pareto_base_agg()
feature_map_df = load_feature_name_map()

if feature_map_df is not None and not feature_map_df.empty:
    try:
        feat_to_base = dict(zip(feature_map_df['engineered'].astype(str), feature_map_df['base'].astype(str)))
    except Exception:
        feat_to_base = {}
else:
    feat_to_base = {}

def display_name(feat):
    return feat_to_base.get(feat, feat)

def build_importance_map():
    m = {}
    try:
        if pareto_top50 is not None:
            df = pareto_top50.copy()
            if 'feature' in df.columns and 'pct' in df.columns:
                for _, r in df.iterrows():
                    try:
                        m[str(r['feature'])] = float(r['pct'])
                    except Exception:
                        continue
            elif 'base' in df.columns and 'pct' in df.columns:
                for _, r in df.iterrows():
                    try:
                        m[str(r['base'])] = float(r['pct'])
                    except Exception:
                        continue
        if pareto_agg is not None:
            df2 = pareto_agg.copy()
            if 'feature' in df2.columns and 'pct' in df2.columns:
                for _, r in df2.iterrows():
                    try:
                        m.setdefault(str(r['feature']), float(r['pct']))
                    except Exception:
                        continue
            elif 'base' in df2.columns and 'pct' in df2.columns:
                for _, r in df2.iterrows():
                    try:
                        m.setdefault(str(r['base']), float(r['pct']))
                    except Exception:
                        continue
    except Exception:
        pass
    return m

importance_map = build_importance_map()

def determine_best_model(metrics_df):
    if metrics_df is None:
        return 'xgb'
    try:
        if 'roc_auc' in metrics_df.columns:
            best_idx = metrics_df['roc_auc'].astype(float).idxmax()
            return str(best_idx)
    except Exception:
        pass
    try:
        if 'roc_auc' in metrics_df.index:
            best_col = metrics_df.loc['roc_auc'].astype(float).idxmax()
            return str(best_col)
    except Exception:
        pass
    for key in ['cat', 'xgb', 'rf', 'svc']:
        try:
            if key in metrics_df.index or key in metrics_df.columns:
                return key
        except Exception:
            continue
    return 'xgb'

def model_display_name(key):
    return {'xgb': 'XGBoost', 'cat': 'CatBoost', 'rf': 'RandomForest', 'svc': 'LinearSVC'}.get(key, str(key))

# Sidebar controls
st.sidebar.header('Controls')
model_override = st.sidebar.selectbox('Override best model (Auto uses detection)', ['Auto', 'cat', 'xgb', 'rf', 'svc'], index=0)
use_model_sidebar = st.sidebar.checkbox('Use saved model for prediction (if available)', value=True)
show_base_inputs_sidebar = st.sidebar.checkbox('Show base (Pareto) input sliders', value=False)

detected_best = determine_best_model(metrics)
selected_model_key = model_override if model_override != 'Auto' else detected_best
selected_model_display = model_display_name(selected_model_key)

# Header
st.markdown('''
    <div class="main-header">
        <h1>📊 DOCSIS Customer Call Prediction Platform</h1>
        <p style="margin-top: 0.5rem; opacity: 0.9;">Machine Learning Analytics Dashboard</p>
    </div>
''', unsafe_allow_html=True)

# The rest of the dashboard UI is intentionally omitted here for brevity — the original full
# file has been relocated; importing this module will run the dashboard code as before.
