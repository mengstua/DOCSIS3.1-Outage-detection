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

# Custom CSS for Orange branding
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #ff6600 0%, #ff8533 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border-left: 6px solid #ff6600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 88px;
    }
    .metric-card .label {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 2.25rem;
        font-weight: 600;
        margin-top: 0.1rem;
        margin-bottom: 0.25rem;
        color: #222;
    }
    .metric-card .caption {
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6600;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

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
    # Pick the most recent metrics CSV matching possible filenames (allow recomputed audit copies)
    candidates = []
    for fname in os.listdir(LOGS):
        if fname.startswith('models_metrics') and fname.endswith('.csv'):
            candidates.append(os.path.join(LOGS, fname))
    if not candidates:
        return None
    # pick newest by modification time
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
    # Prefer explicit pareto_base_top50, else try pareto_feature_importance_with_base
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
            # prefer aggregated base column if present
            if 'base' in df.columns:
                # aggregate by base to produce similar top50
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
    """Robustly fetch a metric value from metrics DataFrame.

    Handles both orientations:
    - index=model, columns=metrics (default written by scripts)
    - index=metric, columns=model (alternative layouts)
    Also checks alternative metric names provided in `alt_names`.
    Returns `default` when not found or conversion fails.
    """
    if metrics_df is None:
        return default

    candidates = [metric_name]
    if alt_names:
        candidates = candidates + list(alt_names)

    # Try index=model, columns=metric
    for m in candidates:
        if model_key in metrics_df.index and m in metrics_df.columns:
            try:
                return float(metrics_df.loc[model_key, m])
            except Exception:
                pass

    # Try index=metric, columns=model
    for m in candidates:
        if m in metrics_df.index and model_key in metrics_df.columns:
            try:
                return float(metrics_df.loc[m, model_key])
            except Exception:
                pass

    # Try flattened column like 'xgb_f1' if present
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

# build a fast mapping dict engineered -> base for display purposes
if feature_map_df is not None and not feature_map_df.empty:
    try:
        feat_to_base = dict(zip(feature_map_df['engineered'].astype(str), feature_map_df['base'].astype(str)))
    except Exception:
        feat_to_base = {}
else:
    feat_to_base = {}

def display_name(feat):
    # prefer base name if available, else return original
    return feat_to_base.get(feat, feat)


# Build a mapping feature -> importance (prefer pareto outputs)
def build_importance_map():
    m = {}
    try:
        if pareto_top50 is not None:
            df = pareto_top50.copy()
            # prefer columns 'feature' and 'pct' or 'base' and 'pct'
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

# --- Best model detection and UI controls (sidebar) ---
def determine_best_model(metrics_df):
    """Return the internal model key with highest ROC AUC ('xgb','cat','rf','svc')."""
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

# Determine selected model key (apply override if chosen)
detected_best = determine_best_model(metrics)
selected_model_key = model_override if model_override != 'Auto' else detected_best
selected_model_display = model_display_name(selected_model_key)


# Header
st.markdown("""
    <div class="main-header">
        <h1>📊 DOCSIS Customer Call Prediction Platform</h1>
        <p style="margin-top: 0.5rem; opacity: 0.9;">Machine Learning Analytics Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
# Use precomputed selected model key across tabs
best_model_key = selected_model_key
best_model = selected_model_display

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔄 Model Comparison", "🔍 Feature Analysis", "⚙️ Prediction Tool"]) 

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    # Use the sidebar-selected model (allows user override) for Overview
    best_model = selected_model_display
    best_model_key = selected_model_key

    with col1:
        # Render custom metric card with controlled HTML so left border indicator aligns
        if metrics is not None:
            roc_val = safe_metric(metrics, best_model_key, 'roc_auc', alt_names=['roc_auc'])
        else:
            roc_val = None
        card_html = f"""
        <div class="metric-card">
          <div class="label">Best Model</div>
          <div class="value">{best_model}</div>
          <div class="caption">{f'ROC AUC: {roc_val:.3f}' if roc_val is not None else ''}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    with col2:
        if metrics is not None:
            roc_val = safe_metric(metrics, best_model_key, 'roc_auc', alt_names=['roc_auc'])
        else:
            roc_val = None
        perf_val = f"{roc_val*100:.1f}%" if roc_val is not None else 'n/a'
        card_html = f"""
        <div class="metric-card">
          <div class="label">Model Performance</div>
          <div class="value">{perf_val}</div>
          <div class="caption">Area Under ROC Curve</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    with col3:
        if metrics is not None:
            precision = safe_metric(metrics, best_model_key, 'precision', alt_names=['precision','precision_score','prec'])
        else:
            precision = None
        prec_val = f"{precision:.3f}" if precision is not None else 'n/a'
        card_html = f"""
        <div class="metric-card">
          <div class="label">Precision</div>
          <div class="value">{prec_val}</div>
          <div class="caption">Positive predictive value</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    with col4:
        if metrics is not None:
            best_f1 = safe_metric(metrics, best_model_key, 'f1', alt_names=['best_f1','f1'])
            best_thr = safe_metric(metrics, best_model_key, 'threshold', alt_names=['best_threshold','threshold'])
        else:
            best_f1 = None
            best_thr = None
        f1_val = f"{best_f1:.3f}" if best_f1 is not None else 'n/a'
        thr_txt = f"At threshold {best_thr:.3f}" if best_thr is not None else ''
        card_html = f"""
        <div class="metric-card">
          <div class="label">Best F1 Score</div>
          <div class="value">{f1_val}</div>
          <div class="caption">{thr_txt}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('ROC Curve (saved image)')
    roc_img = os.path.join(LOGS, 'models_roc.png')
    if os.path.exists(roc_img):
        st.image(roc_img, use_column_width=True)
    else:
        st.info('ROC image not found in logs/ — run the analysis script first.')

# ==================== TAB 2: MODEL COMPARISON ====================
with tab2:
    st.subheader('Model Comparison & Metrics')
    if metrics is not None:
        st.dataframe(metrics)
    else:
        st.info('No metrics CSV found in logs/.')

    st.markdown('---')
    st.subheader('Select model to view SHAP top20 (if available)')
    available_shap = [f.name for f in os.scandir(LOGS) if f.name.startswith('shap_')]
    model_choices = ['xgb','cat','rf','svc']
    # default selected model in SHAP view should reflect the detected best model
    default_idx = 0
    try:
        if best_model_key in model_choices:
            default_idx = model_choices.index(best_model_key)
    except Exception:
        default_idx = 0
    chosen = st.selectbox('Model', model_choices, index=default_idx)
    shap_series = load_shap_csv(chosen)
    if shap_series is not None:
        try:
            # shap CSVs may have index=feature and a single column with values
            if isinstance(shap_series, pd.Series):
                df_shap = shap_series.reset_index()
                df_shap.columns = ['feature','mean_abs_shap']
                # map engineered feature names to base names for display
                df_shap['display_feature'] = df_shap['feature'].apply(display_name)
                df_shap = df_shap.set_index('display_feature')[['mean_abs_shap']].rename_axis('feature').reset_index()
            else:
                df_shap = pd.read_csv(os.path.join(LOGS, f'shap_{chosen}_top20.csv'), index_col=0)
                df_shap = df_shap.iloc[:,0].abs().sort_values(ascending=False).reset_index()
                df_shap.columns = ['feature','mean_abs_shap']
                # map engineered names to base for display
                df_shap['display_feature'] = df_shap['feature'].apply(display_name)
                df_shap = df_shap.set_index('display_feature')[['mean_abs_shap']].rename_axis('feature').reset_index()
            st.bar_chart(df_shap.set_index('feature'))
            st.write(df_shap)
        except Exception as e:
            st.warning(f'Could not parse shap CSV: {e}')
    else:
        st.info('No SHAP CSV found for this model in logs/.')

# ==================== TAB 3: FEATURE ANALYSIS ====================
with tab3:
    st.subheader('Pareto — Base Features')
    if pareto_top50 is not None:
        st.write('Top base features used for Pareto (up to 50)')
        # If pareto_top50 already has a 'base' column use that; else map engineered feature names
        df_display = pareto_top50.copy()
        if 'base' not in df_display.columns and 'feature' in df_display.columns:
            df_display['base'] = df_display['feature'].apply(display_name)
        st.dataframe(df_display)
        img = os.path.join(LOGS, 'pareto_base_top50.png')
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        else:
            # fallback to the new pareto chart generated by the helper script
            new_img = os.path.join(LOGS, 'pareto_chart.png')
            if os.path.exists(new_img):
                st.image(new_img, use_column_width=True)
            else:
                st.info('Pareto image not found in logs/ — run the analysis script first.')
                # allow user to trigger regeneration from the dashboard
                if st.button('Regenerate Pareto (runs scripts/pareto_chart.py)'):
                    import subprocess
                    try:
                        subprocess.run([os.path.join('.venv','Scripts','python'), 'scripts/pareto_chart.py'], check=True)
                        st.success('Pareto regeneration finished — refresh to view.')
                    except Exception as e:
                        st.error(f'Failed to run pareto script: {e}')
    elif pareto_agg is not None:
        st.write('Aggregated base Pareto')
        df_display = pareto_agg.copy()
        # ensure base column exists for readability
        if 'base' not in df_display.columns and 'feature' in df_display.columns:
            df_display['base'] = df_display['feature'].apply(display_name)
        st.dataframe(df_display.head(100))
    else:
        st.info('Pareto CSVs not found in logs/. Run `scripts/model_compare_and_shap.py` first.')

    st.markdown('---')
    st.subheader('Selected variables reaching >=80% cumulative (within top selection)')
    sel80 = os.path.join(LOGS, 'selected_pareto_80pct_vars.csv')
    top20pct = os.path.join(LOGS, 'selected_top20pct_of_top50.csv')
    if os.path.exists(sel80):
        df80 = pd.read_csv(sel80, header=0)
        st.write(df80)
    else:
        st.info('Selected 80% CSV not found.')
    if os.path.exists(top20pct):
        df20 = pd.read_csv(top20pct, header=0)
        st.write('Top-20% subset of the top-N features:')
        st.write(df20)

    # Show SHAP top-10 summary (image + list) when available
    st.markdown('---')
    st.subheader('SHAP — Top Features (saved)')
    # Prefer the dedicated top10 PNG, else fall back to top20 mean PNG
    shap10_img = os.path.join(LOGS, 'shap_top10.png')
    shap20_mean = os.path.join(LOGS, 'shap_top20_mean.png')
    if os.path.exists(shap10_img):
        st.write('SHAP summary for the top features (saved image)')
        st.image(shap10_img, use_column_width=True)
    elif os.path.exists(shap20_mean):
        st.write('SHAP summary (top20 mean) — saved image')
        st.image(shap20_mean, use_column_width=True)
    else:
        st.info('No SHAP image found in logs/. Run the SHAP script to generate a top-10 plot.')

    # --- Advanced SHAP visualizations (beeswarm, dependence, waterfall) ---
    st.markdown('---')
    st.subheader('Advanced SHAP Visualizations')
    beeswarm_img = os.path.join(LOGS, 'shap_beeswarm.png')
    bar_mean_img = os.path.join(LOGS, 'shap_bar_mean.png')
    # show beeswarm if available else hint to generate
    if os.path.exists(beeswarm_img):
        st.write('SHAP summary (beeswarm)')
        st.image(beeswarm_img, use_column_width=True)
    else:
        st.info('Beeswarm image not found in logs/. Click to generate detailed SHAP plots (may take time).')

    if os.path.exists(bar_mean_img):
        st.write('Bar plot — Mean |SHAP|')
        st.image(bar_mean_img, use_column_width=True)

    # Dependence and waterfall plots (generated per-feature or sample)
    # Look for any existing dependence files and waterfall files and display a small gallery
    dep_files = [f.path for f in os.scandir(LOGS) if f.name.startswith('shap_dependence_') and f.name.endswith('.png')]
    wf_files = [f.path for f in os.scandir(LOGS) if f.name.startswith('shap_waterfall_') and f.name.endswith('.png')]
    if dep_files:
        st.write('Dependence plot (example)')
        st.image(dep_files[0], use_column_width=True)
    if wf_files:
        st.write('Force / Waterfall plot (example)')
        st.image(wf_files[0], use_column_width=True)

    # Allow user to trigger generation of detailed SHAP plots
    with st.expander('Generate detailed SHAP plots (beeswarm, dependence, waterfall)'):
        st.write('This will run `scripts/generate_shap_plots.py` using the saved results in `logs/results_full_run.pkl`.')
        st.write('Note: this requires the `shap` package and may take time to compute (samples up to ~1000 rows).')
        if st.button('Run SHAP plot generator'):
            import subprocess
            try:
                with st.spinner('Generating SHAP plots. This may take a few minutes...'):
                    # run generator synchronously so we can show results after completion
                    subprocess.run([os.path.join('.venv','Scripts','python'), 'scripts/generate_shap_plots.py'], check=True)
                st.success('SHAP plots generated. Refresh this section if images do not appear automatically.')
            except subprocess.CalledProcessError as e:
                st.error(f'Generator failed: {e}')
            except Exception as e:
                st.error(f'Failed to run generator: {e}')

    # Try to show a compact top-10 feature list from available shap CSVs
    # but enforce the canonical allowed feature list so engineered/target-derived
    # features (e.g., call_flag_*, *_lag*, rollmean*) are never shown.
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

    shap_candidates = [f.path for f in os.scandir(LOGS) if f.name.startswith('shap_') and (f.name.endswith('_top20.csv') or f.name.endswith('_top10.csv') or 'top20' in f.name)]
    if shap_candidates:
        feats = []
        try:
            shp = pd.read_csv(shap_candidates[0], index_col=0, header=None)
            if shp.shape[1] >= 1:
                feats = shp.iloc[:,0].astype(str).tolist()
        except Exception:
            try:
                shp = pd.read_csv(shap_candidates[0])
                if 'feature' in shp.columns:
                    feats = shp['feature'].astype(str).tolist()
                else:
                    feats = shp.iloc[:,0].astype(str).tolist()
            except Exception:
                feats = []

        # Map engineered -> base names and then filter to canonical allowed list
        mapped = [display_name(str(f)) for f in feats]

        # Keep only canonical features and preserve canonical order
        cleaned = [f for f in canonical_allowed if f in mapped]

        # If none of the canonical features were present in the SHAP CSV, fall back
        # to a conservative cleaned list (remove obvious engineered/target-derived)
        if not cleaned:
            cleaned = []
            for bf in mapped:
                lx = bf.lower()
                if 'call_flag' in lx or lx.startswith('target') or any(s in lx for s in ['_lag', 'roll', 'rollmean', 'rollstd']):
                    continue
                if bf not in cleaned:
                    cleaned.append(bf)
                if len(cleaned) >= 10:
                    break

        if cleaned:
            st.write('Top features (SHAP-derived, mapped to base names where possible):')
            for i, f in enumerate(cleaned[:10], 1):
                st.write(f'{i}. {f}')
        else:
            st.info('SHAP CSV found but no clean feature names could be extracted.')
    else:
        st.info('No SHAP CSV artifacts found in logs/.')

# ==================== TAB 4: PREDICTION TOOL ====================
with tab4:
    st.markdown('''
    <div style="background: linear-gradient(90deg, #ff6600 0%, #ff8533 100%); 
                padding: 1.2rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
        <h3 style="margin: 0;">🎯 Predict Call Probability</h3>
    </div>
    ''', unsafe_allow_html=True)

    # load top features and model (use best model)
    top_features = results.get('top_features') if results else None
    scaler = results.get('scaler') if results else None
    # pick model corresponding to best_model_key
    model_selected = results.get(best_model_key) if results else None

    st.write('Provide values for the selected features (or use quick presets).')
    # Use sidebar toggles (centralized controls)
    use_model = use_model_sidebar
    show_base_inputs = show_base_inputs_sidebar

    # choose features to input: prefer selected 80pct if available
    sel80_path = os.path.join(LOGS, 'selected_pareto_80pct_vars.csv')
    # Authoritative allowed model features (do not show engineered/lag/roll variants or target-derived features)
    allowed_model_features = [
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
    if os.path.exists(sel80_path):
        # load saved selected features and filter to allowed model features
        try:
            sf = pd.read_csv(sel80_path, header=0).iloc[:,0].astype(str).tolist()
        except Exception:
            sf = []
        # preserve order, keep only allowed features
        input_feats = [f for f in sf if f in allowed_model_features]
        # Ensure we present a reasonably-sized set (prefer 10 canonical features)
        desired_n = 10
        # Fill from the canonical allowed list in order, preserving any selections from the saved file
        for f in allowed_model_features:
            if len(input_feats) >= desired_n:
                break
            if f not in input_feats:
                input_feats.append(f)
        # If still short, try to pull base features from pareto_top50 (respecting allowed list)
        if len(input_feats) < desired_n and pareto_top50 is not None:
            try:
                if 'base' in pareto_top50.columns:
                    base_list = pareto_top50['base'].astype(str).tolist()
                elif 'feature' in pareto_top50.columns:
                    base_list = [display_name(x) for x in pareto_top50['feature'].astype(str).tolist()]
                else:
                    base_list = []
                for f in base_list:
                    if len(input_feats) >= desired_n:
                        break
                    if f in allowed_model_features and f not in input_feats:
                        input_feats.append(f)
            except Exception:
                pass
        # Final fallback: use model top_features (if using model) to fill any remaining spots
        if len(input_feats) < desired_n and top_features is not None:
            for f in list(top_features):
                if len(input_feats) >= desired_n:
                    break
                if f in allowed_model_features and f not in input_feats:
                    input_feats.append(f)
        # Deduplicate and crop to desired length
        input_feats = list(dict.fromkeys(input_feats))[:desired_n]
    elif top_features is not None:
        # When no saved sel80 exists, prefer showing the model's `top_features` but ensure canonical allowed features are present
        desired_n = 10
        input_feats = []
        # If using the model, allow a larger upper bound to inspect more features
        max_from_model = 20 if use_model else 10
        # start with model's top features that are allowed
        for f in list(top_features):
            if f in allowed_model_features and f not in input_feats:
                input_feats.append(f)
            if len(input_feats) >= max_from_model:
                break
        # Ensure canonical allowed order fills to desired size if model top_features are missing some
        for f in allowed_model_features:
            if len(input_feats) >= desired_n:
                break
            if f not in input_feats:
                input_feats.append(f)
        # Deduplicate and limit to a reasonable count
        input_feats = list(dict.fromkeys(input_feats))[:max_from_model]
    else:
        input_feats = ['snr_ds_avg','snr_us_avg','p95_rtt','jitter','fec31_uncor_sum']

    # create inputs as sliders and compute prediction live
    # Helper to get sensible ranges from saved training data when available
    def _get_range_for_feature(fname):
        # fallback sensible ranges for common telecom features
        fallback = {
            'snr_ds_avg': (0.0, 50.0, 35.0),
            'snr_us_avg': (0.0, 50.0, 38.0),
            'p95_rtt': (0.0, 1000.0, 25.0),
            'jitter': (0.0, 100.0, 5.0),
            'fec31_uncor_sum': (0.0, 1e6, 100.0),
            'fec_ofdma_uncor_sum': (0.0, 1e6, 50.0),
            'download_speed_mbps': (0.0, 1000.0, 500.0)
        }
        # Prefer using training data ranges when available, but clamp minima to zero
        try:
            if results is not None and isinstance(results.get('X_train'), pd.DataFrame) and fname in results['X_train'].columns:
                col = results['X_train'][fname].dropna().astype(float)
                if not col.empty:
                    lo = float(col.min())
                    hi = float(col.max())
                    mid = float(col.median())
                    # Do not expand beyond observed training range to avoid extrapolation artifacts
                    lo = max(0.0, lo)
                    hi = max(hi, lo + 1e-6)
                    mid = min(max(mid, lo), hi)
                    return (lo, hi, mid)
        except Exception:
            pass

        # Fallback ranges: ensure non-negative minima and valid hi/default
        lo, hi, default = fallback.get(fname, (0.0, 100.0, 0.0))
        lo = max(0.0, float(lo))
        hi = max(float(hi), lo + 1e-6)
        default = float(default)
        default = min(max(default, lo), hi)
        return (lo, hi, default)

    user_vals = {}
    # Two-column layout: inputs (wider) on left, prediction result on right
    cols_main = st.columns([2, 1])
    left_col = cols_main[0]
    right_col = cols_main[1]
    sliders_state = {}
    with left_col:
        slider_cols = st.columns(2)
        for i, f in enumerate(input_feats):
            lo, hi, default = _get_range_for_feature(f)
            step = (hi - lo) / 200.0 if (hi - lo) > 0 else 0.1
            with slider_cols[i % 2]:
                # show friendly label (base name) but keep `f` as the key in user_vals
                label = display_name(f)
                try:
                    val = st.slider(label=label, min_value=float(lo), max_value=float(hi), value=float(default), step=float(step), format="%.3f")
                except Exception:
                    # fallback to number input if slider fails for some reason
                    val = st.number_input(label, value=float(default), format='%.3f')
                user_vals[f] = float(val)

    # Render base inputs if requested
    if show_base_inputs:
        base_feats = None
        sel80_path = os.path.join(LOGS, 'selected_pareto_80pct_vars.csv')
        if os.path.exists(sel80_path):
            try:
                base_feats = pd.read_csv(sel80_path, header=0).iloc[:,0].tolist()
            except Exception:
                base_feats = None

        if base_feats is None and pareto_top50 is not None:
            # prefer base features but filter out engineered lags/rolls and target-derived columns
            pf = pareto_top50.iloc[:,0].astype(str).tolist()
            def is_allowed_base(x):
                lx = x.lower()
                if 'call_flag' in lx or lx.startswith('target') or lx in ('call_flag', 'target', 'y'):
                    return False
                # filter engineered suffixes
                if any(s in lx for s in ['_lag', 'roll', 'rollmean', 'rollstd', '_p']):
                    return False
                return True
            base_feats = [f for f in pf if is_allowed_base(f)]

        if base_feats:
            with left_col.expander('Base (Pareto) feature sliders'):
                cols_b = st.columns(2)
                for j, bf in enumerate(base_feats):
                    if bf in user_vals:
                        continue  # already rendered as model input
                    lo, hi, default = _get_range_for_feature(bf)
                    step = (hi - lo) / 200.0 if (hi - lo) > 0 else 0.1
                    with cols_b[j % 2]:
                        label = display_name(bf)
                        try:
                            val = st.slider(label=label, min_value=float(lo), max_value=float(hi), value=float(default), step=float(step), format="%.3f")
                        except Exception:
                            val = st.number_input(label, value=float(default), format='%.3f')
                        user_vals[bf] = float(val)

    # use_model defined above

    # Compute prediction immediately (live)
    def compute_probability(user_vals_dict, use_model_pref=True):
        # Build input DataFrame
        X_in = pd.DataFrame([user_vals_dict])

        # Attempt model prediction when requested and available
        if use_model_pref and model_selected is not None and top_features is not None:
            missing = [c for c in top_features if c not in X_in.columns]
            if missing:
                # Try to synthesize engineered features from available base sliders
                def synthesize_features(required, base_vals):
                    out = {}
                    base_keys = list(base_vals.keys())
                    for feat in required:
                        if feat in base_vals:
                            out[feat] = base_vals[feat]
                            continue
                        # strip common engineered suffixes
                        m = re.match(r"(?P<b>.+?)_(lag|rollmean|rollstd|delta)\d+$", feat)
                        if m:
                            base = m.group('b')
                            if base in base_vals:
                                out[feat] = base_vals[base]
                                continue
                            candidates = [k for k in base_keys if k.startswith(base) or base.startswith(k)]
                            if candidates:
                                out[feat] = base_vals[candidates[0]]
                                continue
                            out[feat] = 0.0
                            continue
                        # pXX style (percentiles) or suffix like _p95
                        m2 = re.match(r"(?P<b>.+?)_p\d+$", feat)
                        if m2:
                            base = m2.group('b')
                            if base in base_vals:
                                out[feat] = base_vals[base]
                                continue
                            candidates = [k for k in base_keys if k.startswith(base) or base.startswith(k)]
                            if candidates:
                                out[feat] = base_vals[candidates[0]]
                                continue
                            out[feat] = 0.0
                            continue
                        # generic: try to find base token
                        for k in base_keys:
                            if k in feat or feat in k:
                                out[feat] = base_vals[k]
                                break
                        else:
                            out[feat] = 0.0
                    return out

                try:
                    synth = synthesize_features(missing, user_vals_dict)
                    # merge synthesized into X_in
                    for k, v in synth.items():
                        X_in[k] = v
                except Exception:
                    return None, f"missing_model_features:{missing[:5]}"

            # now pick up features in order
            X_proc = X_in.reindex(columns=top_features, fill_value=0.0)
            # apply scaler if available. Align input columns to the scaler's feature order
            try:
                if scaler is not None and hasattr(scaler, 'transform'):
                    # determine columns scaler was fitted on
                    scaler_cols = None
                    if hasattr(scaler, 'feature_names_in_'):
                        try:
                            scaler_cols = list(scaler.feature_names_in_)
                        except Exception:
                            scaler_cols = None
                    if scaler_cols is None and results is not None and isinstance(results.get('X_train'), pd.DataFrame):
                        try:
                            scaler_cols = list(results['X_train'].columns)
                        except Exception:
                            scaler_cols = None

                    if scaler_cols is not None:
                        # construct a full-width row aligned to scaler columns, fill missing with 0
                        X_full = X_proc.reindex(columns=scaler_cols, fill_value=0.0)
                        X_scaled = scaler.transform(X_full)
                        # build DataFrame with scaler column names, then select model's top_features
                        X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_cols)
                        X_proc = X_scaled_df.reindex(columns=top_features, fill_value=0.0)
                    else:
                        # fallback: scale current X_proc and keep top_features columns
                        X_proc = pd.DataFrame(scaler.transform(X_proc), columns=top_features)
            except Exception:
                pass

            try:
                proba = float(model_selected.predict_proba(X_proc)[:, 1][0])
                return proba, 'model'
            except Exception as e:
                return None, f'model_error:{e}'

        # Heuristic fallback when model not available or requested
        score = 0.0
        score += user_vals_dict.get('p95_rtt', 25) * 0.003
        score += user_vals_dict.get('jitter', 5) * 0.008
        score += user_vals_dict.get('fec31_uncor_sum', 100) * 0.0002
        score += user_vals_dict.get('fec_ofdma_uncor_sum', 50) * 0.0003
        score += (500 - user_vals_dict.get('download_speed_mbps', 500)) * 0.0001
        probability = 1 / (1 + np.exp(-score))
        probability = max(0.01, min(0.99, probability))
        return probability, 'heuristic'

    proba, source = compute_probability(user_vals, use_model_pref=use_model)
    # display result in right column
    if proba is None:
        with right_col:
            st.warning(f'Could not compute model prediction ({source}). Showing heuristic instead.')
        proba, _ = compute_probability(user_vals, use_model_pref=False)

    pct = float(proba) * 100.0
    with right_col:
        st.markdown('**Predicted call probability**')
        st.metric('Call probability', f"{pct:.2f}%", delta=None)
        try:
            # show a visual slider-like widget reflecting the probability (read-only)
            st.slider('Probability', min_value=0.0, max_value=100.0, value=float(pct), step=0.1, format="%.2f%%", disabled=True)
        except Exception:
            # fall back to progress bar
            st.progress(min(100, max(0, int(pct))))
        st.caption(f'Prediction source: {source}')

# Footer
st.markdown('---')
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1.2rem;">
        <p><strong>DOCSIS ML Platform</strong> • Artifacts in `logs/` • Use the analysis scripts to regenerate models and plots</p>
    </div>
""", unsafe_allow_html=True)
