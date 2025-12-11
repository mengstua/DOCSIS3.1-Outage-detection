# Dependency Manifest — DOCSIS3.1-Outage-detection

Generated: 2025-11-30

This manifest lists files that reference core project artifacts (dashboard, model runner, SHAP, logs, master CSV) and groups them into `core`, `related`, `archive-candidate`, and `archived` based on the quick repository scan.

## Scan criteria
- Keywords scanned: `docsis_dashboard`, `streamlit`, `model_compare_and_shap`, `generate_shap_plots`, `regenerate_shap_clean`, `ml_master_table_1M`, `feature_name_map`, `pareto_feature_importance_with_base`, `results_full_run.pkl`, `logs/`, `shap_`

## Core (do not remove)
- `docsis_dashboard.py` — Streamlit app; reads `logs/` artifacts, SHAP CSVs, and can invoke `scripts/generate_shap_plots.py`.
- `data/ml_master_table_1M.csv` — master dataset used by training scripts and notebooks.
- `scripts/model_compare_and_shap.py` — main training / compare script; reads master CSV and coordinates model runs.
- `scripts/generate_shap_plots.py` — produces SHAP CSVs and PNGs used by the dashboard.
- `scripts/regenerate_shap_clean.py` — aggregates/cleans SHAP outputs (`logs/shap_top10_clean.csv`, `logs/shap_top10.png`).
- `logs/results_full_run.pkl` (artifact produced by training notebooks / runner). See `trainig_ml_1.ipynb`.
- `logs/feature_name_map.csv` and `logs/pareto_feature_importance_with_base.csv` (artifacts referenced by the dashboard).

## Related (supporting tools; keep unless you want to archive)
- `scripts/pareto_chart.py` — plotting of pareto/importance results.
- `scripts/clean_selected_pareto.py`
- `scripts/inspect_shap_for_dashboard.py` — helper to turn SHAP CSVs into dashboard-friendly data.
- `scripts/inspect_results.py`, `scripts/inspect_models.py` — helpers to inspect `results_full_run.pkl` and models.
- `scripts/save_shap_top10.py` — saves top10 SHAP assets.
- `scripts/check_metrics_dashboard.py`, `scripts/smoke_check_dashboard.py`, `scripts/verify_catboost_metrics.py` — smoke checks and small validators.
- Notebooks that produce artifacts: `trainig_ml_1.ipynb`, `trainig_ml_1 copy.ipynb`, `Model_traning.ipynb`, `new_model.ipynb`, `MLM.ipynb`.
- `data/HFC_detection/model.py`, `data/HFC_detection/model_streamlit.py` — smaller model utilities / alternate streamlit helper.

## Archive candidates (no direct references discovered from core files in quick scan; review before archiving)
- `scripts/inspect_input_feats.py`
- `scripts/run_notebook_simple.py` (automation helper)
- `scripts/pareto_chart.py` (if you prefer to keep only generated artifacts)
- `scripts/inspect_*` helpers if they are only used interactively and not by CI

## Already archived
- `archive/notebooks/trainig_ml_1.ipynb`
- `archive/notebooks/trainig_ml_1 copy.ipynb`
- `archive/notebooks/results_view.ipynb`
- `archive/scripts/test_prediction_logic.py`

## Notes & next steps
1. This is an automated, keyword-based scan. It is conservative: files that are used interactively but not referenced by name in other files may appear in `archive-candidate` but still be important.
2. I can now:
   - produce a strict CSV manifest with absolute paths and matched keywords, or
   - proceed to copy selected `scripts/` files into `archive/scripts/` and replace originals with safe stubs (reversible), or
   - run a quick validation (no-SHAP) and start Streamlit to exercise the dashboard.

Reply with one of: `manifest-csv`, `archive-proposed`, `validate`, or `scan-deeper`.
