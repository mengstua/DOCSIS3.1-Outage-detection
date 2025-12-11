# DOCSIS3.1-Outage-detection

This repository contains code and assets for DOCSIS customer call prediction and analysis.

Project layout (cleaned):

- `src/` - main importable Python package containing analysis and dashboard modules.
  - `src/analysis/` - core training and analysis routines.
  - `src/dashboard/` - Streamlit dashboard (the canonical dashboard module).
- `notebooks/` - Jupyter notebooks for exploration and training.
- `scripts/` - small entrypoint scripts and helpers (kept minimal).
- `data/` - input datasets (kept as-is, don't commit sensitive data if present).
- `logs/` - output artifacts (models, SHAP CSVs, PNGs).
- `archive/` - original legacy scripts and notebooks (preserved for reference).

Quick start

1. Create / activate your venv (Windows cmd/powershell):

```powershell
.venv\Scripts\Activate.ps1  # PowerShell
.venv\Scripts\activate      # cmd.exe
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit dashboard (development):

```powershell
python -m streamlit run src/dashboard/docsis_dashboard.py
```

Or use the launcher:

```powershell
python docsis_dashboard.py --run
```

Notes

- Heavy data-processing (SHAP generation, full retraining) is opt-in and can be long-running. Use `--no-shap` when running `scripts/model_compare_and_shap.py` for faster smoke tests.
- The `archive/` folder contains original scripts; use them only when you need the legacy exact code.
