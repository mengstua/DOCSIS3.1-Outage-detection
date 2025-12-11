# DOCSIS 3.1 Outage Detection

Machine learning platform for predicting customer calls based on DOCSIS 3.1 network telemetry and performance metrics.

## Overview

This project uses ML models (XGBoost, CatBoost, RandomForest, LinearSVC) to predict customer service calls by analyzing cable modem telemetry, network performance indicators, and quality metrics.

## Features

- **Interactive Dashboard**: Streamlit-based UI for model visualization and live predictions
- **Multiple ML Models**: Comparison of XGBoost, CatBoost, RandomForest, and LinearSVC classifiers
- **SHAP Analysis**: Feature importance and explainability visualizations
- **Pareto Analysis**: Aggregated base feature importance for dimensionality reduction
- **Live Predictions**: Slider-based input tool for real-time probability scoring

## Project Structure

```
DOCSIS3.1-Outage-detection/
├── docsis_dashboard.py          # Main Streamlit dashboard
├── data/                         # Training data (not versioned)
│   └── ml_master_table_1M.csv   # 1M rows, 55 features
├── logs/                         # Model artifacts and visualizations (not versioned)
│   ├── results_full_run.pkl     # Trained models, scaler, datasets
│   ├── models_metrics.csv       # Performance metrics
│   ├── models_roc.png           # ROC curves
│   ├── pareto_*.csv/png         # Feature importance charts
│   └── shap_*.csv/png           # SHAP visualizations
├── scripts/                      # Analysis and helper scripts
│   └── model_compare_and_shap.py
├── archive/                      # Archived notebooks and scripts
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### Prerequisites
- Python 3.8+
- 2GB+ RAM (for model training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mengstua/DOCSIS3.1-Outage-detection.git
cd DOCSIS3.1-Outage-detection
```

2. Create virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare data:
   - Place your training CSV in `data/ml_master_table_1M.csv`
   - CSV should have 55 columns including `call_flag` target

## Usage

### Launch Dashboard

```bash
streamlit run docsis_dashboard.py
```

Open browser to `http://localhost:8501`

### Train Models

```bash
# Quick validation (no SHAP)
python archive/scripts/model_compare_and_shap.py --no-shap

# Full pipeline with SHAP
python archive/scripts/model_compare_and_shap.py
```

Artifacts saved to `logs/` directory.

## Dashboard Features

### 1. Overview Tab
- Best model selection (auto-detected by ROC AUC)
- Key metrics: ROC AUC, Precision, F1 Score
- ROC curve visualization

### 2. Model Comparison Tab
- Metrics table for all models
- SHAP top-20 features per model
- Interactive model selector

### 3. Feature Analysis Tab
- Pareto charts (base feature aggregation)
- Top 80% cumulative importance features
- SHAP summary plots (beeswarm, bar, dependence, waterfall)

### 4. Prediction Tool Tab
- Live probability scoring
- Slider inputs for top features
- Model vs heuristic prediction modes

## Data Schema

**Input Features** (55 columns):
- Network metrics: SNR, Rx power, FEC errors, uncorrectables
- Modem events: offline counts, keepalive misses, deregistrations
- Latency: RTT p50/p95, jitter
- Profile quality: time-in-profile, ratios, quality scores
- Node/CMTS load: downstream/upstream utilization
- Speedtest: download/upload Mbps

**Target**: `call_flag` (binary: 0 = no call, 1 = customer call within prediction window)

## Model Performance

| Model | ROC AUC | Precision | F1 Score |
|-------|---------|-----------|----------|
| XGBoost | 0.XXX | 0.XXX | 0.XXX |
| CatBoost | 0.XXX | 0.XXX | 0.XXX |
| RandomForest | 0.XXX | 0.XXX | 0.XXX |
| LinearSVC | 0.XXX | 0.XXX | 0.XXX |

*(Run training pipeline to populate metrics)*

## Key Dependencies

- `streamlit` - Dashboard UI
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Preprocessing, baseline models
- `xgboost` - Gradient boosting
- `catboost` - Gradient boosting with categorical support
- `shap` - Model explainability
- `plotly` - Interactive visualizations
- `matplotlib`, `seaborn` - Static plots

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License.
