# AeroPulse: Air Quality Forecasting & Model Intelligence
### Empirical Machine Learning & Deep Sequence Modeling for Ambient Air Quality

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-228B22.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()

A scientifically disciplined, leak-free time-series forecasting system evaluating gradient boosting, deep sequence architectures (Bi-LSTM, Bi-GRU, Dilated TCN), and stacked ensembles under a strictly prior-only formulation:

```text
X(t-14, ..., t-1) -> AQI(t)
```

---

## 1. What does this do?

AeroPulse builds an end-to-end forecasting and environmental intelligence platform for the Delhi National Capital Region (NCR), one of the most severely polluted urban airsheds in the world. 

The system:
1. **Ingests and Cleans** 2,009 daily ground observations (2015–2020) from the Central Pollution Control Board (CPCB) India National Ambient Air Quality Network, unified with ECMWF ERA5 Atmospheric Reanalysis meteorology.
2. **Engineers 114 Prior-Only Features**: Calendar encodings, cyclic harmonics, lag observations (t-1, t-2, t-3, t-7), shifted rolling statistics, and 3D sliding sequence tensors (W=14) with zero contemporaneous lookahead.
3. **Trains and Evaluates 8 Architectures**: Climatological baseline, Standalone XGBoost, Extra Trees, PyTorch Bi-LSTM, PyTorch Bi-GRU, PyTorch Dilated TCN, and two Stacking Ensembles (SE-1 and SE-2) with Out-Of-Fold (OOF) cross-validation.
4. **Serves an Interactive Web Dashboard**: Built with FastAPI and Bootstrap 5, enabling users to select any historical date to inspect the preceding 14-day context, run real-time model inference, and compare predictions against ground truth.

---

## 2. How do I run it?

### 2.1 Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd AQI_Research_Project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2.2 Dataset Sourcing & Local Structure
The repository relies on historical observational records:
* **Pollutant Data Source:** Central Pollution Control Board (CPCB) India National Ambient Air Quality Monitoring Programme (NAAQM), published under official open data initiatives. Expected path: `data/raw/city_day.csv` (2.5 MB).
* **Meteorological Data Source:** European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 atmospheric reanalysis for Delhi NCR coordinates. Expected path: `data/raw/weather_daily.csv` (46 KB).
* **Processed Bundle:** The feature engineering pipeline parses and saves precomputed sequence and tabular matrices to `data/processed/dataset_bundle.pkl` (4.0 MB) to enable instantaneous dashboard inference and benchmark reproducibility. All data files are compact and tracked directly in the repository without requiring external cloud buckets or Git LFS.

### 2.3 Run Automated Verification Tests
Run the 11 unit tests verifying leakage prevention, rolling shift enforcement, scaler isolation, and API integrity:
```bash
pytest tests/ -v
```

### 2.4 Execute Full Experimental Benchmark
Trains all models, executes 5-fold TimeSeriesSplit stacking, multi-seed stability (3 seeds), statistical hypothesis tests, and generates publication figures:
```bash
PYTHONPATH=. python experiments/run_full_prior_experiments.py
```

### 2.5 Run the Exploratory Data Analysis Notebook
Open and run `notebooks/01_exploratory_data_analysis.ipynb` in Jupyter or VS Code:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 2.6 Launch Web Dashboard
Start the local FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` to interact with:
* **Forecast AQI:** Select any date in the evaluation period to forecast using preceding 14-day data.
* **Environmental Insights:** Inspect 30-day historical criteria pollutant curves and summary metrics.
* **Model Benchmark:** Review official performance metrics and latency numbers.

---

## 3. What did you find?

### 3.1 Headline Result: XGBoost is the Definitive Top Performer
XGBoost achieved the best overall point-forecasting performance on the held-out test period, while SE-2 was the strongest evaluated ensemble.

| Model Architecture | Framework | Test RMSE (lower is better) | Test MAE (lower is better) | Test MAPE (%) | Test R² (higher is better) | Multi-Seed Stability (3 Seeds) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Climatology** (±7 days) | NumPy | 100.45 | 83.17 | 61.09% | 0.2097 | N/A | 1.753 ms |
| **TCN (Standalone)** | PyTorch | 77.24 | 57.62 | 34.91% | 0.5328 | 79.12 ± 1.59 | 0.248 ms |
| **Bi-LSTM (Standalone)** | PyTorch | 59.15 | 45.72 | 27.77% | 0.7260 | 54.07 ± 4.17 | 0.076 ms |
| **SE-1 (XGB + Bi-LSTM + TCN)** | Ensemble | 45.85 | 37.97 | 27.28% | 0.8354 | 44.15 ± 1.96 | 0.268 ms |
| **Bi-GRU (Standalone)** | PyTorch | 42.67 | 30.73 | 16.93% | 0.8574 | 44.11 ± 2.00 | 0.051 ms |
| **SE-2 (XGB + Bi-GRU + TCN)** | Ensemble | 39.14 | 29.84 | 17.82% | 0.8800 | 40.55 ± 1.12 | 0.306 ms |
| **Extra Trees (Standalone)** | Sklearn | 36.37 | 26.90 | 16.09% | 0.8964 | N/A | 0.086 ms |
| **XGBoost (Standalone)** | **XGBoost** | **34.58** | **26.08** | **15.99%** | **0.9063** | **34.49 ± 0.16** | **0.016 ms** |

*Evaluated on the held-out Delhi test partition (402 held-out test days, May 27, 2019 to July 1, 2020).*

### 3.2 Key Scientific Insights
1. **Tree Models Dominate Tabular Time-Series**: Standalone XGBoost trained on 114 carefully engineered lag and rolling statistics outperformed all deep neural sequence models and multi-model stacking ensembles (Diebold-Mariano test p = 0.00054). It was also **16x faster** in inference latency (0.016 ms/sample).
2. **SE-2 Decisively Beats SE-1**: In a controlled comparison between recurrent cell architectures inside stacking, SE-2 (Bi-GRU) achieved significantly lower error than SE-1 (Bi-LSTM) (RMSE 39.14 vs. 45.85, paired t = -8.74, p < 1e-16) with 43% tighter multi-seed stability (SD 1.12 vs. 1.96). However, SE-2 did not surpass standalone XGBoost.
3. **Complementarity in Extremes**: While XGBoost had the lowest overall point error, deep sequence models provided orthogonal inductive bias during extreme pollution spikes (z > 3.0); stacking in SE-1 reduced peak spike underestimation from 70.5 to 40.8 AQI points.

---

## 4. What would you do next?

1. **Multi-Horizon Forecasting**: Extend from 1-day ahead (tau = 1) to multi-step recursive and direct horizons (tau = 1, 2, 3, 7 days).
2. **Multi-City Spatio-Temporal Graph Networks**: Expand the pipeline across all 26 CPCB cities using Spatio-Temporal Graph Neural Networks (ST-GNN) to model atmospheric advection and cross-city pollutant transport.
3. **Numerical Weather Prediction (NWP) Forcing**: Ingest forecasted meteorological variables from numerical models (e.g. GFS / ECMWF 24-48h weather forecasts) rather than relying exclusively on past weather lags.
4. **Extreme-Value Loss Optimization**: Implement asymmetric Pinball / Huber loss penalizing underestimation during severe winter pollution episodes.

---

## 5. Audit & Leakage Discovery

During an architectural audit of early research prototypes, a critical methodological vulnerability was identified and resolved:
* **The Issue:** Early academic implementations often compute rolling statistics or feature matrices that include unlagged observations from timestamp t, or fit scalers across the full dataset prior to temporal splitting. In AQI forecasting, same-day pollutant values contain near-perfect contemporaneous correlation with same-day AQI, creating massive artificial performance inflation.
* **The Resolution:** All unlagged raw variables from day t were strictly eliminated. All rolling statistics now enforce `shifted = df[col].shift(1)` so index t is strictly excluded from window calculations. Scalers and IQR outlier bounds are fit strictly on the training partition, and the stacking meta-learner is trained exclusively on out-of-fold temporal cross-validation folds.

---

## 6. Project Architecture & Directory Layout

```
AQI_Research_Project/
├── app/                      # FastAPI backend & Jinja2 templates
│   ├── main.py               # REST API endpoints (/api/predict, /api/available_dates, /api/insights)
│   └── templates/index.html  # Responsive Bootstrap 5 user dashboard
├── data/
│   ├── raw/                  # CPCB city_day.csv and ERA5 weather_daily.csv
│   ├── interim/              # Cleaned and merged daily time-series
│   └── processed/            # Scaled feature tensors and train/test splits
├── docs/                     # Methodological documentation & audit reports
├── experiments/              # Full experiment runner and saved model checkpoints
│   ├── run_full_prior_experiments.py
│   ├── xgboost/, baseline/, bilstm/, bigru/, tcn/, se1/, se2/
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb   # Comprehensive Exploratory Data Analysis notebook
├── results/
│   ├── figures/              # Core figures (01 to 06)
│   ├── metrics/              # Model comparison, ablation, and stability CSVs
│   ├── predictions/          # Held-out test predictions
│   └── reports/              # Statistical tests, spike analysis, DTW
├── src/
│   ├── data/                 # Ingestion and linear interpolation
│   ├── features/             # Calendar, lag, shifted rolling, and 3D sequence builders
│   ├── models/               # PyTorch (Bi-LSTM, Bi-GRU, TCN) & Tree models
│   ├── stacking/             # TimeSeriesSplit OOF stacking ensemble engine
│   └── utils/                # Configuration and logging utilities
├── tests/                    # Unit test suite (11/11 passing)
├── PROJECT_LOG.md            # Comprehensive engineering & research decision log
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 7. Technical Rigor & Scientific Validation

* **Complex Dataset:** 5.5 years of continuous multi-pollutant criteria monitoring and ERA5 meteorology (Delhi NCR).
* **Exploratory Analysis:** Documented in `notebooks/01_exploratory_data_analysis.ipynb` with 6 publication-ready figures.
* **Data Integrity:** Strict causal feature construction with training-only outlier thresholding.
* **Feature Engineering:** 114 lag, cyclic, and shifted rolling features + 14-day 3D sequence tensors.
* **Models Implemented (8 Total):** Historical Climatology, Standalone XGBoost, Extra Trees, PyTorch Bi-LSTM, PyTorch Bi-GRU, PyTorch Dilated TCN, SE-1, and SE-2.
* **Proper Cross-Validation:** 5-fold TimeSeriesSplit out-of-fold validation without lookahead.
* **Held-Out Evaluation:** 20% chronological temporal partition (402 held-out test days) with full statistical significance testing.
* **Reproducibility:** Multi-seed stability (mean ± SD across 3 seeds), deterministic initialization, and automated verification tests.
