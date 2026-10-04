# SHAP Interpretability Report: AQI Multi-City Research

**Experiment:** Multi-City Heterogeneous Stacking AQI Forecasting  
**Dataset:** 6 Major Indian Metropolitan Regions (Delhi, Gurugram, Bengaluru, Chennai, Hyderabad, Patna)  
**Evaluation Set:** Held-out Temporal Test Split ($N = 1,770$, 2019-09-08 to 2020-07-01)  
**Date:** September 19, 2026  

---

## 1. Overview & Methodological Framing

This report provides model-based explainability for:
1. **Primary Tabular Predictor (Standalone XGBoost):** Feature-level SHAP attributions identifying which lagged, rolling, and atmospheric variables most strongly influenced point forecasts across all test instances.
2. **Heterogeneous Meta-Learners (SE-1 & SE-2 Extra Trees):** Ensemble-level SHAP attributions quantifying how predictions from tree-based (XGBoost), convolutional (TCN), and recurrent (BiLSTM / BiGRU) base models were weighted during stacking.

> [!IMPORTANT]
> **Causal Disclaimer:** SHAP values measure conditional feature attribution (predictive influence) within the mathematical model. They do **not** establish physical causality or direct atmospheric mechanism. High SHAP attribution signifies predictive reliance by the estimator, not causation.

---

## 2. Experimental Data & Model Parameters

- **Test Observations ($N$):** 1,770 daily observations across 6 cities.
- **Tabular Features ($D$):** 65 engineered prior-only features ($X_{<t}$).
- **Base Models Analyzed:**
  - **XGBoost:** 200 estimators, max depth 5, learning rate 0.04.
  - **SE-1 Extra Trees Meta-Learner:** 100 trees, max depth 5, inputs: $[\hat{y}_{\text{XGB}}, \hat{y}_{\text{BiLSTM}}, \hat{y}_{\text{TCN}}]$.
  - **SE-2 Extra Trees Meta-Learner:** 100 trees, max depth 5, inputs: $[\hat{y}_{\text{XGB}}, \hat{y}_{\text{BiGRU}}, \hat{y}_{\text{TCN}}]$.
- **SHAP Method:** Exact TreeSHAP (Lundberg et al., 2020) with polynomial complexity for decision tree ensembles, ensuring exact local additivity:
  $$\sum_{j=1}^{D} \phi_j(x) + \phi_0 = f(x)$$

---

## 3. XGBoost Feature-Level Importance

### 3.1 Top 15 Most Influential Features (Test Set $N=1,770$)

| Rank | Feature Name | Mean $|\text{SHAP}|$ (AQI pts) | Feature Category | Description |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `PM2.5_lag_1` | **54.50** | Autoregressive Lag | Prior-day fine particulate matter ($t-1$) |
| **2** | `AQI_lag_1` | **20.32** | Autoregressive Lag | Immediate prior-day Air Quality Index ($t-1$) |
| **3** | `PM10_lag_1` | **5.38** | Autoregressive Lag | Prior-day coarse particulate matter ($t-1$) |
| **4** | `AQI_ewma_7` | **4.82** | Temporal EWMA | 7-day Exponentially Weighted Moving Average of AQI |
| **5** | `AQI_roll_mean_14` | **4.80** | Rolling Statistic | 14-day shifted rolling mean AQI |
| **6** | `Wind_Speed_lag_1` | **2.50** | Meteorology | Prior-day mean wind speed ($t-1$) |
| **7** | `PM2.5_lag_2` | **2.43** | Autoregressive Lag | 2-day lagged fine particulate matter ($t-2$) |
| **8** | `Humidity_lag_1` | **2.33** | Meteorology | Prior-day relative humidity ($t-1$) |
| **9** | `city_Delhi` | **2.32** | Spatial Identity | One-hot binary indicator for Delhi NCR |
| **10** | `AQI_roll_mean_7` | **1.91** | Rolling Statistic | 7-day shifted rolling mean AQI |
| **11** | `CO_lag_1` | **1.86** | Autoregressive Lag | Prior-day Carbon Monoxide ($t-1$) |
| **12** | `AQI_roll_min_7` | **1.56** | Rolling Statistic | 7-day shifted rolling minimum AQI |
| **13** | `O3_lag_1` | **1.32** | Autoregressive Lag | Prior-day Ground-Level Ozone ($t-1$) |
| **14** | `WindSpeed_roll_mean_7` | **1.30** | Meteorology | 7-day shifted rolling mean wind speed |
| **15** | `Humidity_roll_mean_7` | **1.11** | Meteorology | 7-day shifted rolling mean relative humidity |

### 3.2 Environmental & Methodological Interpretations
1. **Particulate & Autoregressive Dominance:** `PM2.5_lag_1` (Mean $|\text{SHAP}| = 54.50$) and `AQI_lag_1` (Mean $|\text{SHAP}| = 20.32$) account for the overwhelming majority of predictive variance. Because Indian National AQI sub-indices are predominantly determined by 24-hour fine particulate matter concentrations, the gradient-boosted trees accurately discover this physical relationship.
2. **Temporal Trend Smoothing:** Multi-day exponential and rolling windows (`AQI_ewma_7`, `AQI_roll_mean_14`) act as baseline trend anchors, stabilizing forecasts against single-day sensor anomalies.
3. **Atmospheric Ventilation & Dispersion:** Meteorology variables (`Wind_Speed_lag_1`, `Humidity_lag_1`, `WindSpeed_roll_mean_7`) serve as non-linear gating features that adjust expected accumulation during calm, high-humidity winter stagnation versus turbulent dispersion periods.

---

## 4. Meta-Learner Stack Explainability

### 4.1 Hybrid 1 (SE-1: XGBoost + BiLSTM + TCN $\to$ Extra Trees)

| Base Learner Component | Mean $|\text{SHAP}|$ Value (AQI pts) | SHAP Relative Share (%) | Tree Gini Importance (%) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | **26.50** | **73.9%** | **74.7%** |
| **TCN** | **6.79** | **18.9%** | **19.9%** |
| **Bi-LSTM** | **2.56** | **7.1%** | **5.4%** |

### 4.2 Hybrid 2 (SE-2: XGBoost + BiGRU + TCN $\to$ Extra Trees)

| Base Learner Component | Mean $|\text{SHAP}|$ Value (AQI pts) | SHAP Relative Share (%) | Tree Gini Importance (%) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | **27.24** | **70.4%** | **75.9%** |
| **TCN** | **7.71** | **19.9%** | **17.2%** |
| **Bi-GRU** | **3.72** | **9.6%** | **6.9%** |

---

## 5. Synthesis of Meta-Learner Behavior

1. **Heavy Reliance on Tree Predictors:** The Extra Trees meta-learner allocates $>70\%$ of its predictive attribution to XGBoost in both architectures.
2. **Convolutional vs. Recurrent Utility:** TCN consistently provides the secondary signal ($18.9\%\text{--}19.9\%$ attribution), whereas recurrent architectures (BiLSTM / BiGRU) contribute $<10\%$.
3. **Instance-Level Heatmap Findings:** Heatmaps reveal that during standard non-extreme days, XGBoost governs nearly $100\%$ of the meta-learner output, while TCN activations increase during sudden transition days.

---

## 6. Generated Artifacts & File Manifest

| Artifact Type | File Location | Description |
| :--- | :--- | :--- |
| **XGBoost Summary Beeswarm** | `results/research/shap/shap_xgboost_summary.png` | Top 15 features beeswarm distribution on test set |
| **XGBoost Bar Chart** | `results/research/shap/shap_xgboost_bar.png` | Global mean absolute SHAP bar chart |
| **SE-1 Meta Attribution** | `results/research/shap/shap_se1_meta.png` | Beeswarm + bar plot for SE-1 meta-learner |
| **SE-2 Meta Attribution** | `results/research/shap/shap_se2_meta.png` | Beeswarm + bar plot for SE-2 meta-learner |
| **SE-1 SHAP Heatmap** | `results/research/shap/shap_se1_heatmap.png` | Instance-level SHAP heatmap across 1,770 test samples |
| **SE-2 SHAP Heatmap** | `results/research/shap/shap_se2_heatmap.png` | Instance-level SHAP heatmap across 1,770 test samples |
| **XGBoost Importance CSV** | `results/research/shap/shap_xgboost_importance.csv` | Full ranked table of all 65 features by mean $|\text{SHAP}|$ |
| **SE-1 Meta Importance CSV**| `results/research/shap/shap_se1_meta_importance.csv` | Numerical SHAP share vs. Gini importance table |
| **SE-2 Meta Importance CSV**| `results/research/shap/shap_se2_meta_importance.csv` | Numerical SHAP share vs. Gini importance table |
| **Full Report Markdown** | `results/research/shap/SHAP_REPORT.md` | Complete interpretability documentation |

---
