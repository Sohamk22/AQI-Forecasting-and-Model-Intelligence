# AQI Forecasting, Model Intelligence & Statutory Policy Assistant

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-228B22.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-29%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

> **"Understand your air. Make better decisions."**  
> An end-to-end ambient air quality forecasting and environmental intelligence platform that combines **strictly leak-free machine learning sequence models** with an **offline-capable statutory policy RAG assistant** for citizens and environmental policymakers.

---

## Quick Navigation
1. [Project Overview & Core Features](#1-project-overview)
2. [Step-by-Step How to Run (100% Reproducible)](#2-step-by-step-how-to-run-reproducibility-guide)
3. [Contemporaneous Target Leakage Discovery & Methodological Rigor](#3-contemporaneous-target-leakage-discovery--methodological-rigor)
4. [Timeline Honesty & Project Evolution](#4-timeline-honesty--project-evolution)
5. [System Architecture & Workflow](#5-system-architecture--workflow)
6. [Machine Learning Forecasting & Benchmark Evidence](#6-machine-learning-forecasting--benchmark-evidence)
7. [Policy RAG & Public Assistant Layer](#7-policy-rag--public-assistant-layer)
8. [Automated Verification & Test Suite](#8-automated-verification--test-suite)
9. [Repository Directory Structure](#9-repository-directory-structure)
10. [API Reference](#10-api-reference)
11. [What I Would Do Differently & Future Improvements](#11-what-i-would-do-differently--future-improvements)
12. [Troubleshooting & FAQ](#12-troubleshooting--faq)

---

## 1. Project Overview

Predicting urban air quality in severely polluted airsheds (such as Delhi NCR and the Indo-Gangetic Plain) is a vital challenge for public health, city governance, and environmental law enforcement.

This platform bridges empirical time-series forecasting with actionable statutory policy intelligence:
1. **Empirical Multi-Model ML Forecasting**: Implements 7 model architectures + Climatology baseline under a strictly prior-only formulation ($X(t-14, \dots, t-1) \to \text{AQI}(t)$), evaluating gradient boosting, deep sequence architectures (Bi-LSTM, Bi-GRU, Dilated TCN), and heterogeneous stacking ensembles.
2. **Policy Intelligence RAG Engine**: A hybrid sparse/dense (BM25 + FAISS) retrieval engine grounded in official Indian environmental statutes (Graded Response Action Plan [GRAP 2024], National Clean Air Programme [NCAP], Air Act 1981, CAQM Act 2021, and CPCB NAAQS standards).
3. **Dual-Mode Interactive Web Application**:
   - **Public AI Assistant**: Conversational citizen interface for daily health advisories, outdoor activity recommendations, and statutory awareness.
   - **Government & Analytics Intelligence**: Real-time forecast evaluation, model ablation comparison, criteria pollutant trends, and statutory intervention simulation.

---

## 2. Step-by-Step How to Run (Reproducibility Guide)

Follow these exact steps from a clean terminal to clone, configure, run, and test the project without guesswork.

### Step 2.1 — Prerequisites
- **Operating System**: macOS, Linux, or Windows (WSL recommended).
- **Python Version**: Python 3.12.x (verified on Python 3.12.14).
- **Git**: Installed and configured.

### Step 2.2 — Clone the Repository
```bash
git clone https://github.com/Sohamk22/AQI-Forecasting-and-Model-Intelligence.git
cd AQI-Forecasting-and-Model-Intelligence
```

### Step 2.3 — Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS / Linux:
source venv/bin/activate

# (Or activate on Windows PowerShell: .\venv\Scripts\Activate.ps1)
```

### Step 2.4 — Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Expected install time: 1–2 minutes. All dependencies (PyTorch CPU, XGBoost, Scikit-learn, FastAPI, Sentence-Transformers, FAISS-CPU) install cleanly without binary compilation errors.*

### Step 2.5 — Verify Precomputed Data & Checkpoints
All required datasets and precomputed feature bundles are tracked directly in the repository:
- `data/raw/city_day.csv` (2.5 MB) — CPCB official daily criteria pollutant observations.
- `data/raw/weather_daily.csv` (46 KB) — ECMWF ERA5 reanalysis meteorology.
- `data/processed/dataset_bundle.pkl` (4.0 MB) — Preprocessed Delhi sequence and tabular feature matrices.
- `data/policy_documents/` — Authoritative statutory PDFs.
- `data/cache/rag_index.pkl` (1.7 MB) — Precomputed hybrid BM25 + FAISS search index for instant offline startup.

No external downloads or cloud credentials are required to run the application or tests.

### Step 2.6 — Run the Automated Test Suite
Run the full automated test suite (29 tests) verifying temporal leakage isolation, stacking cross-validation, REST API contracts, RAG retrieval, citation fidelity, and safety guardrails:
```bash
pytest tests/ -v
```
**Expected Output:**
```text
======================== 29 passed, 7 warnings in ~9s ========================
```

### Step 2.7 — Launch the Web Application
Start the FastAPI server using Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2.8 — Access the Application
Open your web browser and navigate to:
```text
http://localhost:8000
```

- **Testing the Public Assistant**:
  Type a question into the search bar, for example:
  - *"Is it safe for a morning jog today?"*
  - *"What GRAP Stage 3 actions are in effect?"*
  - *"Why is Delhi's AQI spiking this week?"*
- **Testing Government Intelligence**:
  Click the **"Government Intelligence"** toggle at the top of the interface:
  - Select any historical test date (e.g., `2020-01-15`).
  - View real-time model forecasts (XGBoost, SE-1, SE-2, Extra Trees) vs. actual observed AQI.
  - Review 14-day prior criteria pollutant trends and statutory policy advisories.

### Step 2.9 — (Optional) Re-run ML Benchmark Experiments
To retrain and reproduce the Delhi NCR experimental benchmark from scratch:
```bash
python experiments/run_full_prior_experiments.py
```
To run the multi-city research evaluation across 6 cities:
```bash
python experiments/research/run_training.py
```

---

## 3. Contemporaneous Target Leakage Discovery & Methodological Rigor

### 3.1 The Original Problem: Contemporaneous Feature Leakage
In early air quality forecasting research, pipelines often construct features from the same day ($t$) as the target AQI prediction:
$$\text{Input: } [PM_{2.5}(t), PM_{10}(t), NO_2(t), \dots] \longrightarrow \text{Target: } AQI(t)$$

**Why this was flawed:**
Because the Indian National Air Quality Index is mathematically derived from sub-indices of same-day criteria pollutants (predominantly $PM_{2.5}$ and $PM_{10}$), providing same-day pollutant values produces near-perfect correlation with the target ($R^2 > 0.99$). This transforms the problem from **genuine temporal forecasting** into **trivial concurrent estimation**.

Furthermore, computing unshifted rolling statistics (`rolling(7).mean()`) without shifting by 1 step includes observation $t$ in the calculation, silently leaking contemporaneous ground truth into test samples.

### 3.2 Identification & Resolution
During a methodological audit, this vulnerability was identified and eliminated:
1. **Strictly Prior-Only Formulation**: All contemporaneous inputs from day $t$ were eliminated. Predictions strictly rely on historical windows:
   $$X(t-14, \dots, t-1) \longrightarrow AQI(t)$$
2. **Shifted Rolling Windows**: All rolling statistical features explicitly enforce `shifted = df[col].shift(1)` before computing window aggregates, guaranteeing that index $t$ is strictly excluded.
3. **Split-Isolated Scalers**: RobustScalers and IQR clipping boundaries are fitted **strictly on training partitions** and subsequently applied to validation/test partitions without lookahead.
4. **Out-of-Fold (OOF) Stacking**: Meta-learners are trained exclusively on genuine out-of-fold predictions generated via `TimeSeriesSplit(n_splits=5)`, preventing meta-learner target memorization.

### 3.3 Automated Verification
These methodological guarantees are enforced via automated tests in `tests/test_leakage.py` and `tests/test_stacking.py`.

---

## 4. Timeline Honesty & Project Evolution

In accordance with academic integrity and coursework grading criteria, the development history is transparently delineated into two distinct phases:

### Phase 1 — Original Coursework Submission (September 2026)
- Implemented the core leak-free ML forecasting engine.
- Implemented the 7-model benchmark comparison against a historical Climatology baseline.
- Formulated the strictly prior-only feature pipeline ($X(t-14, \dots, t-1) \to AQI(t)$).
- Built the initial FastAPI backend, unit test suite (11 unit tests), and baseline dashboard.

### Phase 2 — Post-Submission Extended Development (October 2026)
- **Policy RAG Layer**: Integrated statutory document parsing, semantic chunking, and hybrid sparse/dense (BM25 + FAISS) retrieval over official Indian environmental statutes (`src/rag/`).
- **Public AI Assistant**: Implemented a 9-class intent router with conversational health advisories, citation enforcement, and prompt-injection defense (`src/rag/assistant.py`).
- **Dual-Mode User Interface**: Upgraded the UI into a citizen-facing Public Assistant with an advanced Government & Research portal.
- **Multi-City Research & SHAP Interpretability**: Expanded evaluation across 6 Indian cities (Delhi, Gurugram, Bengaluru, Chennai, Hyderabad, Patna) with SHAP TreeExplainer feature importance.
- **Expanded Test Suite**: Extended automated test coverage from 11 to **29 unit and integration tests** (100% passing).

---

## 5. System Architecture & Workflow

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AEROPULSE PLATFORM                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────────────┐          ┌───────────────────────────────┐   │
│   │    Public AI Assistant    │          │  Government Analytics Portal  │   │
│   │ (Conversational Citizens) │          │    (Policymakers / Experts)   │   │
│   └─────────────┬─────────────┘          └───────────────┬───────────────┘   │
│                 │                                        │                   │
│                 └──────────────────┬─────────────────────┘                   │
│                                    ▼                                         │
│                      FastAPI Application Gateway                             │
│                  (/api/chat, /api/predict, /api/policy)                      │
│                                    │                                         │
│         ┌──────────────────────────┴──────────────────────────┐              │
│         ▼                                                     ▼              │
│   ┌──────────────────────────────┐              ┌──────────────────────────┐ │
│   │    ML Forecasting Engine     │              │   Policy RAG Engine      │ │
│   ├──────────────────────────────┤              ├──────────────────────────┤ │
│   │ • 114 Prior-Only Features    │              │ • Chunked Statutory Docs │ │
│   │ • 14-Day 3D Sequence Tensors │              │ • BM25 + FAISS Dense     │ │
│   │ • XGBoost / Extra Trees      │              │ • Reciprocal Rank Fusion │ │
│   │ • PyTorch Bi-LSTM / Bi-GRU   │              │ • Citation Verifier      │ │
│   │ • Dilated Temporal Conv(TCN) │              │ • Prompt Guardrails      │ │
│   │ • Stacking Meta-Learners     │              │ • 9-Class Intent Router  │ │
│   └──────────────┬───────────────┘              └────────────┬─────────────┘ │
│                  │                                           │               │
│                  └─────────────────────┬─────────────────────┘               │
│                                        ▼                                     │
│                       Multi-Tier LLM Provider Router                         │
│             [ NVIDIA NIM  →  Gemini  →  OpenAI  →  Deterministic Offline ]   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Machine Learning Forecasting & Benchmark Evidence

### 6.1 Seven-Model Comparison & Climatology Baseline (Delhi Held-Out Test Set)
Evaluated on 402 held-out test days (May 27, 2019 to July 1, 2020) under chronological splitting:

| Model Architecture | Framework / Type | Test RMSE ↓ | Test MAE ↓ | Test MAPE (%) | Test R² ↑ | Multi-Seed Stability (3 Seeds) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Climatology** (±7 days) | Baseline (NumPy) | 100.45 | 83.17 | 61.09% | 0.2097 | N/A | 1.753 ms |
| **Dilated TCN (Standalone)** | PyTorch (Deep Seq) | 77.24 | 57.62 | 34.91% | 0.5328 | 79.12 ± 1.59 | 0.248 ms |
| **Bi-LSTM (Standalone)** | PyTorch (Deep Seq) | 59.15 | 45.72 | 27.77% | 0.7260 | 54.07 ± 4.17 | 0.076 ms |
| **SE-1 (XGB + Bi-LSTM + TCN)** | Stacking Ensemble | 45.85 | 37.97 | 27.28% | 0.8354 | 44.15 ± 1.96 | 0.268 ms |
| **Bi-GRU (Standalone)** | PyTorch (Deep Seq) | 42.67 | 30.73 | 16.93% | 0.8574 | 44.11 ± 2.00 | 0.051 ms |
| **SE-2 (XGB + Bi-GRU + TCN)** | Stacking Ensemble | 39.14 | 29.84 | 17.82% | 0.8800 | 40.55 ± 1.12 | 0.306 ms |
| **Extra Trees (Standalone)** | Sklearn (Tree) | 36.37 | 26.90 | 16.09% | 0.8964 | N/A | 0.086 ms |
| **XGBoost (Standalone)** | XGBoost (Gradient Boost) | **34.58** | **26.08** | **15.99%** | **0.9063** | **34.49 ± 0.16** | **0.016 ms** |

### 6.2 Key Scientific Insights
1. **Tree Models Outperform Neural Models on Tabular Lags**: Standalone XGBoost trained on 114 engineered prior-only lag and rolling statistics achieved the lowest error, outperforming deep sequence models (Diebold-Mariano test $p = 0.00054$).
2. **SE-2 Beats SE-1 in Recurrent Stacking**: SE-2 (fusing Bi-GRU representations) significantly outperformed SE-1 (Bi-LSTM) ($p < 10^{-16}$) with 43% tighter multi-seed stability.
3. **Severe Pollution Spike Complementarity**: While XGBoost achieved the best mean point metrics, deep sequence models provided orthogonal inductive bias during extreme winter pollution spikes ($Z > 3.0$), reducing peak spike underestimation from 70.5 to 40.8 AQI points.

---

## 7. Policy RAG & Public Assistant Layer

### 7.1 Authoritative Statutory Knowledge Base
The RAG system indexes official statutory legal documents located in `data/policy_documents/`:
- **GRAP 2024 (Graded Response Action Plan)**: Mandated emergency actions across Stage I (Poor, AQI 201–300), Stage II (Very Poor, AQI 301–400), Stage III (Severe, AQI 401–450), and Stage IV (Severe+, AQI > 450).
- **NCAP (National Clean Air Programme)**: City-level reduction targets and monitoring protocols.
- **Air Act 1981**: Statutory authority of State and Central Pollution Control Boards under Sections 19, 21, and 31A.
- **CAQM Act 2021**: Commission for Air Quality Management in NCR & Adjoining Areas powers and jurisdiction.
- **CPCB NAAQS Standards**: Permissible thresholds for criteria pollutants ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$).

### 7.2 Hybrid Retrieval & Intent Routing
- **BM25 + FAISS Dense Indexing**: Documents are split into semantic chunks with statutory metadata (Act, Section, Severity, Pollutant).
- **Reciprocal Rank Fusion (RRF)**: Combines exact statutory keyword matching with dense embedding cosine similarity (`all-MiniLM-L6-v2`).
- **9-Class Intent Classification**: Automatically classifies user queries into *Outdoor Activity Advisory, Sensitive Groups Health, Government Action & GRAP, Pollution Causes, Legal/Regulatory Authority, General AQI Explanation, Forecast Inquiry, Evidence Request, or Out-of-Domain Refusal*.
- **Multi-Tier LLM Architecture**:
  1. NVIDIA NIM (`openai/gpt-oss-20b`)
  2. Google Gemini (`gemini-1.5-flash`)
  3. OpenAI (`gpt-4o-mini`)
  4. **Deterministic Offline Grounded Engine** (Default fallback: synthesizes fully cited advisories with zero external API dependencies).

---

## 8. Automated Verification & Test Suite

The test suite contains **29 unit and integration tests** organized by subsystem:

```bash
pytest tests/ -v
```

| Test Module | Tests | Description |
| :--- | :---: | :--- |
| `tests/test_leakage.py` | 4 | Verifies no future lookahead, 1-step rolling shifts, scaler isolation, and metric formulas. |
| `tests/test_stacking.py` | 1 | Validates 5-fold TimeSeriesSplit out-of-fold cross-validation execution. |
| `tests/test_api.py` | 6 | Tests `/api/predict`, `/api/insights`, `/api/available_dates`, `/api/metrics`, and backward compatibility. |
| `tests/test_rag.py` | 10 | Verifies chunking, hybrid retrieval, citation verification, prompt injection defense, and policy endpoints. |
| `tests/test_assistant.py` | 8 | Tests intent routing across 9 categories, answer formatting, LLM provider diagnostics, and domain refusal. |

---

## 9. Repository Directory Structure

```text
AQI-Forecasting-and-Model-Intelligence/
├── app/
│   ├── main.py                     # FastAPI backend & REST endpoints
│   ├── static/                     # CSS & JS assets
│   └── templates/
│       └── index.html              # Unified Public Assistant & Government UI
├── data/
│   ├── cache/                      # Precomputed RAG search index (rag_index.pkl)
│   ├── interim/                    # Merged daily time-series & weather records
│   ├── policy_documents/           # Statutory Indian environmental legal PDFs
│   ├── processed/                  # Precomputed feature bundles (dataset_bundle.pkl, research_bundle.pkl)
│   └── raw/                        # CPCB raw criteria pollutants (city_day.csv, weather_daily.csv)
├── experiments/
│   ├── baseline/, bigru/, bilstm/, se1/, se2/, tcn/, xgboost/ # Model checkpoints
│   ├── research/                   # Multi-city benchmark & SHAP scripts
│   └── run_full_prior_experiments.py # Delhi NCR full experiment runner
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Complete EDA notebook
├── results/
│   ├── figures/                    # Publication charts (01 to 06)
│   ├── metrics/                    # Frozen benchmark metrics and JSON tables
│   ├── predictions/                # Out-of-sample prediction artifacts
│   ├── reports/                    # Statistical tests & DTW analysis reports
│   └── research/                   # Multi-city metrics, figures & SHAP reports
├── src/
│   ├── analysis/                   # DTW alignment & spike detection
│   ├── data/                       # Ingestion and interpolation
│   ├── evaluation/                 # Metrics & statistical tests
│   ├── explainability/             # SHAP game-theoretic explainability
│   ├── features/                   # Prior-only feature extraction & sequence builders
│   ├── models/                     # PyTorch deep sequence models & XGBoost
│   ├── rag/                        # Complete Policy RAG & Conversational Assistant
│   │   ├── assistant.py            # Public conversational assistant & intent router
│   │   ├── chunker.py              # Semantic markdown & PDF chunking
│   │   ├── citations.py            # Statutory citation validation
│   │   ├── evaluator.py            # Automated RAG benchmarking
│   │   ├── generator.py            # Multi-provider LLM synthesis
│   │   ├── loader.py               # Statutory document loader
│   │   ├── pipeline.py             # End-to-end RAG orchestrator
│   │   ├── policy_context.py       # Regulatory context generator
│   │   ├── retriever.py            # Hybrid BM25 + FAISS retriever
│   │   ├── safety.py               # Prompt injection defense & refusal
│   │   └── schemas.py              # Pydantic data models
│   ├── stacking/                   # TimeSeriesSplit OOF stacking meta-learner
│   └── utils/                      # Logging & configuration
├── tests/                          # 29 automated test cases
├── PROJECT_LOG.md                  # Engineering decision log
├── pytest.ini                      # Pytest runner configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive platform documentation
```

---

## 10. API Reference

### 1. Forecast Endpoint (`POST /api/predict`)
```json
// Request
{ "date": "2020-01-15" }

// Response
{
  "target_date": "2020-01-15",
  "actual_aqi": 272.0,
  "forecasts": {
    "XGBoost": { "aqi": 268.4, "category": "Poor", "color": "#fd7e14" },
    "SE1": { "aqi": 262.1, "category": "Poor", "color": "#fd7e14" },
    "SE2": { "aqi": 258.9, "category": "Poor", "color": "#fd7e14" },
    "ExtraTrees": { "aqi": 265.0, "category": "Poor", "color": "#fd7e14" }
  },
  "consensus_aqi": 263.6,
  "prior_14_days_summary": { "PM2.5": 142.3, "PM10": 218.1, "NO2": 52.4 },
  "methodology_note": "Forecast uses exclusively information from [t-14, ..., t-1]. Zero contemporaneous lookahead."
}
```

### 2. Public Assistant Endpoint (`POST /api/chat` or `POST /api/assistant/query`)
```json
// Request
{
  "query": "Is it safe for morning outdoor exercise today?",
  "date": "2020-01-15"
}

// Response
{
  "query": "Is it safe for morning outdoor exercise today?",
  "intent": "outdoor_activity",
  "synthesized_response": "...",
  "risk_level": "High",
  "primary_pollutant": "PM2.5",
  "citations": [
    { "document": "GRAP_2024", "section": "Stage II Actions", "relevance": 0.88 }
  ],
  "llm_provider_used": "deterministic_offline"
}
```

---

## 11. What I Would Do Differently & Future Improvements

1. **Multi-Horizon Direct Forecasting**: The current operational engine forecasts 1-day ahead ($\tau = 1$). Expanding to multi-step recursive and direct horizons ($\tau \in \{1, 3, 7\}$ days) would provide advance warning for multi-day emergency planning.
2. **Spatial Atmospheric Advection Modeling**: Integrating Spatio-Temporal Graph Neural Networks (ST-GNN) to explicitly model cross-city atmospheric transport and wind advection vectors across Northern India.
3. **Numerical Weather Prediction (NWP) Forcing**: Ingesting real-time forecast weather grids (e.g., GFS/ECMWF 48h meteorological predictions) rather than relying exclusively on past weather lags.
4. **Bioreactor / Active Air Purification Simulation**: Evaluating localized mitigation interventions (such as microalgae photobioreactors for particulate and carbon capture) within high-density urban canyons.

---

## 12. Troubleshooting & FAQ

- **Port 8000 already in use?**  
  Run on a different port: `uvicorn app.main:app --port 8080 --reload`.
- **Deprecation Warnings during Pytest?**  
  Minor `StarletteDeprecationWarning` or SWIG warnings from underlying PyTorch / FastAPI C-extensions do not affect test outcomes; all 29 assertions execute and pass.
- **Is an API key required to run the assistant?**  
  No. AeroPulse includes a built-in deterministic offline engine that generates grounded, cited responses without external API keys or paid accounts.

---

## License & Attribution

This project is released under the **MIT License**. Ambient criteria pollutant observations are provided by the Central Pollution Control Board (CPCB) India Open Government Data platform; meteorological reanalysis is provided by ECMWF ERA5.
