# AeroPulse — Project Log

This log records the major decisions, problems, validation findings, and implementation changes that shaped AeroPulse.

---

## 1. Project Goal

The original goal was to build a system that could forecast next-day AQI using historical air-quality and weather data while comparing different machine-learning and deep-learning approaches.

The project was designed as a forecasting problem, meaning that the model should only use information that would realistically be available before the target day's AQI was known.

---

## 2. Data Leakage — The First Major Problem

During validation, I found that the initial pipeline was using same-day pollutant values to predict same-day AQI.

This was a serious issue because AQI is calculated from pollutant concentrations. Giving the model target-day pollutant measurements meant that the model was effectively receiving information about the target it was supposed to forecast.

Therefore, the original results could not be treated as genuine forecasting results.

### What I changed

I rebuilt the feature pipeline so that the model only uses information available before the target day.

The final feature setup includes:

- Previous AQI and pollutant values
- Previous weather observations
- Lag features
- Shifted rolling statistics
- Calendar features
- Previous 14 days for sequence models

Target-day pollutant measurements were removed.

The forecasting formulation became:

**Previous 14 days → Target day's AQI**

More formally:

**X(t-14), ..., X(t-1) → AQI(t)**

This change made the forecasting setup consistent with the actual prediction task.

---

## 3. Time-Series Validation

I also found that standard random cross-validation was not appropriate for this problem.

In a forecasting task, randomly mixing observations can allow information from later periods to influence evaluation of earlier periods.

I therefore changed the stacking evaluation to a chronological expanding `TimeSeriesSplit` approach.

The base models generate out-of-fold predictions using only the appropriate historical training folds, and the meta-learner is trained on those out-of-fold predictions.

This keeps the stacking process consistent with the temporal nature of the problem.

---

## 4. The Result I Didn't Expect

After rebuilding the pipeline, I expected the stacked ensembles to be the strongest models.

They weren't.

XGBoost performed the best overall on the held-out chronological test period.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **XGBoost** | **34.58** | **26.08** | **0.9063** |
| Extra Trees | 36.37 | 26.90 | 0.8964 |
| **SE-2** | **39.14** | **29.84** | **0.8800** |
| Bi-GRU | 42.67 | 30.73 | 0.8574 |
| SE-1 | 45.85 | 37.97 | 0.8354 |
| Bi-LSTM | 59.15 | 45.72 | 0.7260 |
| TCN | 77.24 | 57.62 | 0.5328 |
| Historical Climatology | 100.45 | 83.17 | 0.2097 |

SE-2 was the strongest ensemble, but it still did not outperform standalone XGBoost.

This was an important result: increasing model complexity or combining several architectures does not automatically produce a better forecast.

---

## 5. Preprocessing Caveat

During the audit, I also found that linear interpolation had been applied across the complete dataset before the train/test split.

This creates a potential dependency around the train/test boundary.

Rather than silently changing the original capstone benchmark, I kept the reported experiment frozen and documented this as a limitation for future research work.

A future version should perform preprocessing separately within the appropriate temporal partitions.

---

## 6. Clean-Clone and Reproducibility Audit

After the project was extended and the repository documentation was improved, I performed a clean-clone verification of the public GitHub repository.

The repository was cloned into a fresh local environment using Python 3.12.

The initial dependency installation exposed an environment compatibility issue:

- NumPy 2.3.5 was installed by the unconstrained requirements.
- PyTorch 2.2.2 could not properly interface with this NumPy version.
- The application initially loaded its web page but AQI prediction failed when PyTorch attempted to convert model output to NumPy.

The runtime error was:

**`RuntimeError: Numpy is not available`**

### Resolution

The environment was adjusted to use:

- NumPy 1.26.4
- SciPy 1.12.0

After this change, the application started successfully and the trained models loaded correctly.

The following application requests were then verified successfully:

- `GET /` → 200 OK
- `POST /api/predict` → 200 OK
- `POST /api/assistant/query` → 200 OK

This confirmed that the actual application workflow was functioning after the dependency correction.

---

## 7. Test Environment Issue

The first test run also exposed a missing test dependency.

The FastAPI/Starlette test client required `httpx2`, which was not present in the original dependency list.

After installing `httpx2`, the complete test suite was able to run.

The suite collected 29 tests.

The final run produced:

**27 passed, 2 failed**

The two failures were both related to the policy-document loader because `pypdf` was missing.

The application itself was already able to start and successfully process both the forecasting and assistant API requests.

The dependency list was therefore updated to include:

- `httpx2`
- `pypdf`
- compatible NumPy and SciPy versions
- the matching scikit-learn version used by the saved model artifacts

The purpose of this change is to make the environment reproducible from a fresh clone rather than relying on manually installed packages.

---

## 8. Final Project Scope

AeroPulse is a Delhi-focused AQI forecasting and model-intelligence system covering the 2015–2020 dataset.

The current system includes:

- 114 engineered tabular features
- 14-day historical sequences
- Multiple machine-learning and deep-learning forecasting models
- Two stacked ensemble configurations
- Chronological held-out evaluation
- TimeSeriesSplit-based stacking
- Interactive date-based forecasting
- Environmental insights
- Policy-document retrieval and question answering
- FastAPI backend
- Web interface
- Automated tests

---

## 9. Project Evolution

### Phase 1 — Original Coursework

The original coursework focused on:

- Leak-free AQI forecasting
- Model comparison
- Chronological evaluation
- Stacked ensemble experiments
- FastAPI application
- Automated leakage and application tests

The target was to build and evaluate a genuine forecasting pipeline rather than a same-day AQI estimation system.

### Phase 2 — Post-Submission Extension

After the original coursework submission, the project was extended with:

- Policy-document RAG
- Hybrid keyword and semantic retrieval
- Public-facing policy assistant
- Improved web interface
- Additional API functionality
- Expanded automated testing
- Additional research and model-analysis work

These additions are documented separately from the original coursework scope.

---

## 10. What I Took Away

The most important lessons from the project were:

- Forecasting problems need a clear definition of what information is available at prediction time.
- AQI is especially sensitive to leakage because it is derived from pollutant concentrations.
- Time-series validation must respect chronology.
- More complex models do not automatically perform better.
- Dependency versions matter when saved machine-learning models are being redistributed.
- A project is not truly reproducible if it works only in the original development environment.
- Sometimes the most useful result is the one you did not expect.

The biggest change in AeroPulse was not a new model.

It was making sure that the question being answered was actually the question the project started with.
