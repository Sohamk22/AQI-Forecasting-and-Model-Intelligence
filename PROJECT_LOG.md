# AeroPulse — Project Log

This log captures the major decisions, problems, and changes that shaped AeroPulse.

---

## 1. Project Goal

The goal was to build a system that could forecast next-day AQI using historical air-quality and weather data, while comparing different machine-learning and deep-learning approaches.

---

## 2. Data Leakage — The First Major Problem

During validation, I found that the initial pipeline was using same-day pollutant values to predict same-day AQI.

That was a problem because AQI is calculated from pollutant concentrations. In other words, the model was being given information that would not actually be available when making a forecast.

So the original results could not be treated as genuine forecasting results.

### What I changed

I rebuilt the feature pipeline so that the model only uses information available before the target day.

- Previous AQI and pollutant values
- Previous weather observations
- Lag features
- Shifted rolling statistics
- Calendar features
- Previous 14 days for the sequence models

The target-day pollutant measurements were removed.

The forecasting setup became:

**Previous 14 days → Target day's AQI**

---

## 3. Time-Series Validation

I also found that standard random cross-validation was not appropriate for this problem.

For a forecasting task, training on future observations while evaluating on earlier observations can leak information across time.

I therefore changed the stacking evaluation to a 5-fold expanding TimeSeriesSplit.

The base models generate out-of-fold predictions from future folds, and the meta-learner is trained only on those predictions.

---

## 4. The Result I Didn't Expect

After rebuilding the pipeline, I expected the stacked ensembles to be the strongest models.

They weren't.

XGBoost performed the best overall on the held-out test period.

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

SE-2 was the best ensemble, but it still did not beat XGBoost.

This showed that adding more model complexity did not necessarily improve the forecast.

---

## 5. Preprocessing Caveat

During the audit, I also found that linear interpolation had been applied across the complete dataset before the train/test split.

This creates a potential dependency around the split boundary.

I kept the capstone benchmark frozen rather than changing the experiment at this stage, and documented this as a limitation to address in the future research version.

---

## 6. Where the Project Ended Up

AeroPulse is now a Delhi-focused AQI forecasting and model-intelligence system covering 2015–2020.

It includes:

- 114 engineered tabular features
- 14-day historical sequences
- 8 model approaches
- TimeSeriesSplit-based stacking
- Held-out chronological evaluation
- Interactive date-based forecasting
- Environmental insights
- Model benchmarking
- Automated tests

---

## 7. What I Took Away

- Forecasting problems need a clear definition of what information is actually available at prediction time.
- AQI is especially sensitive to leakage because it is derived from pollutant concentrations.
- Time-series validation needs to respect chronology.
- More complex models do not automatically perform better.
- Sometimes the most useful result is the one you didn't expect.

The biggest change in AeroPulse wasn't a new model.

It was making sure the question being answered was actually the question I started with.