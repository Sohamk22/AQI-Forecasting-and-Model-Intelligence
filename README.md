# AeroPulse — Air Quality Forecasting & Model Intelligence

### Comparing Machine Learning, Deep Learning and Ensemble Models for AQI Forecasting

AeroPulse is an end-to-end air quality forecasting project built to answer a simple question:

> **Which type of machine learning model can best predict tomorrow's AQI using only information from previous days?**

The project uses historical air-quality and weather data from Delhi NCR and compares **8 different approaches**, ranging from traditional tree-based models to deep learning and stacking ensembles.

The result was not what I initially expected: **XGBoost outperformed the deep learning and ensemble models**, achieving an RMSE of **34.58** and an R² of **0.9063** on the held-out test set.

## What makes the project interesting?

- Compares classical ML, deep learning and ensemble approaches under the same forecasting setup
- Uses strictly prior-day information to avoid same-day lookahead
- Evaluates models on a chronological held-out test set
- Goes beyond overall accuracy to study model behaviour during extreme pollution events
- Includes an interactive dashboard for forecasting and model comparison

## Model Comparison

| Model | RMSE ↓ | R² ↑ |
|---|---:|---:|
| Historical Climatology | 100.45 | 0.2097 |
| TCN | 77.24 | 0.5328 |
| Bi-LSTM | 59.15 | 0.7260 |
| Bi-GRU | 42.67 | 0.8574 |
| SE-2 | 39.14 | 0.8800 |
| Extra Trees | 36.37 | 0.8964 |
| **XGBoost** | **34.58** | **0.9063** |

## Dashboard

AeroPulse includes an interactive dashboard to:

- Forecast AQI for historical dates
- Explore air-quality trends
- Compare model performance


## How to Run

1. Clone the repository
git clone https://github.com/Sohamk22/AQI-Forecasting-and-Model-Intelligence.git
cd AQI-Forecasting-and-Model-Intelligence]

2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

6. Run the tests
pytest tests/ -v

8. Start the dashboard
uvicorn app.main:app --host 0.0.0.0 --port 8000

Open http://localhost:8000 in your browser.

## Tech Stack

Python, XGBoost, PyTorch, Scikit-learn, Pandas, FastAPI and JavaScript.

## Project Status

**Complete — AeroPulse**

The project is focused on one thing: **building a fair model comparison and letting the results, rather than model complexity, determine what works best.**
