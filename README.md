# AeroPulse

> **Understand your air. Make better decisions.**

AeroPulse is an ambient air-quality intelligence platform that combines **AQI forecasting** with a **policy-aware RAG assistant**.

The platform has two main parts:

- **AQI Forecasting:** Machine-learning models predict AQI using only information from previous days.
- **Policy Assistant:** A document-based RAG assistant answers air-quality and environmental-policy questions using the official documents included in the project.

---

## 🚀 Run AeroPulse Locally

If you just want to run the project, follow these steps.

### 1. Clone the repository

Open Terminal and run:

```bash
git clone https://github.com/Sohamk22/AQI-Forecasting-and-Model-Intelligence.git
cd AQI-Forecasting-and-Model-Intelligence
```

### 2. Use Python 3.12

Python **3.12.x** is recommended for this project.

Check your version:

```bash
python3.12 --version
```

You should see something similar to:

```text
Python 3.12.14
```

### 3. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

You should now see `(.venv)` at the beginning of your terminal.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs the libraries required for the forecasting models, FastAPI application, RAG system, and supporting components.

### 5. Start the application

Run:

```bash
uvicorn app.main:app --reload
```

**Important:** `main.py` is inside the `app/` directory, so use `app.main:app`.

When the server starts, you should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

### 6. Open the website

Open your browser and go to:

```text
http://127.0.0.1:8000
```

That's it. AeroPulse should now be running locally.

---

## 🌫️ What Can You Do With AeroPulse?

### AQI Forecasting

Use the forecasting interface to:

- View AQI forecasts
- Compare predictions from multiple models
- Explore historical AQI behaviour
- View prior pollutant trends
- Understand the factors behind predictions

The forecasting system uses information from previous observations:

```text
X(t-14, ..., t-1) → AQI(t)
```

This means information from the target day is not used to make the prediction.

### 🤖 Policy & Air-Quality Assistant

The Public Assistant provides answers to questions about:

- Air-quality conditions
- Outdoor activity
- Pollution causes
- Government actions
- GRAP
- Environmental regulations
- AQI interpretation

Try questions such as:

> Is it safe for a morning jog today?

or:

> What GRAP actions apply during severe pollution?

The assistant uses the policy documents included in the repository and can operate using its built-in offline retrieval and response mechanism.

**No API key is required for the basic local workflow.**

---

## 🔬 What Makes the Forecasting Setup Different?

During development, an important target-leakage issue was discovered.

Using same-day pollutant measurements to predict same-day AQI would make the task closer to calculating AQI than forecasting it.

The original formulation was therefore changed to a strictly prior-only setup:

```text
X(t-14, ..., t-1) → AQI(t)
```

Rolling features are also shifted so that information from day `t` cannot enter the prediction for day `t`.

The evaluation uses:

- Chronological train/test splitting
- A historical climatology baseline
- Seven forecasting model configurations
- Leakage checks
- Out-of-fold stacking for the ensemble models

This makes the evaluation closer to a genuine forecasting problem rather than concurrent AQI estimation.

---

## 📊 Model Benchmark

The Delhi held-out evaluation compared the following approaches:

| Model | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|
| Historical Climatology | 100.45 | 0.2097 |
| Dilated TCN | 77.24 | 0.5328 |
| Bi-LSTM | 59.15 | 0.7260 |
| SE-1 | 45.85 | 0.8354 |
| Bi-GRU | 42.67 | 0.8574 |
| SE-2 | 39.14 | 0.8800 |
| Extra Trees | 36.37 | 0.8964 |
| **XGBoost** | **34.58** | **0.9063** |

The results show that tree-based models perform particularly well when working with engineered prior-day lag and rolling features, while the sequence models provide a different inductive bias for extreme pollution events.

---

## 🧠 Policy RAG

The RAG component works with statutory environmental documents included in:

```text
data/policy_documents/
```

The knowledge base includes documents covering areas such as:

- GRAP
- NCAP
- Air Act 1981
- CAQM Act 2021
- CPCB air-quality standards

The retrieval pipeline combines:

- BM25 keyword retrieval
- Dense vector retrieval
- Reciprocal Rank Fusion
- Citation verification
- Intent routing
- Prompt-injection safeguards
- Offline fallback responses

The goal is to provide answers grounded in the project's supplied policy documents rather than relying only on general model knowledge.

---

## 🧪 Running the Tests

Tests are available under:

```text
tests/
```

To run them:

```bash
python -m pytest tests/ -v
```

The test suite covers areas including:

- API behaviour
- Forecasting logic
- Leakage protection
- Stacking
- RAG retrieval
- Citation verification
- Assistant behaviour
- Safety checks

> **Note:** The tests are an optional verification step. You do not need to run them to launch the web application.

---

## 📁 Project Structure

```text
AQI-Forecasting-and-Model-Intelligence/
│
├── app/
│   ├── main.py              # FastAPI application
│   ├── static/              # Frontend assets
│   └── templates/           # Web interface
│
├── data/
│   ├── raw/                 # Raw air-quality/weather data
│   ├── processed/           # Processed datasets
│   ├── policy_documents/    # RAG source documents
│   └── cache/               # Precomputed RAG resources
│
├── experiments/             # Model experiments
├── notebooks/               # Exploratory analysis
├── results/                 # Metrics and research outputs
├── src/
│   ├── models/              # Forecasting models
│   ├── features/            # Feature engineering
│   ├── evaluation/          # Evaluation utilities
│   ├── stacking/            # Ensemble models
│   └── rag/                 # RAG and assistant
│
├── tests/                   # Automated tests
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🛠️ Troubleshooting

### `python3.12: command not found`

Install Python 3.12 and make sure it is available from your Terminal.

### `pip install` fails

Check that the virtual environment is active:

```bash
source .venv/bin/activate
```

Then verify:

```bash
python --version
```

It should show Python 3.12.x.

### `Could not import module "main"`

Use:

```bash
uvicorn app.main:app --reload
```

not:

```bash
uvicorn main:app --reload
```

The FastAPI entry point is located at:

```text
app/main.py
```

### Port 8000 is already in use

Stop the other running server with:

```text
Ctrl + C
```

Or use another port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```

---

## 📝 Project Evolution

### Phase 1 — Original Coursework

The original project focused on the AQI forecasting system, including:

- Prior-only forecasting
- Seven-model comparison
- Historical climatology baseline
- Chronological evaluation
- Leakage protection
- FastAPI application

### Phase 2 — Post-Submission Development

The project was subsequently extended with:

- Policy RAG
- Public air-quality assistant
- Statutory document retrieval
- Citation verification
- Safety and intent routing
- UI improvements
- Expanded research analysis

These later additions are intentionally distinguished from the original coursework submission.

---

## 💡 What I Would Do Differently

Future improvements could include:

1. **Multi-horizon forecasting** for 1, 3, and 7-day predictions.
2. **Spatial modelling** to better capture pollution movement between cities.
3. **Forecast weather inputs** instead of relying only on historical weather observations.
4. **Larger and more diverse datasets** to improve generalisation across regions and seasons.

---

## 📌 Limitations

- Forecasting performance depends on the quality and availability of historical observations.
- The current operational forecasting setup focuses primarily on short-horizon prediction.
- The RAG assistant is grounded in the policy documents included with the project and should not be treated as a substitute for official legal or medical advice.
- Research results and the interactive application serve different purposes: the former evaluates forecasting methodology, while the latter provides an accessible demonstration.

---

## License & Data

This project is released under the **MIT License**.

Air-quality observations are sourced from the Central Pollution Control Board (CPCB) India Open Government Data platform, while meteorological reanalysis data is based on ECMWF ERA5.
