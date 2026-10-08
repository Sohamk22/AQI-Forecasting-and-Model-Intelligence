# AeroPulse

### Air Quality Forecasting & Policy Intelligence

AeroPulse is an air-quality intelligence system that combines **AQI forecasting** with a **policy-aware AI assistant**.

It provides:

- AQI predictions using trained machine-learning and deep-learning models
- A comparison of seven forecasting approaches
- A policy assistant backed by official air-quality and environmental documents
- A simple web interface for interacting with both capabilities

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sohamk22/AQI-Forecasting-and-Model-Intelligence.git
cd AQI-Forecasting-and-Model-Intelligence
```

### 2. Use Python 3.12

Python **3.12** is recommended for this project.

Check your version:

```bash
python3.12 --version
```

### 3. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The repository includes the trained models, datasets required by the application, and policy documents used by the assistant.

### 5. Start AeroPulse

```bash
uvicorn app.main:app --reload
```

You should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

### 6. Open the application

Go to:

```text
http://127.0.0.1:8000
```

That's it.

---

## What Can You Do?

### AQI Forecasting

Enter the required date and location information through the web interface to generate an AQI prediction using the trained forecasting models.

The forecasting pipeline uses only information available **before the prediction date**:

```text
X(t-14), ..., X(t-1) → AQI(t)
```

This prevents future information from leaking into the prediction.

### Policy Assistant

The built-in assistant can answer questions about:

- Air-quality regulations
- Government actions
- Pollution-control measures
- Outdoor activity guidance
- Evidence from official policy documents

The assistant uses a hybrid retrieval pipeline combining keyword and semantic search over the included policy documents.

---

## Model Performance

The project evaluates multiple forecasting approaches on a chronological held-out test set.

| Model | RMSE | R² |
|---|---:|---:|
| Historical Climatology | 100.45 | 0.2097 |
| TCN | 77.24 | 0.5328 |
| Bi-LSTM | 59.15 | 0.7260 |
| SE-1 | 45.85 | 0.8354 |
| Bi-GRU | 42.67 | 0.8574 |
| SE-2 | 39.14 | 0.8800 |
| Extra Trees | 36.37 | 0.8964 |
| XGBoost | **34.58** | **0.9063** |

The results show that tree-based models perform particularly well on the engineered lag and rolling features, while the stacked ensembles combine complementary modelling approaches.

---

## Project Structure

```text
AeroPulse/
├── app/                    # FastAPI application and web interface
├── src/
│   ├── models/             # Forecasting models
│   ├── stacking/           # Stacked ensemble models
│   └── rag/                # Policy retrieval pipeline
├── data/
│   └── policy_documents/   # Policy and regulatory documents
├── experiments/            # Trained model checkpoints
├── tests/                  # Automated tests
├── results/                # Evaluation results
├── notebooks/              # Analysis notebooks
├── requirements.txt
└── README.md
```

---

## Important Note

The original coursework focused on the **leak-free AQI forecasting pipeline and model comparison**.

The policy assistant, RAG functionality, and additional interface improvements were developed later as an extension of the project.

---

## Limitations

- Forecast accuracy depends on the quality and availability of historical air-quality data.
- The current system focuses on the forecasting setup implemented in the project rather than multi-horizon operational forecasting.
- Policy answers are grounded in the documents included with the repository.

---

## License & Data

This project was developed for academic and research purposes.

Please refer to the included project documentation for dataset and policy-document attribution.
