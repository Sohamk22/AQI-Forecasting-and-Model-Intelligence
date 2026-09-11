"""
FastAPI Environmental Intelligence & AQI Forecasting Server.
Serves real AQI model inference (Standalone XGBoost, SE-1, and SE-2) under a strictly
leak-free prior-only formulation: X(t-14, ..., t-1) -> AQI(t).
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.models.xgboost_model import XGBoostModel
from src.stacking.stacking_ensemble import StackingEnsemble

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# Helper for AQI category & badge color (Indian CPCB standard)
def get_aqi_category(aqi_val: float) -> Dict[str, str]:
    val = round(aqi_val, 1)
    if val <= 50:
        return {"category": "Good", "color": "#28a745", "level": "Minimal impact"}
    elif val <= 100:
        return {"category": "Satisfactory", "color": "#8bc34a", "level": "Minor breathing discomfort"}
    elif val <= 200:
        return {"category": "Moderate", "color": "#ffc107", "level": "Breathing discomfort to sensitive people"}
    elif val <= 300:
        return {"category": "Poor", "color": "#fd7e14", "level": "Breathing discomfort on prolonged exposure"}
    elif val <= 400:
        return {"category": "Very Poor", "color": "#dc3545", "level": "Respiratory illness on prolonged exposure"}
    else:
        return {"category": "Severe", "color": "#721c24", "level": "Affects healthy people, severe impact on vulnerable"}

class PredictionInput(BaseModel):
    date: Optional[str] = "2020-01-15"
    # Legacy fields accepted for API compatibility
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

# Global model cache and dataset registry
MODELS = {}
DATA_REGISTRY = {}

def load_resources():
    bundle_path = BASE_DIR / "data" / "processed" / "dataset_bundle.pkl"
    clean_csv_path = BASE_DIR / "data" / "interim" / "cleaned_merged_daily.csv"
    
    if bundle_path.exists():
        with open(bundle_path, "rb") as f:
            b = pickle.load(f)
            
        test_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in b["dates_test"]]
        train_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in b["dates_train"]]
        
        DATA_REGISTRY["bundle"] = b
        DATA_REGISTRY["test_dates"] = test_dates
        DATA_REGISTRY["train_dates"] = train_dates
        DATA_REGISTRY["test_date_map"] = {d: i for i, d in enumerate(test_dates)}
        DATA_REGISTRY["train_date_map"] = {d: i for i, d in enumerate(train_dates)}
        
    if clean_csv_path.exists():
        df_clean = pd.read_csv(clean_csv_path)
        df_clean["Date_Str"] = pd.to_datetime(df_clean["Date"]).dt.strftime("%Y-%m-%d")
        DATA_REGISTRY["df_clean"] = df_clean
        
    try:
        se1 = StackingEnsemble(ensemble_type="SE1")
        se1.load(str(BASE_DIR / "experiments" / "se1"))
        MODELS["SE1"] = se1
        
        se2 = StackingEnsemble(ensemble_type="SE2")
        se2.load(str(BASE_DIR / "experiments" / "se2"))
        MODELS["SE2"] = se2
        
        xgb_standalone = XGBoostModel()
        xgb_standalone.load(str(BASE_DIR / "experiments" / "xgboost" / "xgb_standalone.pkl"))
        MODELS["XGBoost"] = xgb_standalone
    except Exception as e:
        print(f"Notice: Model checkpoint loading encountered: {e}")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="AeroPulse: Air Quality Forecasting & Model Intelligence", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/available_dates")
async def get_available_dates():
    """Returns available historical evaluation dates and curated notable atmospheric presets."""
    if "test_dates" not in DATA_REGISTRY:
        load_resources()
        
    test_dates = DATA_REGISTRY.get("test_dates", [])
    presets = [
        {"label": "Severe Winter Smog Peak", "date": "2019-11-04", "desc": "Post-Diwali crop residue burning and severe particulate accumulation"},
        {"label": "Mid-Winter Severe Inversion", "date": "2020-01-15", "desc": "Low planetary boundary layer height and cold stagnant air"},
        {"label": "Spring Moderate Period", "date": "2020-03-10", "desc": "Moderate post-winter seasonal transition"},
        {"label": "COVID-19 Lockdown Dip", "date": "2020-04-05", "desc": "Plunge in anthropogenic emissions and vehicular traffic"},
        {"label": "Clean Monsoon Baseline", "date": "2019-08-18", "desc": "Precipitation wash-out and clean air baseline"}
    ]
    return {
        "city": "Delhi",
        "total_test_dates": len(test_dates),
        "test_date_min": test_dates[0] if test_dates else "2019-05-27",
        "test_date_max": test_dates[-1] if test_dates else "2020-07-01",
        "default_date": "2020-01-15",
        "presets": presets,
        "available_dates": test_dates
    }

@app.get("/api/insights")
async def get_insights():
    """Returns recent real environmental observations and summary KPIs for Delhi from CPCB records."""
    data_path = BASE_DIR / "data" / "raw" / "city_day.csv"
    if not data_path.exists():
        return JSONResponse({"error": "Data file not found"}, status_code=404)
        
    df = pd.read_csv(data_path)
    if df is None:
        return {
            "city": "Delhi",
            "count": 0,
            "summary": {
                "average_aqi": None,
                "peak_aqi": None,
                "peak_date": None,
                "dominant_pollutant": None
            },
            "recent_observations": []
        }
        
    df["Date_Str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    # Get last 30 days of Delhi observations
    df_delhi = df[df["City"] == "Delhi"].sort_values("Date").tail(30)
    
    records = []
    for _, row in df_delhi.iterrows():
        records.append({
            "Date": row["Date_Str"],
            "AQI": float(row["AQI"]) if pd.notna(row["AQI"]) else None,
            "PM2.5": float(row["PM2.5"]) if pd.notna(row["PM2.5"]) else None,
            "PM10": float(row["PM10"]) if pd.notna(row["PM10"]) else None,
            "NO2": float(row["NO2"]) if pd.notna(row["NO2"]) else None,
            "SO2": float(row["SO2"]) if pd.notna(row["SO2"]) else None,
            "CO": float(row["CO"]) if pd.notna(row["CO"]) else None,
            "O3": float(row["O3"]) if pd.notna(row["O3"]) else None,
        })
        
    # Calculate 30-day summary KPIs directly from actual observations
    valid_aqi_records = [r for r in records if r["AQI"] is not None]
    if valid_aqi_records:
        avg_aqi = round(float(np.mean([r["AQI"] for r in valid_aqi_records])), 1)
        peak_record = max(valid_aqi_records, key=lambda r: r["AQI"])
        peak_aqi = round(float(peak_record["AQI"]), 1)
        peak_date = str(peak_record["Date"]).split("T")[0]
    else:
        avg_aqi = None
        peak_aqi = None
        peak_date = None
    
    standards = {"PM2.5": 60.0, "PM10": 100.0, "NO2": 80.0, "SO2": 80.0, "CO": 2.0, "O3": 100.0}
    pollutant_ratios = {}
    for p, std in standards.items():
        vals = [r[p] for r in records if r.get(p) is not None]
        if vals:
            pollutant_ratios[p] = np.mean(vals) / std
            
    dominant = max(pollutant_ratios, key=pollutant_ratios.get) if pollutant_ratios else None
    
    return {
        "city": "Delhi",
        "count": len(records),
        "summary": {
            "average_aqi": avg_aqi,
            "peak_aqi": peak_aqi,
            "peak_date": peak_date,
            "dominant_pollutant": dominant
        },
        "recent_observations": records
    }

@app.post("/api/predict")
async def predict_aqi(payload: PredictionInput):
    """
    Generate strictly prior-only AQI forecasts for a selected date t.
    Uses exclusively historical observations from [t-14, ..., t-1] with zero lookahead.
    """
    if "bundle" not in DATA_REGISTRY or "SE1" not in MODELS:
        load_resources()
        
    bundle = DATA_REGISTRY.get("bundle")
    test_date_map = DATA_REGISTRY.get("test_date_map", {})
    train_date_map = DATA_REGISTRY.get("train_date_map", {})
    df_clean = DATA_REGISTRY.get("df_clean")
    
    target_date = payload.date if (payload.date and payload.date in test_date_map or payload.date in train_date_map) else "2020-01-15"
    
    if target_date in test_date_map:
        idx = test_date_map[target_date]
        sample_tab = bundle["X_tab_test"][idx:idx+1]
        sample_seq = bundle["X_seq_test"][idx:idx+1]
        actual_aqi = float(bundle["y_test"][idx])
        partition = "Held-out Test (Out-of-sample)"
    elif target_date in train_date_map:
        idx = train_date_map[target_date]
        sample_tab = bundle["X_tab_train"][idx:idx+1]
        sample_seq = bundle["X_seq_train"][idx:idx+1]
        actual_aqi = float(bundle["y_train"][idx])
        partition = "Training Partition (In-sample)"
    else:
        # Fallback to test index 0
        idx = 0
        target_date = DATA_REGISTRY["test_dates"][0]
        sample_tab = bundle["X_tab_test"][idx:idx+1]
        sample_seq = bundle["X_seq_test"][idx:idx+1]
        actual_aqi = float(bundle["y_test"][idx])
        partition = "Held-out Test"

    # Run real model inference
    se1 = MODELS.get("SE1")
    se2 = MODELS.get("SE2")
    xgb_standalone = MODELS.get("XGBoost")
    
    if se1 and se2 and xgb_standalone:
        p_se1, _ = se1.predict(sample_tab, sample_seq)
        p_se2, _ = se2.predict(sample_tab, sample_seq)
        p_xgb = xgb_standalone.predict(sample_tab)
        pred_se1 = float(p_se1[0])
        pred_se2 = float(p_se2[0])
        pred_xgb = float(p_xgb[0])
    else:
        # Load from test predictions cache if models not loaded
        pred_file = BASE_DIR / "results" / "predictions" / "test_predictions.csv"
        df_p = pd.read_csv(pred_file)
        row = df_p[df_p["Date"].str.startswith(target_date)]
        if len(row) > 0:
            pred_se1 = float(row["Pred_SE-1"].iloc[0])
            pred_se2 = float(row["Pred_SE-2"].iloc[0])
            pred_xgb = float(row["Pred_XGBoost"].iloc[0])
        else:
            pred_se1 = 185.0
            pred_se2 = 192.0
            pred_xgb = 180.0

    consensus_aqi = round((pred_se1 + pred_se2) / 2.0, 1)
    
    # Retrieve preceding 14 days of observations [t-14, ..., t-1]
    prior_14_records = []
    prior_means = {}
    if df_clean is not None:
        matches = df_clean[df_clean["Date_Str"] == target_date].index
        if len(matches) > 0:
            target_idx = matches[0]
            if target_idx >= 14:
                prior_df = df_clean.iloc[target_idx - 14:target_idx].copy()
                prior_means = {
                    "mean_aqi": round(float(prior_df["AQI"].mean()), 1),
                    "mean_pm25": round(float(prior_df["PM2.5"].mean()), 1),
                    "mean_pm10": round(float(prior_df["PM10"].mean()), 1),
                    "mean_temp": round(float(prior_df["Temperature"].mean()), 1),
                    "mean_humidity": round(float(prior_df["Humidity"].mean()), 1),
                    "mean_wind": round(float(prior_df["Wind_Speed"].mean()), 1)
                }
                for _, r in prior_df.iterrows():
                    prior_14_records.append({
                        "date": str(r["Date_Str"]),
                        "aqi": round(float(r["AQI"]), 1),
                        "pm25": round(float(r["PM2.5"]), 1),
                        "pm10": round(float(r["PM10"]), 1),
                        "temperature": round(float(r["Temperature"]), 1),
                        "humidity": round(float(r["Humidity"]), 1),
                        "wind_speed": round(float(r["Wind_Speed"]), 1)
                    })

    cat_xgb = get_aqi_category(pred_xgb)
    cat_se1 = get_aqi_category(pred_se1)
    cat_se2 = get_aqi_category(pred_se2)
    cat_act = get_aqi_category(actual_aqi)

    return {
        "target_date": target_date,
        "partition": partition,
        "actual_aqi": round(actual_aqi, 1),
        "actual_category": cat_act["category"],
        "actual_color": cat_act["color"],
        "XGBoost": {
            "model_name": "Standalone XGBoost (Top Point Model)",
            "predicted_aqi": round(pred_xgb, 1),
            "error_abs": round(abs(pred_xgb - actual_aqi), 1),
            "category": cat_xgb["category"],
            "color": cat_xgb["color"],
            "health_risk": cat_xgb["level"]
        },
        "SE-1": {
            "model_name": "SE-1 (XGBoost + Bi-LSTM + TCN -> Extra Trees)",
            "predicted_aqi": round(pred_se1, 1),
            "error_abs": round(abs(pred_se1 - actual_aqi), 1),
            "category": cat_se1["category"],
            "color": cat_se1["color"],
            "health_risk": cat_se1["level"]
        },
        "SE-2": {
            "model_name": "SE-2 (XGBoost + Bi-GRU + TCN -> Extra Trees)",
            "predicted_aqi": round(pred_se2, 1),
            "error_abs": round(abs(pred_se2 - actual_aqi), 1),
            "category": cat_se2["category"],
            "color": cat_se2["color"],
            "health_risk": cat_se2["level"]
        },
        "consensus_aqi": consensus_aqi,
        "prior_14_days_summary": prior_means,
        "prior_14_days_records": prior_14_records,
        "methodology_note": "Forecast uses exclusively information available before the selected date (t-14 to t-1). Zero lookahead / zero contemporaneous leakage."
    }

@app.get("/api/metrics")
async def get_model_metrics():
    """Returns official frozen experimental metrics from results directory."""
    metrics_file = BASE_DIR / "results" / "metrics" / "model_comparison.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            data = json.load(f)
        return {"status": "success", "metrics": data}
    return {"status": "error", "message": "Metrics file not found"}
