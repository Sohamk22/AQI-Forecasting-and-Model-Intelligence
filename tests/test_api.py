"""
Unit tests for AeroPulse FastAPI Web Dashboard endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "AeroPulse" in response.text
    assert "Prior-Only Formulation" in response.text

def test_api_insights():
    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.json()
    assert "recent_observations" in data
    assert len(data["recent_observations"]) > 0
    assert "summary" in data
    summary = data["summary"]
    assert "average_aqi" in summary and summary["average_aqi"] is not None
    assert "peak_aqi" in summary and summary["peak_aqi"] is not None
    assert "peak_date" in summary and summary["peak_date"] is not None
    assert "dominant_pollutant" in summary and summary["dominant_pollutant"] is not None

def test_api_available_dates():
    response = client.get("/api/available_dates")
    assert response.status_code == 200
    data = response.json()
    assert "available_dates" in data
    assert len(data["available_dates"]) > 0
    assert "presets" in data
    assert len(data["presets"]) > 0

def test_api_predict_date():
    """Verify strictly prior-only date prediction for held-out test date."""
    payload = {"date": "2020-01-15"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_date"] == "2020-01-15"
    assert "actual_aqi" in data
    assert data["actual_aqi"] > 0
    assert "XGBoost" in data
    assert "SE-1" in data
    assert "SE-2" in data
    assert data["XGBoost"]["predicted_aqi"] > 0
    assert data["SE-1"]["predicted_aqi"] > 0
    assert data["SE-2"]["predicted_aqi"] > 0
    assert "prior_14_days_summary" in data
    assert "mean_aqi" in data["prior_14_days_summary"]
    assert "methodology_note" in data

def test_api_predict_legacy_compatibility():
    """Verify legacy payload compatibility without crash."""
    payload = {
        "pm25": 140.0,
        "pm10": 250.0,
        "no2": 50.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "SE-1" in data
    assert "SE-2" in data
    assert "XGBoost" in data

def test_api_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["metrics"]) > 0
