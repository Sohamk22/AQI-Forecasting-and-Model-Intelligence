"""
Central configuration for AQI Forecasting Research Project.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

CITY_COORDINATES: Dict[str, Dict[str, float]] = {
    "Delhi": {"latitude": 28.6139, "longitude": 77.2090},
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707},
    "Hyderabad": {"latitude": 17.3850, "longitude": 78.4867},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
    "Kolkata": {"latitude": 22.5726, "longitude": 88.3639}
}

@dataclass
class DataConfig:
    city: str = "Delhi"
    raw_pollutant_file: str = "data/raw/city_day.csv"
    raw_weather_file: str = "data/raw/weather_daily.csv"
    processed_file: str = "data/processed/aqi_delhi_processed.csv"
    pollutant_cols: List[str] = field(default_factory=lambda: [
        "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"
    ])
    weather_cols: List[str] = field(default_factory=lambda: [
        "Temperature", "Humidity", "Wind_Speed"
    ])
    target_col: str = "AQI"
    train_test_split_ratio: float = 0.80
    sequence_length: int = 14
    forecast_horizon: int = 1
    random_seed: int = 42

@dataclass
class ModelConfig:
    # XGBoost
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Bi-LSTM
    bilstm_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 40,
        "patience": 8
    })
    
    # Bi-GRU (matches Bi-LSTM for controlled comparison)
    bigru_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 40,
        "patience": 8
    })
    
    # TCN
    tcn_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_channels": [32, 32, 64, 64],
        "kernel_size": 3,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 40,
        "patience": 8
    })
    
    # Extra Trees (Meta-Learner & Standalone Baseline)
    et_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1
    })

    # Stacking TimeSeriesSplit CV
    n_splits_cv: int = 5
