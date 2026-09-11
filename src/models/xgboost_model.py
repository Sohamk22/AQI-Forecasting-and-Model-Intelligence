"""
XGBoost model wrapper for AQI forecasting.
"""
import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("XGBoostModel")

class XGBoostModel:
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": 2
            }
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            eval_set=None, verbose: bool = False):
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved XGBoost model to {filepath}")
        
    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        logger.info(f"Loaded XGBoost model from {filepath}")
        return self
