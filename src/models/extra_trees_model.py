"""
Extra Trees model wrapper for AQI forecasting (both as standalone baseline and meta-learner).
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.ensemble import ExtraTreesRegressor
from src.utils.logger import get_logger

logger = get_logger("ExtraTreesModel")

class ExtraTreesModel:
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": 2
            }
        self.params = params
        self.model = ExtraTreesRegressor(**self.params)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved Extra Trees model to {filepath}")
        
    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        logger.info(f"Loaded Extra Trees model from {filepath}")
        return self
