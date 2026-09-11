"""
Evaluation metrics for AQI forecasting models.
Includes RMSE, MAE, MAPE, sMAPE, and R2.
"""
import numpy as np
from typing import Dict

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    """Mean Absolute Percentage Error (percentage format e.g. 15.2%)."""
    mask = np.abs(y_true) > eps
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-5
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)

def r2_score_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate and return dictionary of all evaluation metrics."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    
    return {
        "RMSE": round(rmse(y_t, y_p), 2),
        "MAE": round(mae(y_t, y_p), 2),
        "MAPE": round(mape(y_t, y_p), 2),
        "sMAPE": round(smape(y_t, y_p), 2),
        "R2": round(r2_score_custom(y_t, y_p), 4)
    }
