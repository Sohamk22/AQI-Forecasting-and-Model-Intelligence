"""
Pollution Spike Detection and Extreme-Event Performance Evaluation.
Implements the 30-day rolling z-score methodology (z > 3.0) specified in the research paper.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.evaluation.metrics import calculate_all_metrics
from src.utils.logger import get_logger

logger = get_logger("SpikeDetection")

def detect_spikes_rolling_zscore(series: pd.Series, window: int = 30, threshold: float = 3.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate rolling z-scores using a 30-day moving average and standard deviation (shifted by 1 to prevent leakage).
    Returns: (is_spike boolean series, rolling_mean, rolling_std, z_scores)
    """
    s = series.copy()
    shifted = s.shift(1)
    rolling_mean = shifted.rolling(window=window, min_periods=7).mean().bfill()
    rolling_std = shifted.rolling(window=window, min_periods=7).std().replace(0, np.nan).bfill()
    
    z_scores = (s - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)
    is_spike = z_scores > threshold
    
    return is_spike, rolling_mean, z_scores

def evaluate_spike_performance(y_true: np.ndarray, y_preds: Dict[str, np.ndarray],
                               is_spike_mask: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate model errors specifically during detected pollution spike periods
    versus normal periods.
    """
    n_spikes = int(np.sum(is_spike_mask))
    n_total = len(y_true)
    logger.info(f"Evaluating spike performance: {n_spikes}/{n_total} observations classified as spikes ({n_spikes/n_total*100:.2f}%)")
    
    results = {
        "spike_count": n_spikes,
        "total_count": n_total,
        "spike_percentage": round(n_spikes / n_total * 100, 2),
        "models": {}
    }
    
    for model_name, pred in y_preds.items():
        # Overall metrics
        overall_m = calculate_all_metrics(y_true, pred)
        
        # Spike-period metrics
        if n_spikes > 0:
            spike_m = calculate_all_metrics(y_true[is_spike_mask], pred[is_spike_mask])
            # Underestimation metric: how many spikes did the model predict below actual?
            underestimations = np.sum(pred[is_spike_mask] < y_true[is_spike_mask])
            underestimation_rate = float(underestimations / n_spikes * 100.0)
            avg_underestimate_mag = float(np.mean(np.maximum(0, y_true[is_spike_mask] - pred[is_spike_mask])))
        else:
            spike_m = {}
            underestimation_rate = 0.0
            avg_underestimate_mag = 0.0
            
        results["models"][model_name] = {
            "overall": overall_m,
            "spike_period": spike_m,
            "underestimation_rate_pct": round(underestimation_rate, 2),
            "avg_underestimate_magnitude": round(avg_underestimate_mag, 2)
        }
        
    return results
