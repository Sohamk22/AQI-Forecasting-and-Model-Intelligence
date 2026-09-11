"""
Dynamic Time Warping (DTW) Analysis for AQI Trajectory Comparison.
Aligns actual and forecasted sequences to evaluate temporal shape fidelity,
timing shifts, and structural distortion independently of point-by-point error.
"""
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import Dict, Any, Tuple, List
from src.utils.logger import get_logger

logger = get_logger("DTWAnalysis")

def compute_dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute Dynamic Time Warping distance and warping path between two 1D sequences."""
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    distance, path = fastdtw(a, b, dist=lambda x, y: abs(float(x) - float(y)))
    normalized_distance = distance / (len(a) + len(b))
    return float(distance), float(normalized_distance), path

def compare_models_dtw(y_true: np.ndarray, y_preds: Dict[str, np.ndarray],
                       window_size: int = 30) -> Dict[str, Any]:
    """
    Evaluate DTW distance between ground truth and model predictions across
    rolling temporal windows (e.g. 30 days) to quantify structural trajectory agreement.
    """
    logger.info(f"Computing DTW alignment across {len(y_preds)} models over {len(y_true)} test samples...")
    results = {}
    n = len(y_true)
    
    # Overall full test sequence DTW
    for model_name, pred in y_preds.items():
        dist, norm_dist, path = compute_dtw_distance(y_true, pred)
        
        # Also compute average 30-day rolling window DTW distance
        rolling_distances = []
        step = 15  # 50% overlap
        for i in range(0, n - window_size + 1, step):
            w_true = y_true[i:i + window_size]
            w_pred = pred[i:i + window_size]
            d, nd, _ = compute_dtw_distance(w_true, w_pred)
            rolling_distances.append(nd)
            
        avg_rolling_dtw = float(np.mean(rolling_distances)) if rolling_distances else norm_dist
        
        results[model_name] = {
            "total_dtw_distance": round(dist, 2),
            "normalized_dtw_distance": round(norm_dist, 4),
            "mean_30day_window_dtw": round(avg_rolling_dtw, 4),
            "warping_path_length": len(path)
        }
        
    return results
