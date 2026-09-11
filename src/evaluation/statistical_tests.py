"""
Statistical significance testing between competing AQI forecasting architectures.
Implements:
1. Paired Student's t-test (as specified in paper Table 4)
2. Wilcoxon Signed-Rank Test (non-parametric robustness check)
3. Diebold-Mariano Test (time-series econometric benchmark with autocorrelation correction)
"""
import numpy as np
from scipy import stats
from typing import Dict, Any

def paired_student_ttest(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict[str, Any]:
    """
    Perform paired Student's t-test on prediction errors between Model A and Model B.
    A positive t-stat indicates Model B has larger errors than Model A (Model A is superior).
    """
    diff = errors_b - errors_a
    t_stat, p_val = stats.ttest_rel(errors_b, errors_a)
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "mean_error_diff": round(float(np.mean(diff)), 4),
        "std_error_diff": round(float(np.std(diff)), 4),
        "is_significant_005": bool(p_val < 0.05),
        "is_significant_001": bool(p_val < 0.001)
    }

def wilcoxon_signed_rank(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict[str, Any]:
    """Non-parametric alternative to paired t-test without assuming normal differences."""
    stat, p_val = stats.wilcoxon(errors_b, errors_a, zero_method="pratt")
    return {
        "w_statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "is_significant_005": bool(p_val < 0.05)
    }

def diebold_mariano_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                         h: int = 1, criterion: str = "MSE") -> Dict[str, Any]:
    """
    Diebold-Mariano test for predictive accuracy equivalence under autocorrelated loss.
    h: forecast horizon (default 1).
    """
    if criterion == "MAE":
        d = np.abs(y_true - y_pred_a) - np.abs(y_true - y_pred_b)
    else:  # MSE
        d = (y_true - y_pred_a)**2 - (y_true - y_pred_b)**2
        
    n = len(d)
    mean_d = np.mean(d)
    
    # Autocovariance up to lag h-1 (for 1-step ahead, lag 0)
    gamma0 = np.var(d, ddof=0)
    variance_d = gamma0
    for lag in range(1, h):
        gamma_k = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        variance_d += 2 * (1 - lag / h) * gamma_k
        
    if variance_d <= 0:
        variance_d = 1e-6
        
    dm_stat = mean_d / np.sqrt(variance_d / n)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    
    return {
        "dm_statistic": round(float(dm_stat), 4),
        "p_value": float(p_val),
        "loss_criterion": criterion,
        "is_significant_005": bool(p_val < 0.05),
        "superior_model": "Model A" if mean_d < 0 else "Model B"
    }

def run_all_statistical_tests(y_true: np.ndarray, y_pred_ensemble: np.ndarray,
                              base_predictions: Dict[str, np.ndarray],
                              ensemble_name: str = "SE1") -> Dict[str, Any]:
    """
    Benchmark an ensemble against all baseline models across all statistical tests.
    """
    err_ensemble = np.abs(y_true - y_pred_ensemble)
    results = {}
    
    for base_name, pred_base in base_predictions.items():
        err_base = np.abs(y_true - pred_base)
        t_res = paired_student_ttest(err_ensemble, err_base)
        w_res = wilcoxon_signed_rank(err_ensemble, err_base)
        dm_res = diebold_mariano_test(y_true, y_pred_ensemble, pred_base, h=1, criterion="MSE")
        
        results[base_name] = {
            "paired_ttest": t_res,
            "wilcoxon_test": w_res,
            "diebold_mariano_test": dm_res
        }
        
    return results
