"""
SHAP (SHapley Additive exPlanations) Explainability Pipeline.
Calculates:
1. Global feature importance (Mean Absolute SHAP values).
2. Base model contribution attribution in the Extra Trees meta-learner.
3. Summary bar plots and heatmap representations across test intervals.
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger("SHAPAnalysis")

def compute_tabular_shap_xgboost(xgb_model, X_train: np.ndarray, X_test: np.ndarray,
                                feature_names: List[str], max_display: int = 15) -> Dict[str, Any]:
    """
    Compute TreeSHAP explanations for the XGBoost model on the test partition.
    """
    logger.info("Computing TreeSHAP values for XGBoost...")
    import xgboost as xgb
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    contribs = xgb_model.get_booster().predict(dtest, pred_contribs=True)
    shap_values = contribs[:, :-1]
    expected_val = float(contribs[0, -1])
    
    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    logger.info(f"Top 5 most influential features in XGBoost: {importance_df.head(5)['feature'].tolist()}")
    return {
        "shap_values": shap_values,
        "importance_df": importance_df,
        "expected_value": expected_val
    }

def compute_meta_learner_shap(meta_model, Z_test: np.ndarray,
                              base_model_names: List[str]) -> Dict[str, Any]:
    """
    Compute TreeSHAP explanations for the Extra Trees meta-learner to quantify
    how much each base model contributes to the final ensemble prediction.
    """
    logger.info("Computing TreeSHAP values for Extra Trees meta-learner...")
    explainer = shap.TreeExplainer(meta_model)
    shap_values = explainer.shap_values(Z_test)
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "base_model": base_model_names,
        "mean_abs_shap": mean_abs_shap,
        "relative_weight_pct": (mean_abs_shap / np.sum(mean_abs_shap)) * 100.0
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    return {
        "shap_values": shap_values,
        "importance_df": importance_df,
        "expected_value": float(explainer.expected_value)
    }

def save_shap_plots(shap_values: np.ndarray, X_test: np.ndarray, feature_names: List[str],
                    output_dir: str = "results/shap", prefix: str = "xgboost"):
    """Generate and save SHAP summary bar chart and heatmap visualization."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False, max_display=12)
    plt.title(f"Mean Absolute SHAP Values ({prefix.upper()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_file = out_path / f"{prefix}_shap_bar.png"
    plt.savefig(bar_file, dpi=300)
    plt.close()
    
    # 2. Detailed Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=12)
    plt.title(f"SHAP Value Distribution ({prefix.upper()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    beeswarm_file = out_path / f"{prefix}_shap_beeswarm.png"
    plt.savefig(beeswarm_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved SHAP plots to {bar_file} and {beeswarm_file}")
