"""
Comprehensive Full Experiment Engine for AQI Forecasting with Strictly Prior-Only Features.
Executes:
1. Standalone Models: Historical Climatology, XGBoost, Extra Trees, Bi-LSTM, Bi-GRU, TCN
2. Stacking Ensembles: SE-1 (Meta-A) and SE-2 (Meta-B)
3. Multi-Seed Stability: Seeds [42, 101, 2024] reporting Mean ± SD
4. Architectural Ablation Study:
   - XGBoost alone
   - Extra Trees alone
   - XGBoost + Bi-LSTM
   - XGBoost + Bi-GRU
   - XGBoost + TCN
   - Full SE-1 (XGB + Bi-LSTM + TCN)
   - Full SE-2 (XGB + Bi-GRU + TCN)
   - Optimized Regularized Meta-Learner (RidgeCV selected on training OOF)
5. Statistical Significance Testing (paired t-test, Wilcoxon, Diebold-Mariano)
6. Extreme Pollution Spike Analysis (30-day shifted rolling z > 3.0)
7. Dynamic Time Warping (DTW) Trajectory Fidelity
8. TreeSHAP Explainability & Meta-Attribution
9. Computational Complexity Profiling
10. High-Quality Visual Figures
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import json
import pickle
import torch
torch.set_num_threads(1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from src.utils.logger import get_logger
from src.features.build_features import build_train_test_datasets
from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.statistical_tests import run_all_statistical_tests, paired_student_ttest
from src.analysis.spike_detection import detect_spikes_rolling_zscore, evaluate_spike_performance
from src.analysis.dtw_analysis import compare_models_dtw
from src.explainability.shap_analysis import compute_tabular_shap_xgboost, compute_meta_learner_shap, save_shap_plots

# Base Models
from src.models.baseline import HistoricalAnchorBaseline
from src.models.xgboost_model import XGBoostModel
from src.models.extra_trees_model import ExtraTreesModel
from src.models.bilstm_model import BiLSTMModel
from src.models.bigru_model import BiGRUModel
from src.models.tcn_model import TCNModel
from src.stacking.stacking_ensemble import StackingEnsemble

logger = get_logger("FullExperimentRunner", log_file="results/reports/full_experiment_execution.log")

def run_all_experiments():
    logger.info("=================================================================")
    logger.info("STARTING FULL STRICTLY PRIOR-ONLY AQI FORECASTING EXPERIMENTAL SUITE")
    logger.info("=================================================================")
    
    # 1. Load Dataset Bundle
    logger.info("Loading preprocessed dataset bundle (strictly prior-only features)...")
    with open("data/processed/dataset_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
        
    X_tab_train = bundle["X_tab_train"]
    X_tab_test = bundle["X_tab_test"]
    X_seq_train = bundle["X_seq_train"]
    X_seq_test = bundle["X_seq_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    dates_train = bundle["dates_train"]
    dates_test = bundle["dates_test"]
    tabular_cols = bundle["tabular_cols"]
    input_dim_seq = X_seq_train.shape[2]
    
    logger.info(f"Dataset summary:")
    logger.info(f"  Train: N={len(y_train)}, X_tab shape={X_tab_train.shape}, X_seq shape={X_seq_train.shape}")
    logger.info(f"  Test:  N={len(y_test)}, X_tab shape={X_tab_test.shape}, X_seq shape={X_seq_test.shape}")
    logger.info(f"  Tabular features: {len(tabular_cols)} (all unlagged pollutants/weather eliminated)")
    
    predictions = {}
    computational_stats = {}
    
    # -------------------------------------------------------------
    # 2. Standalone Model Executions (Seed 42 Baseline)
    # -------------------------------------------------------------
    # Model 1: Historical Climatology
    logger.info("\n--- 1. Evaluating Baseline: Historical Climatology ---")
    t0 = time.time()
    clim = HistoricalAnchorBaseline(window_days=7)
    clim.fit(dates_train, y_train)
    infer_t0 = time.time()
    p_clim, _, _ = clim.predict(dates_test)
    infer_time_clim = time.time() - infer_t0
    predictions["Historical Climatology"] = p_clim
    computational_stats["Historical Climatology"] = {
        "train_time_sec": round(time.time() - t0, 3),
        "param_count": 0,
        "inference_latency_ms": round(infer_time_clim / len(y_test) * 1000, 3)
    }
    
    # Model 2: Standalone XGBoost
    logger.info("\n--- 2. Evaluating Baseline: Standalone XGBoost ---")
    t0 = time.time()
    xgb = XGBoostModel(params={
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.04,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": 2
    })
    xgb.fit(X_tab_train, y_train)
    train_time_xgb = time.time() - t0
    infer_t0 = time.time()
    p_xgb = xgb.predict(X_tab_test)
    infer_time_xgb = time.time() - infer_t0
    predictions["XGBoost"] = p_xgb
    Path("experiments/xgboost").mkdir(parents=True, exist_ok=True)
    xgb.save("experiments/xgboost/xgb_standalone.pkl")
    computational_stats["XGBoost"] = {
        "train_time_sec": round(train_time_xgb, 3),
        "param_count": len(xgb.model.get_booster().get_dump()),
        "inference_latency_ms": round(infer_time_xgb / len(y_test) * 1000, 3)
    }
    
    # Model 3: Standalone Extra Trees
    logger.info("\n--- 3. Evaluating Baseline: Standalone Extra Trees ---")
    t0 = time.time()
    et = ExtraTreesModel(params={
        "n_estimators": 100, "max_depth": 8, "min_samples_split": 5,
        "random_state": 42, "n_jobs": 2
    })
    et.fit(X_tab_train, y_train)
    train_time_et = time.time() - t0
    infer_t0 = time.time()
    p_et = et.predict(X_tab_test)
    infer_time_et = time.time() - infer_t0
    predictions["Extra Trees"] = p_et
    Path("experiments/baseline").mkdir(parents=True, exist_ok=True)
    et.save("experiments/baseline/extra_trees_standalone.pkl")
    computational_stats["Extra Trees"] = {
        "train_time_sec": round(train_time_et, 3),
        "param_count": len(et.model.estimators_),
        "inference_latency_ms": round(infer_time_et / len(y_test) * 1000, 3)
    }
    
    # Model 4: Standalone Bi-LSTM
    logger.info("\n--- 4. Evaluating Baseline: Standalone Bi-LSTM ---")
    t0 = time.time()
    torch.manual_seed(42)
    bilstm = BiLSTMModel(input_dim=input_dim_seq, params={
        "hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
        "learning_rate": 0.001, "batch_size": 32, "epochs": 35, "patience": 6
    })
    val_split = int(len(y_train) * 0.90)
    bilstm.fit(X_seq_train[:val_split], y_train[:val_split],
               X_val=X_seq_train[val_split:], y_val=y_train[val_split:], verbose=False)
    train_time_bilstm = time.time() - t0
    infer_t0 = time.time()
    p_bilstm = bilstm.predict(X_seq_test)
    infer_time_bilstm = time.time() - infer_t0
    predictions["Bi-LSTM"] = p_bilstm
    Path("experiments/bilstm").mkdir(parents=True, exist_ok=True)
    bilstm.save("experiments/bilstm/bilstm_standalone.pt")
    n_params_bilstm = sum(p.numel() for p in bilstm.net.parameters() if p.requires_grad)
    computational_stats["Bi-LSTM"] = {
        "train_time_sec": round(train_time_bilstm, 3),
        "param_count": n_params_bilstm,
        "inference_latency_ms": round(infer_time_bilstm / len(y_test) * 1000, 3)
    }

    # Model 5: Standalone Bi-GRU
    logger.info("\n--- 5. Evaluating Baseline: Standalone Bi-GRU ---")
    t0 = time.time()
    torch.manual_seed(42)
    bigru = BiGRUModel(input_dim=input_dim_seq, params={
        "hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
        "learning_rate": 0.001, "batch_size": 32, "epochs": 35, "patience": 6
    })
    bigru.fit(X_seq_train[:val_split], y_train[:val_split],
              X_val=X_seq_train[val_split:], y_val=y_train[val_split:], verbose=False)
    train_time_bigru = time.time() - t0
    infer_t0 = time.time()
    p_bigru = bigru.predict(X_seq_test)
    infer_time_bigru = time.time() - infer_t0
    predictions["Bi-GRU"] = p_bigru
    Path("experiments/bigru").mkdir(parents=True, exist_ok=True)
    bigru.save("experiments/bigru/bigru_standalone.pt")
    n_params_bigru = sum(p.numel() for p in bigru.net.parameters() if p.requires_grad)
    computational_stats["Bi-GRU"] = {
        "train_time_sec": round(train_time_bigru, 3),
        "param_count": n_params_bigru,
        "inference_latency_ms": round(infer_time_bigru / len(y_test) * 1000, 3)
    }

    # Model 6: Standalone TCN
    logger.info("\n--- 6. Evaluating Baseline: Standalone TCN ---")
    t0 = time.time()
    torch.manual_seed(42)
    tcn = TCNModel(input_dim=input_dim_seq, params={
        "num_channels": [32, 32, 64, 64], "kernel_size": 3, "dropout": 0.2,
        "learning_rate": 0.001, "batch_size": 32, "epochs": 35, "patience": 6
    })
    tcn.fit(X_seq_train[:val_split], y_train[:val_split],
            X_val=X_seq_train[val_split:], y_val=y_train[val_split:], verbose=False)
    train_time_tcn = time.time() - t0
    infer_t0 = time.time()
    p_tcn = tcn.predict(X_seq_test)
    infer_time_tcn = time.time() - infer_t0
    predictions["TCN"] = p_tcn
    Path("experiments/tcn").mkdir(parents=True, exist_ok=True)
    tcn.save("experiments/tcn/tcn_standalone.pt")
    n_params_tcn = sum(p.numel() for p in tcn.net.parameters() if p.requires_grad)
    computational_stats["TCN"] = {
        "train_time_sec": round(train_time_tcn, 3),
        "param_count": n_params_tcn,
        "inference_latency_ms": round(infer_time_tcn / len(y_test) * 1000, 3)
    }

    # Model 7: Full SE-1 (Meta-A: XGBoost + Bi-LSTM + TCN -> Extra Trees)
    logger.info("\n--- 7. Training Full Stacking Ensemble SE-1 (Meta-A) ---")
    t0 = time.time()
    se1 = StackingEnsemble(ensemble_type="SE1", meta_learner_type="ExtraTrees",
                           input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    se1.fit_oof_and_meta(X_tab_train, X_seq_train, y_train, verbose=False)
    train_time_se1 = time.time() - t0
    infer_t0 = time.time()
    p_se1, _ = se1.predict(X_tab_test, X_seq_test)
    infer_time_se1 = time.time() - infer_t0
    predictions["SE-1"] = p_se1
    Path("experiments/se1").mkdir(parents=True, exist_ok=True)
    se1.save("experiments/se1")
    computational_stats["SE-1"] = {
        "train_time_sec": round(train_time_se1, 3),
        "param_count": n_params_bilstm + n_params_tcn + 100,
        "inference_latency_ms": round(infer_time_se1 / len(y_test) * 1000, 3)
    }

    # Model 8: Full SE-2 (Meta-B: XGBoost + Bi-GRU + TCN -> Extra Trees)
    logger.info("\n--- 8. Training Full Stacking Ensemble SE-2 (Meta-B) ---")
    t0 = time.time()
    se2 = StackingEnsemble(ensemble_type="SE2", meta_learner_type="ExtraTrees",
                           input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    se2.fit_oof_and_meta(X_tab_train, X_seq_train, y_train, verbose=False)
    train_time_se2 = time.time() - t0
    infer_t0 = time.time()
    p_se2, _ = se2.predict(X_tab_test, X_seq_test)
    infer_time_se2 = time.time() - infer_t0
    predictions["SE-2"] = p_se2
    Path("experiments/se2").mkdir(parents=True, exist_ok=True)
    se2.save("experiments/se2")
    computational_stats["SE-2"] = {
        "train_time_sec": round(train_time_se2, 3),
        "param_count": n_params_bigru + n_params_tcn + 100,
        "inference_latency_ms": round(infer_time_se2 / len(y_test) * 1000, 3)
    }

    # -------------------------------------------------------------
    # 3. Compute and Save Main Benchmark Metrics
    # -------------------------------------------------------------
    logger.info("\n--- Consolidating Primary Benchmark Comparison ---")
    metrics_records = []
    df_preds = pd.DataFrame({"Date": dates_test, "Actual_AQI": y_test})
    for m_name, pred_arr in predictions.items():
        df_preds[f"Pred_{m_name}"] = pred_arr
        m = calculate_all_metrics(y_test, pred_arr)
        m["Model"] = m_name
        metrics_records.append(m)
        
    df_metrics = pd.DataFrame(metrics_records)[["Model", "RMSE", "MAE", "MAPE", "sMAPE", "R2"]]
    logger.info("\n" + df_metrics.to_string(index=False))
    
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    Path("results/reports").mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv("results/metrics/model_comparison.csv", index=False)
    df_metrics.to_json("results/metrics/model_comparison.json", orient="records", indent=2)
    df_preds.to_csv("results/predictions/test_predictions.csv", index=False)
    
    with open("results/reports/computational_complexity.json", "w") as f:
        json.dump(computational_stats, f, indent=2)

    # -------------------------------------------------------------
    # 4. Architectural Ablation Study
    # -------------------------------------------------------------
    logger.info("\n=================================================================")
    logger.info("--- 9. Executing Architectural Ablation Study ---")
    logger.info("=================================================================")
    ablation_records = []
    
    # 1. XGBoost alone
    m_xgb = calculate_all_metrics(y_test, predictions["XGBoost"])
    m_xgb["Architecture"] = "XGBoost (Standalone)"
    m_xgb["Components"] = "XGBoost"
    m_xgb["Meta_Learner"] = "None"
    ablation_records.append(m_xgb)
    
    # 2. Extra Trees alone
    m_et = calculate_all_metrics(y_test, predictions["Extra Trees"])
    m_et["Architecture"] = "Extra Trees (Standalone)"
    m_et["Components"] = "Extra Trees"
    m_et["Meta_Learner"] = "None"
    ablation_records.append(m_et)

    # 3. Pairwise Stacking: XGBoost + Bi-LSTM -> Extra Trees
    logger.info("Ablation 1: XGBoost + Bi-LSTM -> Extra Trees...")
    ens_xgb_lstm = StackingEnsemble(components=["XGBoost", "Bi-LSTM"], meta_learner_type="ExtraTrees",
                                    input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    ens_xgb_lstm.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
    p_xgb_lstm, _ = ens_xgb_lstm.predict(X_tab_test, X_seq_test)
    m_xl = calculate_all_metrics(y_test, p_xgb_lstm)
    m_xl["Architecture"] = "XGBoost + Bi-LSTM (Pairwise)"
    m_xl["Components"] = "XGBoost, Bi-LSTM"
    m_xl["Meta_Learner"] = "Extra Trees"
    ablation_records.append(m_xl)

    # 4. Pairwise Stacking: XGBoost + Bi-GRU -> Extra Trees
    logger.info("Ablation 2: XGBoost + Bi-GRU -> Extra Trees...")
    ens_xgb_gru = StackingEnsemble(components=["XGBoost", "Bi-GRU"], meta_learner_type="ExtraTrees",
                                   input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    ens_xgb_gru.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
    p_xgb_gru, _ = ens_xgb_gru.predict(X_tab_test, X_seq_test)
    m_xg = calculate_all_metrics(y_test, p_xgb_gru)
    m_xg["Architecture"] = "XGBoost + Bi-GRU (Pairwise)"
    m_xg["Components"] = "XGBoost, Bi-GRU"
    m_xg["Meta_Learner"] = "Extra Trees"
    ablation_records.append(m_xg)

    # 5. Pairwise Stacking: XGBoost + TCN -> Extra Trees
    logger.info("Ablation 3: XGBoost + TCN -> Extra Trees...")
    ens_xgb_tcn = StackingEnsemble(components=["XGBoost", "TCN"], meta_learner_type="ExtraTrees",
                                   input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    ens_xgb_tcn.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
    p_xgb_tcn, _ = ens_xgb_tcn.predict(X_tab_test, X_seq_test)
    m_xt = calculate_all_metrics(y_test, p_xgb_tcn)
    m_xt["Architecture"] = "XGBoost + TCN (Pairwise)"
    m_xt["Components"] = "XGBoost, TCN"
    m_xt["Meta_Learner"] = "Extra Trees"
    ablation_records.append(m_xt)

    # 6. Full SE-1 (Meta-A: XGBoost + Bi-LSTM + TCN -> Extra Trees)
    m_se1 = calculate_all_metrics(y_test, predictions["SE-1"])
    m_se1["Architecture"] = "SE-1 (Meta-A: Full Triplet)"
    m_se1["Components"] = "XGBoost, Bi-LSTM, TCN"
    m_se1["Meta_Learner"] = "Extra Trees"
    ablation_records.append(m_se1)

    # 7. Full SE-2 (Meta-B: XGBoost + Bi-GRU + TCN -> Extra Trees)
    m_se2 = calculate_all_metrics(y_test, predictions["SE-2"])
    m_se2["Architecture"] = "SE-2 (Meta-B: Full Triplet)"
    m_se2["Components"] = "XGBoost, Bi-GRU, TCN"
    m_se2["Meta_Learner"] = "Extra Trees"
    ablation_records.append(m_se2)

    # 8. Regularized Meta-Learner (RidgeCV selected on training OOF) for SE-1
    logger.info("Ablation 4: SE-1 with Regularized Linear Meta-Learner (Ridge)...")
    ens_se1_ridge = StackingEnsemble(ensemble_type="SE1", meta_learner_type="Ridge",
                                     input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    ens_se1_ridge.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
    p_se1_ridge, _ = ens_se1_ridge.predict(X_tab_test, X_seq_test)
    m_s1r = calculate_all_metrics(y_test, p_se1_ridge)
    m_s1r["Architecture"] = "SE-1 (Ridge Meta-Learner)"
    m_s1r["Components"] = "XGBoost, Bi-LSTM, TCN"
    m_s1r["Meta_Learner"] = "RidgeCV"
    ablation_records.append(m_s1r)

    # 9. Regularized Meta-Learner (RidgeCV selected on training OOF) for SE-2
    logger.info("Ablation 5: SE-2 with Regularized Linear Meta-Learner (Ridge)...")
    ens_se2_ridge = StackingEnsemble(ensemble_type="SE2", meta_learner_type="Ridge",
                                     input_dim_seq=input_dim_seq, n_splits=5, random_seed=42, epochs=30)
    ens_se2_ridge.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
    p_se2_ridge, _ = ens_se2_ridge.predict(X_tab_test, X_seq_test)
    m_s2r = calculate_all_metrics(y_test, p_se2_ridge)
    m_s2r["Architecture"] = "SE-2 (Ridge Meta-Learner)"
    m_s2r["Components"] = "XGBoost, Bi-GRU, TCN"
    m_s2r["Meta_Learner"] = "RidgeCV"
    ablation_records.append(m_s2r)

    df_ablation = pd.DataFrame(ablation_records)[["Architecture", "Components", "Meta_Learner", "RMSE", "MAE", "MAPE", "sMAPE", "R2"]]
    logger.info("\nArchitectural Ablation Study Table:")
    logger.info("\n" + df_ablation.to_string(index=False))
    df_ablation.to_csv("results/metrics/ablation_study.csv", index=False)
    df_ablation.to_json("results/metrics/ablation_study.json", orient="records", indent=2)

    # -------------------------------------------------------------
    # 5. Multi-Seed Stability Analysis (Seeds: 42, 101, 2024)
    # -------------------------------------------------------------
    logger.info("\n=================================================================")
    logger.info("--- 10. Multi-Seed Stability Testing (Seeds: 42, 101, 2024) ---")
    logger.info("=================================================================")
    seeds = [42, 101, 2024]
    models_to_test = ["XGBoost", "Bi-LSTM", "Bi-GRU", "TCN", "SE-1", "SE-2"]
    stability_summary = []
    
    for m_label in models_to_test:
        seed_rmse, seed_mae, seed_mape, seed_r2 = [], [], [], []
        for s in seeds:
            if m_label == "XGBoost":
                m_obj = XGBoostModel(params={"n_estimators": 200, "max_depth": 5, "learning_rate": 0.04,
                                             "subsample": 0.8, "colsample_bytree": 0.8, "random_state": s, "n_jobs": 2})
                m_obj.fit(X_tab_train, y_train)
                p = m_obj.predict(X_tab_test)
            elif m_label == "Bi-LSTM":
                torch.manual_seed(s)
                m_obj = BiLSTMModel(input_dim=input_dim_seq, params={"hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
                                                                     "learning_rate": 0.001, "batch_size": 32, "epochs": 35, "patience": 6})
                m_obj.fit(X_seq_train[:val_split], y_train[:val_split], X_val=X_seq_train[val_split:], y_val=y_train[val_split:], verbose=False)
                p = m_obj.predict(X_seq_test)
            elif m_label == "Bi-GRU":
                torch.manual_seed(s)
                m_obj = BiGRUModel(input_dim=input_dim_seq, params={"hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
                                                                    "learning_rate": 0.001, "batch_size": 32, "epochs": 35, "patience": 6})
                m_obj.fit(X_seq_train[:val_split], y_train[:val_split], X_val=X_seq_train[val_split:], y_val=y_train[val_split:], verbose=False)
                p = m_obj.predict(X_seq_test)
            elif m_label == "TCN":
                torch.manual_seed(s)
                m_obj = TCNModel(input_dim=input_dim_seq, params={"num_channels": [32, 32, 64, 64], "kernel_size": 3, "dropout": 0.2,
                                                                 "learning_rate": 0.001, "batch_size": 32, "epochs": 35, "patience": 6})
                m_obj.fit(X_seq_train[:val_split], y_train[:val_split], X_val=X_seq_train[val_split:], y_val=y_train[val_split:], verbose=False)
                p = m_obj.predict(X_seq_test)
            elif m_label == "SE-1":
                ens = StackingEnsemble(ensemble_type="SE1", meta_learner_type="ExtraTrees",
                                       input_dim_seq=input_dim_seq, n_splits=5, random_seed=s, epochs=30)
                ens.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
                p, _ = ens.predict(X_tab_test, X_seq_test)
            elif m_label == "SE-2":
                ens = StackingEnsemble(ensemble_type="SE2", meta_learner_type="ExtraTrees",
                                       input_dim_seq=input_dim_seq, n_splits=5, random_seed=s, epochs=30)
                ens.fit_oof_and_meta(X_tab_train, X_seq_train, y_train)
                p, _ = ens.predict(X_tab_test, X_seq_test)
                
            met = calculate_all_metrics(y_test, p)
            seed_rmse.append(met["RMSE"])
            seed_mae.append(met["MAE"])
            seed_mape.append(met["MAPE"])
            seed_r2.append(met["R2"])
            
        stability_summary.append({
            "Model": m_label,
            "RMSE Mean ± SD": f"{np.mean(seed_rmse):.2f} ± {np.std(seed_rmse):.2f}",
            "MAE Mean ± SD": f"{np.mean(seed_mae):.2f} ± {np.std(seed_mae):.2f}",
            "MAPE Mean ± SD": f"{np.mean(seed_mape):.2f}% ± {np.std(seed_mape):.2f}%",
            "R2 Mean ± SD": f"{np.mean(seed_r2):.4f} ± {np.std(seed_r2):.4f}"
        })
        
    df_stability = pd.DataFrame(stability_summary)
    logger.info("\nMulti-Seed Stability Summary:")
    logger.info("\n" + df_stability.to_string(index=False))
    df_stability.to_csv("results/metrics/repeated_runs_stability.csv", index=False)

    # -------------------------------------------------------------
    # 6. Statistical Significance Tests
    # -------------------------------------------------------------
    logger.info("\n--- 11. Computing Statistical Significance Tests ---")
    stat_dir = Path("results/statistical_tests")
    stat_dir.mkdir(parents=True, exist_ok=True)
    
    base_preds_for_tests = {
        "XGBoost": predictions["XGBoost"],
        "Bi-LSTM": predictions["Bi-LSTM"],
        "Bi-GRU": predictions["Bi-GRU"],
        "TCN": predictions["TCN"],
        "Extra Trees": predictions["Extra Trees"]
    }
    stat_res_se1 = run_all_statistical_tests(y_test, predictions["SE-1"], base_preds_for_tests, ensemble_name="SE-1")
    stat_res_se2 = run_all_statistical_tests(y_test, predictions["SE-2"], base_preds_for_tests, ensemble_name="SE-2")
    se1_vs_se2 = paired_student_ttest(np.abs(y_test - predictions["SE-1"]), np.abs(y_test - predictions["SE-2"]))
    
    combined_stats = {
        "SE-1_vs_Baselines": stat_res_se1,
        "SE-2_vs_Baselines": stat_res_se2,
        "SE-1_vs_SE-2": se1_vs_se2
    }
    with open(stat_dir / "statistical_tests_results.json", "w") as f:
        json.dump(combined_stats, f, indent=2)
        
    ttest_rows = []
    for b_name in ["XGBoost", "Bi-LSTM", "Bi-GRU", "TCN", "Extra Trees"]:
        ttest_rows.append({
            "Model": b_name,
            "SE1 t-stat": stat_res_se1[b_name]["paired_ttest"]["t_statistic"],
            "SE1 p-value": "<0.001" if stat_res_se1[b_name]["paired_ttest"]["p_value"] < 0.001 else f"{stat_res_se1[b_name]['paired_ttest']['p_value']:.4f}",
            "SE2 t-stat": stat_res_se2[b_name]["paired_ttest"]["t_statistic"],
            "SE2 p-value": "<0.001" if stat_res_se2[b_name]["paired_ttest"]["p_value"] < 0.001 else f"{stat_res_se2[b_name]['paired_ttest']['p_value']:.4f}"
        })
    df_ttest = pd.DataFrame(ttest_rows)
    df_ttest.to_csv(stat_dir / "ttest_summary_table.csv", index=False)
    logger.info("\nPaired t-test Summary Table:\n" + df_ttest.to_string(index=False))

    # -------------------------------------------------------------
    # 7. Pollution Spike Detection & Extreme Regime Analysis (z > 3.0)
    # -------------------------------------------------------------
    logger.info("\n--- 12. Pollution Spike Analysis (30-day shifted rolling z > 3.0) ---")
    s_full = pd.Series(y_test)
    is_spike, roll_mean, z_scores = detect_spikes_rolling_zscore(s_full, window=30, threshold=3.0)
    spike_eval = evaluate_spike_performance(y_test, predictions, is_spike.values)
    with open("results/reports/spike_analysis.json", "w") as f:
        json.dump(spike_eval, f, indent=2)
    logger.info(f"Spike evaluation saved. Spike count: {spike_eval['spike_count']} / {spike_eval['total_count']}")

    # -------------------------------------------------------------
    # 8. Dynamic Time Warping (DTW) Trajectory Fidelity
    # -------------------------------------------------------------
    logger.info("\n--- 13. Dynamic Time Warping (DTW) Trajectory Analysis ---")
    dtw_results = compare_models_dtw(y_test, predictions, window_size=30)
    with open("results/reports/dtw_analysis.json", "w") as f:
        json.dump(dtw_results, f, indent=2)
    logger.info("\nDTW Results:\n" + pd.DataFrame(dtw_results).T.to_string())

    # -------------------------------------------------------------
    # 9. TreeSHAP Explainability & Meta Attribution
    # -------------------------------------------------------------
    logger.info("\n--- 14. TreeSHAP Feature Attribution ---")
    Path("results/shap").mkdir(parents=True, exist_ok=True)
    shap_results_xgb = compute_tabular_shap_xgboost(
        xgb.model, X_tab_train, X_tab_test, tabular_cols
    )
    save_shap_plots(shap_results_xgb["shap_values"], X_tab_test, tabular_cols,
                    output_dir="results/shap", prefix="xgboost_tabular")
    shap_results_xgb["importance_df"].to_csv("results/shap/xgboost_feature_importance.csv", index=False)
    
    # Meta Learner SHAP for SE-1
    Z_test_se1 = np.column_stack([predictions["XGBoost"], predictions["Bi-LSTM"], predictions["TCN"]])
    shap_results_meta_se1 = compute_meta_learner_shap(
        se1.meta_learner.model, Z_test_se1, ["XGBoost", "Bi-LSTM", "TCN"]
    )
    shap_results_meta_se1["importance_df"].to_csv("results/shap/se1_meta_feature_importance.csv", index=False)
    
    # Meta Learner SHAP for SE-2
    Z_test_se2 = np.column_stack([predictions["XGBoost"], predictions["Bi-GRU"], predictions["TCN"]])
    shap_results_meta_se2 = compute_meta_learner_shap(
        se2.meta_learner.model, Z_test_se2, ["XGBoost", "Bi-GRU", "TCN"]
    )
    shap_results_meta_se2["importance_df"].to_csv("results/shap/se2_meta_feature_importance.csv", index=False)
    logger.info("SE-1 Meta SHAP:\n" + shap_results_meta_se1["importance_df"].to_string(index=False))
    logger.info("SE-2 Meta SHAP:\n" + shap_results_meta_se2["importance_df"].to_string(index=False))

    # -------------------------------------------------------------
    # 10. Publication-Quality Visual Figures
    # -------------------------------------------------------------
    logger.info("\n--- 15. Generating Publication Visual Figures ---")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Fig 1: Actual vs Predicted Time Series
    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, y_test, label="Actual AQI (Ground Truth)", color="black", linewidth=2.0, alpha=0.9)
    plt.plot(dates_test, predictions["XGBoost"], label="XGBoost (Prior Only)", color="#2ca02c", linestyle="--", linewidth=1.5)
    plt.plot(dates_test, predictions["SE-1"], label="SE-1 Ensemble", color="#d62728", linewidth=1.8)
    plt.plot(dates_test, predictions["SE-2"], label="SE-2 Ensemble", color="#1f77b4", linewidth=1.8)
    plt.axhline(200, color="orange", linestyle=":", label="Poor (201-300)")
    plt.axhline(300, color="red", linestyle=":", label="Very Poor (301-400)")
    plt.title("AQI Forecasting: Actual vs Predicted Time Series (Strict Prior Features)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("AQI Value", fontsize=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "actual_vs_predicted_timeseries.png", dpi=300)
    plt.close()
    
    # Fig 2: Residual Distributions
    plt.figure(figsize=(12, 6))
    for m_name, col in [("XGBoost", "#2ca02c"), ("Bi-LSTM", "#ff7f0e"), ("SE-1", "#d62728"), ("SE-2", "#1f77b4")]:
        res = y_test - predictions[m_name]
        sns.kdeplot(res, label=f"{m_name} (std={np.std(res):.1f})", color=col, linewidth=2.0)
    plt.axvline(0, color="black", linestyle="--", alpha=0.6)
    plt.title("Forecast Residual Error Distribution (y - ŷ)", fontsize=14, fontweight="bold")
    plt.xlabel("Prediction Error (AQI Points)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(-120, 120)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_distributions.png", dpi=300)
    plt.close()
    
    # Fig 3: Model Metrics Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    df_plot = df_metrics[df_metrics["Model"] != "Historical Climatology"].copy()
    x = np.arange(len(df_plot))
    width = 0.35
    plt.bar(x - width/2, df_plot["RMSE"], width, label="RMSE", color="#e74c3c")
    plt.bar(x + width/2, df_plot["MAE"], width, label="MAE", color="#3498db")
    plt.xticks(x, df_plot["Model"], rotation=30, ha="right", fontsize=11)
    plt.ylabel("Error (AQI Points)", fontsize=12)
    plt.title("Model Error Comparison (Strictly Prior Features)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "model_metrics_comparison.png", dpi=300)
    plt.close()
    
    # Fig 4: Extreme Pollution Regimes (Spike evaluation)
    plt.figure(figsize=(10, 6))
    spike_rmse = [spike_eval["models"][m]["spike_period"]["RMSE"] for m in df_plot["Model"]]
    normal_rmse = [spike_eval["models"][m]["overall"]["RMSE"] for m in df_plot["Model"]]
    plt.bar(x - width/2, normal_rmse, width, label="Overall RMSE", color="#95a5a6")
    plt.bar(x + width/2, spike_rmse, width, label="Spike Period RMSE (z > 3)", color="#c0392b")
    plt.xticks(x, df_plot["Model"], rotation=30, ha="right", fontsize=11)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("Error Inflation During Extreme Pollution Spikes (z > 3.0)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "high_pollution_regimes.png", dpi=300)
    plt.close()
    
    logger.info("All figures generated successfully.")
    logger.info("=================================================================")
    logger.info("FULL EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY!")
    logger.info("=================================================================")

if __name__ == "__main__":
    run_all_experiments()
