"""
Stacked Ensemble Pipeline for SE-1 (Bi-LSTM), SE-2 (Bi-GRU), and Architectural Ablations.
Implements strictly leakage-free Out-Of-Fold (OOF) TimeSeriesSplit cross-validation
to train the meta-learner.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import copy
import joblib
import torch
torch.set_num_threads(1)
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

from src.utils.logger import get_logger
from src.models.xgboost_model import XGBoostModel
from src.models.bilstm_model import BiLSTMModel
from src.models.bigru_model import BiGRUModel
from src.models.tcn_model import TCNModel
from src.models.extra_trees_model import ExtraTreesModel

logger = get_logger("StackingEnsemble")

class StackingEnsemble:
    def __init__(self, ensemble_type: str = "SE1",
                 components: Optional[List[str]] = None,
                 meta_learner_type: str = "ExtraTrees",
                 input_dim_seq: int = 10,
                 n_splits: int = 5,
                 random_seed: int = 42,
                 epochs: int = 30):
        """
        ensemble_type: 'SE1' (XGBoost + Bi-LSTM + TCN),
                       'SE2' (XGBoost + Bi-GRU + TCN), or 'Custom'
        components: list of model names, e.g. ['XGBoost', 'Bi-LSTM', 'TCN']
        meta_learner_type: 'ExtraTrees', 'Ridge', 'ElasticNet', or 'Auto'
        """
        self.ensemble_type = ensemble_type
        self.input_dim_seq = input_dim_seq
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.epochs = epochs
        self.meta_learner_type = meta_learner_type
        
        # Configure components
        if components is not None:
            self.components = components
        elif ensemble_type == "SE1":
            self.components = ["XGBoost", "Bi-LSTM", "TCN"]
        elif ensemble_type == "SE2":
            self.components = ["XGBoost", "Bi-GRU", "TCN"]
        else:
            raise ValueError(f"Unknown ensemble_type: {ensemble_type}")
            
        self.meta_learner = None
        self.selected_meta_name = meta_learner_type
        self.base_models: Dict[str, Any] = {}
        self._init_base_models()
        self._init_meta_learner(meta_learner_type)
        
    def _init_base_models(self):
        """Initialize base models with deterministic seeds."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.base_models = {}
        
        if "XGBoost" in self.components:
            self.base_models["XGBoost"] = XGBoostModel(params={
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_seed,
                "n_jobs": 2
            })
            
        recurrent_params = {
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": self.epochs,
            "patience": 6
        }
        
        if "Bi-LSTM" in self.components:
            torch.manual_seed(self.random_seed)
            self.base_models["Bi-LSTM"] = BiLSTMModel(input_dim=self.input_dim_seq, params=recurrent_params)
            
        if "Bi-GRU" in self.components:
            torch.manual_seed(self.random_seed)
            self.base_models["Bi-GRU"] = BiGRUModel(input_dim=self.input_dim_seq, params=recurrent_params)
            
        if "TCN" in self.components:
            tcn_params = {
                "num_channels": [32, 32, 64, 64],
                "kernel_size": 3,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": self.epochs,
                "patience": 6
            }
            torch.manual_seed(self.random_seed)
            self.base_models["TCN"] = TCNModel(input_dim=self.input_dim_seq, params=tcn_params)

    def _init_meta_learner(self, meta_type: str):
        """Instantiate meta-learner."""
        if meta_type in ["ExtraTrees", "Auto"]:
            self.meta_learner = ExtraTreesModel(params={
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 5,
                "random_state": self.random_seed,
                "n_jobs": 2
            })
        elif meta_type == "Ridge":
            self.meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 13))
        elif meta_type == "ElasticNet":
            self.meta_learner = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=3, random_state=self.random_seed)
        else:
            raise ValueError(f"Unknown meta_learner_type: {meta_type}")

    def fit_oof_and_meta(self, 
                          X_tab_train: np.ndarray, 
                          X_seq_train: np.ndarray, 
                          y_train: np.ndarray,
                          verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate unbiased out-of-fold predictions on training folds and train the meta-learner.
        Returns: (Z_oof, y_oof_aligned)
        """
        logger.info(f"Starting Out-Of-Fold Cross-Validation for {self.ensemble_type} ({self.components}) with {self.n_splits} temporal splits...")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        oof_dict = {c: [] for c in self.components}
        oof_targets = []
        
        fold = 1
        for train_idx, val_idx in tscv.split(X_tab_train):
            if verbose:
                logger.info(f"--- Fold {fold}/{self.n_splits}: Train size={len(train_idx)}, Val size={len(val_idx)} ---")
                
            f_X_tab_tr, f_y_tr = X_tab_train[train_idx], y_train[train_idx]
            f_X_seq_tr = X_seq_train[train_idx]
            
            f_X_tab_va, f_y_va = X_tab_train[val_idx], y_train[val_idx]
            f_X_seq_va = X_seq_train[val_idx]
            
            for c in self.components:
                if c == "XGBoost":
                    f_xgb = XGBoostModel(params={
                        "n_estimators": 150, "max_depth": 5, "learning_rate": 0.04,
                        "subsample": 0.8, "colsample_bytree": 0.8,
                        "random_state": self.random_seed + fold, "n_jobs": 2
                    })
                    f_xgb.fit(f_X_tab_tr, f_y_tr)
                    oof_dict["XGBoost"].extend(f_xgb.predict(f_X_tab_va).tolist())
                    
                elif c == "Bi-LSTM":
                    rec_params = {
                        "hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
                        "learning_rate": 0.001, "batch_size": 32,
                        "epochs": max(2, int(self.epochs * 0.75)), "patience": 5
                    }
                    torch.manual_seed(self.random_seed + fold)
                    f_rnn = BiLSTMModel(input_dim=self.input_dim_seq, params=rec_params)
                    f_rnn.fit(f_X_seq_tr, f_y_tr, X_val=f_X_seq_va, y_val=f_y_va, verbose=False)
                    oof_dict["Bi-LSTM"].extend(f_rnn.predict(f_X_seq_va).tolist())
                    
                elif c == "Bi-GRU":
                    rec_params = {
                        "hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
                        "learning_rate": 0.001, "batch_size": 32,
                        "epochs": max(2, int(self.epochs * 0.75)), "patience": 5
                    }
                    torch.manual_seed(self.random_seed + fold)
                    f_rnn = BiGRUModel(input_dim=self.input_dim_seq, params=rec_params)
                    f_rnn.fit(f_X_seq_tr, f_y_tr, X_val=f_X_seq_va, y_val=f_y_va, verbose=False)
                    oof_dict["Bi-GRU"].extend(f_rnn.predict(f_X_seq_va).tolist())
                    
                elif c == "TCN":
                    tcn_p = {
                        "num_channels": [32, 32, 64, 64], "kernel_size": 3, "dropout": 0.2,
                        "learning_rate": 0.001, "batch_size": 32,
                        "epochs": max(2, int(self.epochs * 0.75)), "patience": 5
                    }
                    torch.manual_seed(self.random_seed + fold)
                    f_tcn = TCNModel(input_dim=self.input_dim_seq, params=tcn_p)
                    f_tcn.fit(f_X_seq_tr, f_y_tr, X_val=f_X_seq_va, y_val=f_y_va, verbose=False)
                    oof_dict["TCN"].extend(f_tcn.predict(f_X_seq_va).tolist())
                    
            oof_targets.extend(f_y_va.tolist())
            fold += 1
            
        Z_oof = np.column_stack([oof_dict[c] for c in self.components])
        y_oof = np.array(oof_targets, dtype=np.float32)
        
        # Meta-learner selection / fitting (Strictly on Training OOF data)
        if self.meta_learner_type == "Auto":
            logger.info("Evaluating candidate meta-learners strictly on training OOF cross-validation...")
            candidates = {
                "ExtraTrees": ExtraTreesModel(params={"n_estimators": 100, "max_depth": 5, "min_samples_split": 5, "random_state": self.random_seed, "n_jobs": 2}),
                "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
                "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, 1.0], cv=3, random_state=self.random_seed)
            }
            best_name = "ExtraTrees"
            best_rmse = float("inf")
            
            # 3-fold temporal validation on OOF predictions
            oof_tscv = TimeSeriesSplit(n_splits=3)
            for cand_name, cand_model in candidates.items():
                val_losses = []
                for tr_i, val_i in oof_tscv.split(Z_oof):
                    c_fit = copy.deepcopy(cand_model)
                    c_fit.fit(Z_oof[tr_i], y_oof[tr_i])
                    p_val = c_fit.predict(Z_oof[val_i])
                    val_losses.append(np.sqrt(mean_squared_error(y_oof[val_i], p_val)))
                mean_loss = float(np.mean(val_losses))
                logger.info(f"  Candidate {cand_name} OOF Validation RMSE: {mean_loss:.4f}")
                if mean_loss < best_rmse:
                    best_rmse = mean_loss
                    best_name = cand_name
                    
            logger.info(f"Selected Best Meta-Learner: {best_name} (OOF RMSE: {best_rmse:.4f})")
            self.selected_meta_name = best_name
            self._init_meta_learner(best_name)
            
        logger.info(f"OOF predictions shape {Z_oof.shape}. Training final meta-learner ({self.selected_meta_name})...")
        self.meta_learner.fit(Z_oof, y_oof)
        logger.info("Meta-learner trained successfully.")
        
        # Retrain base models on full training set with validation tail for early stopping
        val_split_point = int(len(y_train) * 0.90)
        X_seq_tr_main, y_tr_main = X_seq_train[:val_split_point], y_train[:val_split_point]
        X_seq_tr_val, y_tr_val = X_seq_train[val_split_point:], y_train[val_split_point:]
        
        logger.info(f"Retraining base models on full training set ({len(y_train)} samples)...")
        for c, m in self.base_models.items():
            if c == "XGBoost":
                m.fit(X_tab_train, y_train)
            elif c in ["Bi-LSTM", "Bi-GRU"]:
                torch.manual_seed(self.random_seed)
                m.fit(X_seq_tr_main, y_tr_main, X_val=X_seq_tr_val, y_val=y_tr_val, verbose=False)
            elif c == "TCN":
                torch.manual_seed(self.random_seed)
                m.fit(X_seq_tr_main, y_tr_main, X_val=X_seq_tr_val, y_val=y_tr_val, verbose=False)
                
        logger.info("All final base models retrained on full training set.")
        return Z_oof, y_oof

    def predict(self, X_tab: np.ndarray, X_seq: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate base predictions and final ensemble prediction.
        Returns: (final_pred, {model_name: pred})
        """
        base_preds = {}
        for c in self.components:
            m = self.base_models[c]
            if c == "XGBoost":
                base_preds[c] = m.predict(X_tab)
            else:
                base_preds[c] = m.predict(X_seq)
                
        Z = np.column_stack([base_preds[c] for c in self.components])
        final_pred = self.meta_learner.predict(Z)
        
        # Flatten if numpy 1D array needed
        if isinstance(final_pred, np.ndarray) and final_pred.ndim > 1:
            final_pred = final_pred.ravel()
            
        return final_pred, base_preds

    def save(self, dir_path: str):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        for c, m in self.base_models.items():
            if c == "XGBoost":
                m.save(str(p / "xgb_base.pkl"))
            elif c == "Bi-LSTM":
                m.save(str(p / "bilstm_base.pt"))
            elif c == "Bi-GRU":
                m.save(str(p / "bigru_base.pt"))
            elif c == "TCN":
                m.save(str(p / "tcn_base.pt"))
                
        if isinstance(self.meta_learner, ExtraTreesModel):
            self.meta_learner.save(str(p / "meta_learner.pkl"))
        else:
            joblib.dump(self.meta_learner, str(p / "meta_learner.pkl"))
            
        metadata = {
            "ensemble_type": self.ensemble_type,
            "components": self.components,
            "meta_learner_type": self.meta_learner_type,
            "selected_meta_name": self.selected_meta_name,
            "input_dim_seq": self.input_dim_seq,
            "random_seed": self.random_seed
        }
        joblib.dump(metadata, str(p / "metadata.pkl"))
        logger.info(f"Saved complete ensemble to {dir_path}")

    def load(self, dir_path: str):
        p = Path(dir_path)
        metadata = joblib.load(str(p / "metadata.pkl"))
        self.ensemble_type = metadata["ensemble_type"]
        self.components = metadata["components"]
        self.meta_learner_type = metadata["meta_learner_type"]
        self.selected_meta_name = metadata.get("selected_meta_name", self.meta_learner_type)
        self.input_dim_seq = metadata["input_dim_seq"]
        self.random_seed = metadata["random_seed"]
        self._init_base_models()
        
        for c, m in self.base_models.items():
            if c == "XGBoost":
                m.load(str(p / "xgb_base.pkl"))
            elif c == "Bi-LSTM":
                m.load(str(p / "bilstm_base.pt"))
            elif c == "Bi-GRU":
                m.load(str(p / "bigru_base.pt"))
            elif c == "TCN":
                m.load(str(p / "tcn_base.pt"))
                
        if self.selected_meta_name in ["ExtraTrees"]:
            self.meta_learner = ExtraTreesModel()
            self.meta_learner.load(str(p / "meta_learner.pkl"))
        else:
            self.meta_learner = joblib.load(str(p / "meta_learner.pkl"))
            
        logger.info(f"Loaded ensemble from {dir_path}")
        return self
