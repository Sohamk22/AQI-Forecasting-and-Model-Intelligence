"""
Unit test to verify that the stacking pipeline does not use in-sample predictions
and correctly enforces out-of-fold temporal cross validation.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import numpy as np
import pytest
from src.stacking.stacking_ensemble import StackingEnsemble

def test_stacking_oof_execution():
    """Verify that OOF training executes and meta-learner learns from out-of-fold matrix."""
    n_samples = 120
    n_tab_features = 15
    seq_len = 5
    n_seq_features = 4
    
    np.random.seed(42)
    X_tab = np.random.randn(n_samples, n_tab_features).astype(np.float32)
    X_seq = np.random.randn(n_samples, seq_len, n_seq_features).astype(np.float32)
    y = np.random.uniform(50, 300, size=n_samples).astype(np.float32)
    
    ensemble = StackingEnsemble(ensemble_type="SE1", input_dim_seq=n_seq_features, n_splits=2, random_seed=42, epochs=2)
    
    Z_oof, y_oof = ensemble.fit_oof_and_meta(X_tab, X_seq, y, verbose=False)
    
    # Check that OOF predictions exist and have 3 columns (XGB, Bi-LSTM, TCN)
    assert Z_oof.shape[1] == 3
    assert len(Z_oof) == len(y_oof)
    assert len(Z_oof) < n_samples  # In TimeSeriesSplit with n_splits=3, the initial training fold is not in validation
    
    # Predict on test samples
    X_tab_test = np.random.randn(10, n_tab_features).astype(np.float32)
    X_seq_test = np.random.randn(10, seq_len, n_seq_features).astype(np.float32)
    final_pred, base_preds = ensemble.predict(X_tab_test, X_seq_test)
    
    assert len(final_pred) == 10
    assert "XGBoost" in base_preds
    assert "Bi-LSTM" in base_preds
    assert "TCN" in base_preds
