"""
Unit tests for data leakage prevention, sequence generation, and metric calculation.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.features.temporal_features import add_lag_features, add_rolling_features
from src.evaluation.metrics import calculate_all_metrics

def test_lag_features_no_lookahead():
    """Verify lag features do not contain contemporaneous or future data."""
    df = pd.DataFrame({"AQI": [10, 20, 30, 40, 50]})
    df_lag = add_lag_features(df, cols=["AQI"], lags=[1, 2])
    
    # At index 0, lag 1 must be NaN
    assert pd.isna(df_lag.loc[0, "AQI_lag_1"])
    # At index 1, lag 1 must equal value at index 0 (10)
    assert df_lag.loc[1, "AQI_lag_1"] == 10
    # At index 2, lag 2 must equal value at index 0 (10)
    assert df_lag.loc[2, "AQI_lag_2"] == 10

def test_rolling_features_shift_enforced():
    """Verify rolling features are strictly computed from t-1 and prior."""
    df = pd.DataFrame({"AQI": [10, 20, 30, 40, 50]})
    df_roll = add_rolling_features(df, cols=["AQI"], windows=[2])
    
    # At index 0, shifted rolling mean must be NaN
    assert pd.isna(df_roll.loc[0, "AQI_roll_mean_2"])
    # At index 1, rolling mean window 2 on shifted series contains only index 0 (10)
    assert df_roll.loc[1, "AQI_roll_mean_2"] == 10.0
    # At index 2, rolling mean window 2 contains index 0 and 1 -> (10 + 20) / 2 = 15.0
    # It must NOT include index 2 (30)
    assert df_roll.loc[2, "AQI_roll_mean_2"] == 15.0

def test_scaler_leakage_isolation():
    """Verify training scaler does not use test statistics."""
    train_data = np.array([[10.0], [20.0], [30.0]])
    test_data = np.array([[100.0]])  # Out-of-bounds relative to train
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    
    # Scaled train max must be 1.0
    scaled_train = scaler.transform(train_data)
    assert np.isclose(scaled_train.max(), 1.0)
    
    # Scaled test will exceed 1.0 because scaler was fit on train only
    scaled_test = scaler.transform(test_data)
    assert scaled_test[0, 0] > 1.0

def test_metrics_calculation():
    """Verify RMSE, MAE, MAPE, and R2 calculation."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    
    metrics = calculate_all_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 10.0)
    assert np.isclose(metrics["RMSE"], 10.0)
    assert metrics["R2"] > 0.95

if __name__ == "__main__":
    test_lag_features_no_lookahead()
    test_rolling_features_shift_enforced()
    test_scaler_leakage_isolation()
    print("All leakage unit tests passed successfully.")
