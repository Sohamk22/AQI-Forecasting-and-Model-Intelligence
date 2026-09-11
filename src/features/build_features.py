"""
Pipeline to build engineered features and sequence datasets for AQI forecasting.
Strict chronological splitting with zero data leakage.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger
from src.utils.config import DataConfig
from src.data.preprocess import build_clean_dataset, compute_iqr_bounds, apply_winsorization
from src.features.temporal_features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features
)

logger = get_logger("FeatureEngineering")

def create_feature_pipeline(df: pd.DataFrame, 
                            config: DataConfig = DataConfig()) -> Tuple[pd.DataFrame, list, list]:
    """Apply feature engineering to cleaned daily dataframe."""
    logger.info("Applying calendar and temporal feature engineering...")
    df_feat = add_calendar_features(df, date_col="Date")
    
    # Generate lag features for target and key pollutants/weather
    lag_cols = [config.target_col] + config.pollutant_cols + config.weather_cols
    df_feat = add_lag_features(df_feat, cols=lag_cols, lags=[1, 2, 3, 7])
    
    # Generate rolling window features strictly shifted
    rolling_cols = [config.target_col, "PM2.5", "PM10", "NO2", "Temperature", "Humidity", "Wind_Speed"]
    df_feat = add_rolling_features(df_feat, cols=rolling_cols, windows=[7, 14])
    
    # Identify sequence features (for RNN/TCN) and tabular features (for XGBoost)
    seq_feature_cols = config.pollutant_cols + config.weather_cols + [config.target_col]
    
    # Tabular feature columns: all engineered lag, roll, and calendar columns
    # Strictly exclude ALL raw unlagged columns from timestamp t (pollutants, weather, metadata, target)
    raw_unlagged_cols = set(df.columns)
    tabular_feature_cols = [c for c in df_feat.columns if c not in raw_unlagged_cols]
    
    logger.info(f"Total tabular features: {len(tabular_feature_cols)}")
    logger.info(f"Total sequence features: {len(seq_feature_cols)}")
    
    return df_feat, tabular_feature_cols, seq_feature_cols

def build_train_test_datasets(config: DataConfig = DataConfig()) -> Dict[str, Any]:
    """
    Split data chronologically (80:20), fit scalers strictly on train,
    and build tabular and 3D sequence datasets for both train and test.
    """
    cleaned_df = build_clean_dataset(config)
    df_feat, tabular_cols, seq_cols = create_feature_pipeline(cleaned_df, config)
    
    # Determine chronological split index
    total_len = len(df_feat)
    split_idx = int(total_len * config.train_test_split_ratio)
    
    df_train_raw = df_feat.iloc[:split_idx].copy().reset_index(drop=True)
    df_test_raw = df_feat.iloc[split_idx:].copy().reset_index(drop=True)
    
    logger.info(f"Chronological split at index {split_idx}:")
    logger.info(f"  Train: {len(df_train_raw)} days ({df_train_raw['Date'].min().strftime('%Y-%m-%d')} to {df_train_raw['Date'].max().strftime('%Y-%m-%d')})")
    logger.info(f"  Test:  {len(df_test_raw)} days ({df_test_raw['Date'].min().strftime('%Y-%m-%d')} to {df_test_raw['Date'].max().strftime('%Y-%m-%d')})")
    
    # Compute IQR outlier bounds on TRAIN ONLY
    iqr_cols = config.pollutant_cols + config.weather_cols + [config.target_col]
    iqr_bounds = compute_iqr_bounds(df_train_raw, iqr_cols)
    
    df_train, train_outliers = apply_winsorization(df_train_raw, iqr_bounds)
    df_test, test_outliers = apply_winsorization(df_test_raw, iqr_bounds)
    logger.info(f"Applied IQR Winsorization. Train outlier counts: {train_outliers}")
    
    # Fit Scalers strictly on TRAIN ONLY
    scaler_tabular = MinMaxScaler()
    scaler_tabular.fit(df_train[tabular_cols])
    
    scaler_seq = MinMaxScaler()
    scaler_seq.fit(df_train[seq_cols])
    
    scaler_target = MinMaxScaler()
    scaler_target.fit(df_train[[config.target_col]])
    
    # Transform
    X_tab_train_scaled = scaler_tabular.transform(df_train[tabular_cols])
    X_tab_test_scaled = scaler_tabular.transform(df_test[tabular_cols])
    
    X_seq_train_scaled = scaler_seq.transform(df_train[seq_cols])
    X_seq_test_scaled = scaler_seq.transform(df_test[seq_cols])
    
    y_train = df_train[config.target_col].values
    y_test = df_test[config.target_col].values
    
    # Construct 3D sequence tensors with sliding window W
    W = config.sequence_length
    
    def construct_sequences(seq_data: np.ndarray, tab_data: np.ndarray, targets: np.ndarray, dates: pd.Series):
        X_seq_list, X_tab_list, y_list, dates_list = [], [], [], []
        for i in range(W, len(seq_data)):
            # Sequence covers indices [i - W, i - 1] inclusive (strictly prior history)
            X_seq_list.append(seq_data[i - W:i])
            # Tabular features at day i (which were already computed with past shifts)
            X_tab_list.append(tab_data[i])
            # Target at day i
            y_list.append(targets[i])
            dates_list.append(dates.iloc[i])
            
        return (np.array(X_seq_list, dtype=np.float32),
                np.array(X_tab_list, dtype=np.float32),
                np.array(y_list, dtype=np.float32),
                np.array(dates_list))
    
    X_seq_train, X_tab_train, y_train_seq, dates_train = construct_sequences(
        X_seq_train_scaled, X_tab_train_scaled, y_train, df_train["Date"]
    )
    
    # For test sequences, we concatenate the last W days of train to allow full test sequence creation from index 0
    seq_test_with_context = np.vstack([X_seq_train_scaled[-W:], X_seq_test_scaled])
    tab_test_with_context = np.vstack([X_tab_train_scaled[-W:], X_tab_test_scaled])
    y_test_with_context = np.concatenate([y_train[-W:], y_test])
    dates_test_with_context = pd.concat([df_train["Date"].iloc[-W:], df_test["Date"]], ignore_index=True)
    
    X_seq_test, X_tab_test, y_test_seq, dates_test = construct_sequences(
        seq_test_with_context, tab_test_with_context, y_test_with_context, dates_test_with_context
    )
    
    logger.info(f"Final dataset dimensions:")
    logger.info(f"  Train: X_seq={X_seq_train.shape}, X_tab={X_tab_train.shape}, y={y_train_seq.shape}")
    logger.info(f"  Test:  X_seq={X_seq_test.shape}, X_tab={X_tab_test.shape}, y={y_test_seq.shape}")
    
    dataset_bundle = {
        "X_seq_train": X_seq_train,
        "X_tab_train": X_tab_train,
        "y_train": y_train_seq,
        "dates_train": dates_train,
        "X_seq_test": X_seq_test,
        "X_tab_test": X_tab_test,
        "y_test": y_test_seq,
        "dates_test": dates_test,
        "tabular_cols": tabular_cols,
        "seq_cols": seq_cols,
        "scaler_tabular": scaler_tabular,
        "scaler_seq": scaler_seq,
        "scaler_target": scaler_target,
        "iqr_bounds": iqr_bounds,
        "df_train_raw": df_train_raw,
        "df_test_raw": df_test_raw
    }
    
    # Save processed bundle
    save_path = Path("data/processed/dataset_bundle.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(dataset_bundle, f)
    logger.info(f"Successfully saved complete dataset bundle to {save_path}")
    
    return dataset_bundle

if __name__ == "__main__":
    bundle = build_train_test_datasets()
    print("Bundle keys:", bundle.keys())
