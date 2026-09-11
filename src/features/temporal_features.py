"""
Temporal and Calendar Feature Engineering for AQI Forecasting.
Strictly ensures zero future leakage by lagging all rolling calculations.
"""
import numpy as np
import pandas as pd
from typing import List

def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add calendar, seasonal, and cyclic trigonometric features."""
    df_out = df.copy()
    dates = pd.to_datetime(df_out[date_col])
    
    df_out["day_of_week"] = dates.dt.dayofweek
    df_out["month"] = dates.dt.month
    df_out["day_of_year"] = dates.dt.dayofyear
    df_out["is_weekend"] = (df_out["day_of_week"] >= 5).astype(int)
    
    # India Meteorological Seasons:
    # 0: Winter (Jan-Feb), 1: Summer/Pre-monsoon (Mar-May), 2: Monsoon (Jun-Sep), 3: Post-monsoon (Oct-Dec)
    def get_indian_season(m: int) -> int:
        if m in [1, 2]:
            return 0  # Winter
        elif m in [3, 4, 5]:
            return 1  # Summer
        elif m in [6, 7, 8, 9]:
            return 2  # Monsoon
        else:
            return 3  # Post-monsoon
            
    df_out["season"] = df_out["month"].apply(get_indian_season)
    
    # Cyclic sine/cosine transformations
    df_out["sin_day_of_year"] = np.sin(2 * np.pi * df_out["day_of_year"] / 365.25)
    df_out["cos_day_of_year"] = np.cos(2 * np.pi * df_out["day_of_year"] / 365.25)
    df_out["sin_month"] = np.sin(2 * np.pi * df_out["month"] / 12.0)
    df_out["cos_month"] = np.cos(2 * np.pi * df_out["month"] / 12.0)
    df_out["sin_dow"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["cos_dow"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)
    
    return df_out

def add_lag_features(df: pd.DataFrame, 
                     cols: List[str], 
                     lags: List[int] = [1, 2, 3, 7]) -> pd.DataFrame:
    """
    Generate historical lag features.
    Lag k corresponds to observation at time t - k.
    """
    df_out = df.copy()
    for col in cols:
        for lag in lags:
            df_out[f"{col}_lag_{lag}"] = df_out[col].shift(lag)
    return df_out

def add_rolling_features(df: pd.DataFrame, 
                         cols: List[str], 
                         windows: List[int] = [7, 14]) -> pd.DataFrame:
    """
    Generate rolling statistics (mean, std, min, max, ewma).
    CRITICAL: Uses shift(1) so the current observation t is NEVER included in the rolling window!
    """
    df_out = df.copy()
    for col in cols:
        shifted = df_out[col].shift(1)
        for w in windows:
            df_out[f"{col}_roll_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
            df_out[f"{col}_roll_std_{w}"] = shifted.rolling(window=w, min_periods=1).std().fillna(0.0)
            df_out[f"{col}_roll_min_{w}"] = shifted.rolling(window=w, min_periods=1).min()
            df_out[f"{col}_roll_max_{w}"] = shifted.rolling(window=w, min_periods=1).max()
        # EWMA with 7-day span
        df_out[f"{col}_ewma_7"] = shifted.ewm(span=7, min_periods=1).mean()
    return df_out
