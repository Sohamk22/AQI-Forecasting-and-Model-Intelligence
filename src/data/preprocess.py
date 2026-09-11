"""
Data preprocessing pipeline for Indian AQI and meteorological time-series.
Strictly prevents leakage by fitting transformations only on training data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from src.utils.logger import get_logger
from src.utils.config import DataConfig

logger = get_logger("Preprocessing")

def load_and_merge_data(pollutant_path: str = "data/raw/city_day.csv",
                        weather_path: str = "data/raw/weather_daily.csv",
                        city: str = "Delhi") -> pd.DataFrame:
    """Load CPCB pollutant records for a city and merge with ERA5 weather data."""
    logger.info(f"Loading CPCB air quality data from {pollutant_path} for city: {city}")
    df_pollutants = pd.read_csv(pollutant_path)
    df_city = df_pollutants[df_pollutants["City"].str.lower() == city.lower()].copy()
    
    if len(df_city) == 0:
        available = df_pollutants["City"].dropna().unique().tolist()
        raise ValueError(f"City '{city}' not found in {pollutant_path}. Available: {available[:10]}")
    
    df_city["Date"] = pd.to_datetime(df_city["Date"])
    df_city = df_city.sort_values("Date").reset_index(drop=True)
    
    logger.info(f"Loading meteorological data from {weather_path}...")
    df_weather = pd.read_csv(weather_path)
    df_weather["Date"] = pd.to_datetime(df_weather["Date"])
    
    # Merge chronologically on Date
    merged = pd.merge(df_city, df_weather, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    
    logger.info(f"Successfully merged data: {len(merged)} daily records from {merged['Date'].min().strftime('%Y-%m-%d')} to {merged['Date'].max().strftime('%Y-%m-%d')}")
    return merged

def impute_missing_linear(df: pd.DataFrame, feature_cols: List[str], target_col: str = "AQI") -> pd.DataFrame:
    """
    Impute missing pollutant and target observations using linear interpolation,
    as specified in the reference paper methodology to preserve temporal continuity.
    """
    df_out = df.copy()
    cols_to_impute = [c for c in feature_cols + [target_col] if c in df_out.columns]
    
    for col in cols_to_impute:
        n_missing = df_out[col].isnull().sum()
        if n_missing > 0:
            logger.info(f"Interpolating {n_missing} missing values in '{col}' (linear interpolation with boundary forward/back-fill)...")
            df_out[col] = df_out[col].interpolate(method="linear", limit_direction="both")
            # Fill any remaining edge boundary NaNs
            df_out[col] = df_out[col].ffill().bfill()
            
    return df_out

def compute_iqr_bounds(df_train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute IQR Winsorization bounds strictly on the training partition."""
    bounds = {}
    for col in feature_cols:
        if col in df_train.columns and np.issubdtype(df_train[col].dtype, np.number):
            q1 = df_train[col].quantile(0.25)
            q3 = df_train[col].quantile(0.75)
            iqr = q3 - q1
            lower = max(0.0, q1 - 1.5 * iqr) # Pollutants and weather physical lower bound >= 0 where appropriate
            upper = q3 + 1.5 * iqr
            bounds[col] = (lower, upper)
    return bounds

def apply_winsorization(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply pre-computed training IQR bounds to winsorize extreme outliers."""
    df_out = df.copy()
    outlier_counts = {}
    for col, (lower, upper) in bounds.items():
        if col in df_out.columns:
            n_outliers = ((df_out[col] < lower) | (df_out[col] > upper)).sum()
            outlier_counts[col] = int(n_outliers)
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
    return df_out, outlier_counts

def build_clean_dataset(config: DataConfig = DataConfig()) -> pd.DataFrame:
    """Execute end-to-end data cleaning and imputation pipeline."""
    merged = load_and_merge_data(
        pollutant_path=config.raw_pollutant_file,
        weather_path=config.raw_weather_file,
        city=config.city
    )
    
    all_features = config.pollutant_cols + config.weather_cols
    cleaned = impute_missing_linear(merged, all_features, config.target_col)
    
    # Save interim cleaned data
    interim_path = Path("data/interim/cleaned_merged_daily.csv")
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(interim_path, index=False)
    logger.info(f"Saved interim cleaned data to {interim_path}")
    
    return cleaned

if __name__ == "__main__":
    df = build_clean_dataset()
    print("Cleaned head:")
    print(df[["Date", "City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "Wind_Speed", "AQI"]].head(3))
