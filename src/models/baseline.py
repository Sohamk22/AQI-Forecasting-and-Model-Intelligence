"""
Statistical Climatological Baseline (Historical Calendar Anchor).
Retrieves observations within a +/- 7-day window across previous historical years.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class HistoricalAnchorBaseline:
    """
    Climatological baseline that predicts AQI for date d by computing the mean
    of observations within a +/- window_days window around the same day of year
    across all historical training years.
    """
    def __init__(self, window_days: int = 7):
        self.window_days = window_days
        self.historical_records = None
        
    def fit(self, dates, y_train: np.ndarray):
        """Store historical observations with their day of year and year."""
        dt = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        self.historical_records = pd.DataFrame({
            "year": dt.dt.year,
            "dayofyear": dt.dt.dayofyear,
            "AQI": y_train
        })
        return self
        
    def predict(self, test_dates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions, 5th percentile bounds, and 95th percentile bounds
        for each test date using historical training records.
        """
        dt_test = pd.Series(pd.to_datetime(test_dates)).reset_index(drop=True)
        preds = []
        lower_bounds = []
        upper_bounds = []
        
        overall_mean = self.historical_records["AQI"].mean()
        overall_p05 = self.historical_records["AQI"].quantile(0.05)
        overall_p95 = self.historical_records["AQI"].quantile(0.95)
        
        for d in dt_test.dt.dayofyear:
            # Handle circular year wrap-around (e.g. late Dec to early Jan)
            d_min = d - self.window_days
            d_max = d + self.window_days
            
            if d_min < 1:
                mask = (self.historical_records["dayofyear"] >= (365 + d_min)) | (self.historical_records["dayofyear"] <= d_max)
            elif d_max > 365:
                mask = (self.historical_records["dayofyear"] >= d_min) | (self.historical_records["dayofyear"] <= (d_max - 365))
            else:
                mask = (self.historical_records["dayofyear"] >= d_min) & (self.historical_records["dayofyear"] <= d_max)
                
            matched = self.historical_records[mask]["AQI"]
            if len(matched) > 0:
                preds.append(matched.mean())
                lower_bounds.append(matched.quantile(0.05))
                upper_bounds.append(matched.quantile(0.95))
            else:
                preds.append(overall_mean)
                lower_bounds.append(overall_p05)
                upper_bounds.append(overall_p95)
                
        return np.array(preds), np.array(lower_bounds), np.array(upper_bounds)
