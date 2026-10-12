import numpy as np
import pandas as pd
from src.train import build_features, wape

def test_wape_zero_when_predictions_match():
    assert wape(np.array([2, 4]), np.array([2, 4])) == 0.0

def test_build_features_creates_lags():
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    sample = pd.DataFrame({
        "date": dates,
        "product_id": ["A"] * len(dates),
        "units_sold": [float(i + 1) for i in range(len(dates))],
        "revenue": [float(i + 1) for i in range(len(dates))],
    })
    result = build_features(sample)
    assert {"lag_1", "lag_7", "lag_14", "rolling_mean_7"} <= set(result.columns)
    assert not result[["lag_1", "lag_7", "lag_14", "rolling_mean_7"]].isna().any().any()
