
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = ROOT / "reports" / "test_predictions.csv"
OUTPUT_FILE = ROOT / "reports" / "monitoring_summary.csv"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / total_actual
        if total_actual > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    }


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Run `python -m src.train` first."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    actual = df["units_sold"]
    model_predictions = df["model_prediction"]
    baseline_predictions = df["baseline_prediction"]

    results = pd.DataFrame([
        {
            "model": "Weekly baseline",
            **calculate_metrics(actual, baseline_predictions),
        },
        {
            "model": "Random Forest",
            **calculate_metrics(actual, model_predictions),
        },
    ])

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nModel monitoring summary")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
