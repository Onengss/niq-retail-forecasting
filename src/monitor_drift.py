
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "daily_sales.csv"
PREDICTIONS_FILE = ROOT / "reports" / "test_predictions.csv"
OUTPUT_FILE = ROOT / "reports" / "data_drift_summary.csv"

# Monitoring thresholds for this portfolio demonstration.
KS_STATISTIC_THRESHOLD = 0.10
P_VALUE_THRESHOLD = 0.05
CATEGORY_DRIFT_THRESHOLD = 0.10


def compare_numeric_feature(reference, current, feature_name):
    """Compare a numeric feature using the two-sample KS test."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if len(reference) < 2 or len(current) < 2:
        return {
            "feature": feature_name,
            "reference_samples": len(reference),
            "current_samples": len(current),
            "reference_mean": reference.mean(),
            "current_mean": current.mean(),
            "drift_score": np.nan,
            "p_value": np.nan,
            "status": "INSUFFICIENT_DATA",
        }

    result = ks_2samp(reference, current)

    # Flag drift when the distribution difference is meaningful
    # under both the statistical and configured effect-size rules.
    drift_detected = (
        result.pvalue < P_VALUE_THRESHOLD
        and result.statistic >= KS_STATISTIC_THRESHOLD
    )

    return {
        "feature": feature_name,
        "reference_samples": len(reference),
        "current_samples": len(current),
        "reference_mean": round(reference.mean(), 4),
        "current_mean": round(current.mean(), 4),
        "drift_score": round(float(result.statistic), 4),
        "p_value": round(float(result.pvalue), 6),
        "status": "DRIFT_DETECTED" if drift_detected else "NO_DRIFT",
    }


def compare_category_distribution(reference, current, feature_name):
    """Compare category proportions using total variation distance."""
    reference = reference.fillna("MISSING").astype(str)
    current = current.fillna("MISSING").astype(str)

    if reference.empty or current.empty:
        return {
            "feature": feature_name,
            "reference_samples": len(reference),
            "current_samples": len(current),
            "reference_mean": np.nan,
            "current_mean": np.nan,
            "drift_score": np.nan,
            "p_value": np.nan,
            "status": "INSUFFICIENT_DATA",
        }

    reference_share = reference.value_counts(normalize=True)
    current_share = current.value_counts(normalize=True)

    categories = reference_share.index.union(current_share.index)

    reference_share = reference_share.reindex(
        categories, fill_value=0
    )
    current_share = current_share.reindex(
        categories, fill_value=0
    )

    # Total variation distance ranges from 0 (identical)
    # to 1 (completely different).
    score = 0.5 * np.abs(reference_share - current_share).sum()

    return {
        "feature": feature_name,
        "reference_samples": len(reference),
        "current_samples": len(current),
        "reference_mean": np.nan,
        "current_mean": np.nan,
        "drift_score": round(float(score), 4),
        "p_value": np.nan,
        "status": (
            "DRIFT_DETECTED"
            if score >= CATEGORY_DRIFT_THRESHOLD
            else "NO_DRIFT"
        ),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}. "
            "Run the data preparation step first."
        )

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing predictions: {PREDICTIONS_FILE}. "
            "Run `python -m src.train` first."
        )

    data = pd.read_csv(DATA_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required_data_columns = {"date", "product_id", "units_sold"}
    required_prediction_columns = {"date"}

    if not required_data_columns.issubset(data.columns):
        raise ValueError(
            f"Dataset must contain: {required_data_columns}. "
            f"Found: {set(data.columns)}"
        )

    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(
            "Test predictions must contain a 'date' column."
        )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )

    data = data.dropna(subset=["date"])
    predictions = predictions.dropna(subset=["date"])

    if data.empty or predictions.empty:
        raise ValueError("Dataset or predictions contain no valid dates.")

    # Use the first test date as the boundary between the
    # historical reference period and the current test period.
    test_start_date = predictions["date"].min()

    reference = data[data["date"] < test_start_date].copy()
    current = data[data["date"] >= test_start_date].copy()

    if reference.empty or current.empty:
        raise ValueError(
            "Cannot create reference/current periods. "
            "Check the dates in daily_sales.csv and test_predictions.csv."
        )

    results = [
        compare_numeric_feature(
            reference["units_sold"],
            current["units_sold"],
            "units_sold",
        ),
        compare_category_distribution(
            reference["product_id"],
            current["product_id"],
            "product_id_distribution",
        ),
    ]

    # Monitor revenue too, when available in the prepared dataset.
    if "revenue" in data.columns:
        results.append(
            compare_numeric_feature(
                reference["revenue"],
                current["revenue"],
                "revenue",
            )
        )

    summary = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nData Drift Monitoring Summary")
    print("-" * 65)
    print(f"Reference period: before {test_start_date.date()}")
    print(f"Current period:   from {test_start_date.date()}")
    print(f"Reference rows:   {len(reference)}")
    print(f"Current rows:     {len(current)}")
    print()
    print(summary.to_string(index=False))
    print(f"\nSaved report to: {OUTPUT_FILE}")

    if (summary["status"] == "DRIFT_DETECTED").any():
        print("\nALERT: Potential data drift detected.")
    else:
        print("\nNo drift detected using the configured thresholds.")


if __name__ == "__main__":
    main()
