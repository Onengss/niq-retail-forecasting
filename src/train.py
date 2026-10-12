"""Train a retail sales model and track runs in MLflow."""
from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "daily_sales.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
EXPERIMENT_NAME = "retail-sales-forecasting"

FEATURES = [
    "product_id_code", "day_of_week", "day_of_month", "month",
    "lag_1", "lag_7", "lag_14", "rolling_mean_7"
]

def build_product_mapping(daily):
    product_ids = sorted(daily["product_id"].astype(str).unique())
    return {
        product_id: index
        for index, product_id in enumerate(product_ids)
    }

def wape(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denominator) if denominator else float("nan")

def build_features(daily):
    pieces = []
    # Create one stable mapping across all products; persist this mapping for inference.
    product_mapping = build_product_mapping(daily)
    # Use a consistent daily calendar per product so absent dates are represented as zero sales.
    for product_id, group in daily.groupby("product_id"):
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"])
        group = group.sort_values("date").set_index("date")
        full_dates = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(full_dates)
        group.index.name = "date"
        group["product_id"] = str(product_id)
        group["units_sold"] = group["units_sold"].fillna(0.0)
        group["revenue"] = group["revenue"].fillna(0.0)
        group = group.reset_index()
        group["product_id_code"] = group["product_id"].map(product_mapping).astype(int)
        group["day_of_week"] = group["date"].dt.dayofweek
        group["day_of_month"] = group["date"].dt.day
        group["month"] = group["date"].dt.month
        group["lag_1"] = group["units_sold"].shift(1)
        group["lag_7"] = group["units_sold"].shift(7)
        group["lag_14"] = group["units_sold"].shift(14)
        group["rolling_mean_7"] = group["units_sold"].shift(1).rolling(7).mean()
        pieces.append(group)
    result = pd.concat(pieces, ignore_index=True)
    return result.dropna(subset=["lag_1", "lag_7", "lag_14", "rolling_mean_7"]).copy()

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {DATA_FILE}. Run `python -m src.prepare_data` first."
        )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_FILE, parse_dates=["date"])
    product_mapping = build_product_mapping(raw)
    frame = build_features(raw)
    frame = frame.sort_values("date").reset_index(drop=True)

    # Hold out the latest 20% of unique dates to simulate forecasting into the future.
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) < 30:
        raise ValueError("Not enough dates for a useful time-based evaluation.")
    cutoff_index = max(1, int(len(dates) * 0.8))
    cutoff_date = pd.Timestamp(dates[cutoff_index])
    train = frame[frame["date"] < cutoff_date].copy()
    test = frame[frame["date"] >= cutoff_date].copy()
    if train.empty or test.empty:
        raise ValueError("Time-based split produced an empty train or test set.")

    y_train = train["units_sold"].astype(float)
    y_test = test["units_sold"].astype(float)
    baseline_pred = test["lag_7"].astype(float).clip(lower=0)

    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_wape = wape(y_test, baseline_pred)

    mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Baseline run is logged separately for an honest comparison.
    with mlflow.start_run(run_name="baseline_lag_7"):
        mlflow.log_param("model_type", "naive_weekly_baseline")
        mlflow.log_param("split_type", "chronological")
        mlflow.log_param("test_start_date", str(cutoff_date.date()))
        mlflow.log_metric("mae", float(baseline_mae))
        mlflow.log_metric("rmse", float(baseline_rmse))
        mlflow.log_metric("wape", float(baseline_wape))
        mlflow.set_tag("project", "niq-retail-sales-forecasting")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
    ])
    X_train, X_test = train[FEATURES], test[FEATURES]

    with mlflow.start_run(run_name="random_forest_v1") as run:
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "random_state": 42,
            "split_type": "chronological",
            "test_start_date": str(cutoff_date.date()),
            "feature_count": len(FEATURES),
        })
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, None)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        model_wape = wape(y_test, pred)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "wape": model_wape})
        mlflow.set_tags({
            "project": "niq-retail-sales-forecasting",
            "model_stage": "development",
            "data_source": "UCI Online Retail",
        })
        mlflow.log_dict({"features": FEATURES}, "feature_schema.json")
        mlflow.sklearn.log_model(model, artifact_path="model")

        joblib.dump(model, ARTIFACT_DIR / "forecast_model.joblib")
        metadata = {
            "model_name": "retail-sales-random-forest",
            "mlflow_run_id": run.info.run_id,
            "experiment_name": EXPERIMENT_NAME,
            "features": FEATURES,
            "product_mapping": product_mapping,
            "test_start_date": str(cutoff_date.date()),
            "metrics": {"mae": mae, "rmse": rmse, "wape": model_wape},
            "baseline_metrics": {
                "mae": float(baseline_mae),
                "rmse": float(baseline_rmse),
                "wape": float(baseline_wape),
            },
        }
        (ARTIFACT_DIR / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    comparison = pd.DataFrame([
        {"model": "Weekly naive baseline", "mae": baseline_mae, "rmse": baseline_rmse, "wape": baseline_wape},
        {"model": "Random Forest", "mae": mae, "rmse": rmse, "wape": model_wape},
    ])
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    output = test[["date", "product_id", "units_sold"]].copy()
    output["baseline_prediction"] = baseline_pred.to_numpy()
    output["model_prediction"] = pred
    output.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    print("\nEvaluation results (lower is better):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nTest period starts: {cutoff_date.date()}")
    print(f"Saved model: {ARTIFACT_DIR / 'forecast_model.joblib'}")
    print(f"MLflow run ID: {metadata['mlflow_run_id']}")
    print("\nImportant: do not assume Random Forest is better. Use the test metrics to decide.")

if __name__ == "__main__":
    main()
