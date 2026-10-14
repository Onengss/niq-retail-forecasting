"""Export saved ML evaluation and drift reports to Prometheus."""

import json
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, start_http_server

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

MODEL_MAE = Gauge(
    "niq_model_mae",
    "Mean absolute error of the forecasting model",
    ["model"],
)

MODEL_RMSE = Gauge(
    "niq_model_rmse",
    "Root mean squared error of the forecasting model",
    ["model"],
)

MODEL_WAPE = Gauge(
    "niq_model_wape",
    "Weighted absolute percentage error of the model",
    ["model"],
)

DATA_DRIFT_SCORE = Gauge(
    "niq_data_drift_score",
    "Data drift score for each monitored feature",
    ["feature"],
)

PIPELINE_SUCCESS = Gauge(
    "niq_pipeline_success",
    "Whether the last monitoring pipeline run succeeded",
)

PIPELINE_LAST_RUN = Gauge(
    "niq_pipeline_last_run_timestamp_seconds",
    "Unix timestamp of the last completed pipeline run",
)

PIPELINE_DURATION = Gauge(
    "niq_pipeline_duration_seconds",
    "Duration of the last pipeline execution in seconds",
)


def update_metrics():
    model_report = pd.read_csv(
        REPORTS_DIR / "monitoring_summary.csv"
    )

    for _, row in model_report.iterrows():
        model = str(row["model"])

        MODEL_MAE.labels(model=model).set(float(row["MAE"]))
        MODEL_RMSE.labels(model=model).set(float(row["RMSE"]))
        MODEL_WAPE.labels(model=model).set(float(row["WAPE"]))

    drift_report = pd.read_csv(
        REPORTS_DIR / "data_drift_summary.csv"
    )

    for _, row in drift_report.iterrows():
        DATA_DRIFT_SCORE.labels(
            feature=str(row["feature"])
        ).set(float(row["drift_score"]))

    status_file = REPORTS_DIR / "pipeline_status.json"
    if status_file.exists():
        with status_file.open("r", encoding="utf-8") as file:
            status = json.load(file)
        PIPELINE_SUCCESS.set(float(status["success"]))
        PIPELINE_LAST_RUN.set(
            float(status["last_run_timestamp"])
        )
        PIPELINE_DURATION.set(
            float(status["duration_seconds"])
        )


if __name__ == "__main__":
    import time

    start_http_server(8002)
    print("NIQ model metrics exporter running on port 8002")

    while True:
        update_metrics()
        time.sleep(30)
