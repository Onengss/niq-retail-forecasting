"""Small FastAPI service for the trained retail forecasting model."""
from datetime import date
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    make_asgi_app,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "forecast_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

REQUEST_COUNT = Counter(
    "niq_api_requests_total",
    "Total number of API HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "niq_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)

app = FastAPI(
    title="Retail Sales Forecast API",
    description="Portfolio demonstration API for a retail demand forecasting model.",
    version="1.0.0",
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start_time
        REQUEST_COUNT.labels(
            request.method,
            request.url.path,
            str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            request.method,
            request.url.path,
        ).observe(duration)


metrics_app = make_asgi_app()


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


app.mount("/metrics", metrics_app)


@app.get("/favicon.ico", include_in_schema=False, status_code=204)
def favicon():
    return Response(status_code=204)


class PredictionRequest(BaseModel):
    product_id: str = Field(..., min_length=1, description="Retail stock/product code")
    date: date
    lag_1: float = Field(..., ge=0, description="Units sold one day earlier")
    lag_7: float = Field(..., ge=0, description="Units sold seven days earlier")
    lag_14: float = Field(..., ge=0, description="Units sold fourteen days earlier")
    rolling_mean_7: float = Field(..., ge=0, description="Mean units sold over previous seven days")

def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model files not found. Run `python -m src.train` first.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata

@app.get("/")
def root():
    return {"message": "Retail Sales Forecast API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_not_found",
        "model_available": MODEL_PATH.exists(),
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, metadata = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Reuse the exact product mapping learned during training.
    product_mapping = metadata.get("product_mapping", {})
    if request.product_id not in product_mapping:
        raise HTTPException(
            status_code=422,
            detail="Unknown product_id. Use one of the product IDs in the prepared training dataset.",
        )
    product_code = product_mapping[request.product_id]
    d = pd.Timestamp(request.date)
    features = pd.DataFrame([{
        "product_id_code": product_code,
        "day_of_week": d.dayofweek,
        "day_of_month": d.day,
        "month": d.month,
        "lag_1": request.lag_1,
        "lag_7": request.lag_7,
        "lag_14": request.lag_14,
        "rolling_mean_7": request.rolling_mean_7,
    }])[metadata["features"]]
    prediction = max(0.0, float(model.predict(features)[0]))
    return {
        "product_id": request.product_id,
        "forecast_date": request.date.isoformat(),
        "predicted_units": round(prediction, 2),
        "model_name": metadata["model_name"],
        "note": "Portfolio demonstration using the persisted training-time product mapping.",
    }
