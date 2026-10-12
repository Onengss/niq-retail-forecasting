# NIQ Portfolio Project 1: Retail Sales Forecasting with MLflow

A portfolio-grade starter project demonstrating a reproducible retail demand forecasting workflow.

## What it demonstrates
- Public retail transaction data ingestion (UCI Online Retail)
- Data cleaning and daily aggregation by product
- Time-based train/test split to reduce temporal leakage
- Baseline forecast vs. Random Forest regression
- MAE, RMSE, and WAPE evaluation
- MLflow experiment tracking (parameters, metrics, model artifact)
- FastAPI prediction endpoint
- Unit tests and Docker packaging

> This is a learning/portfolio project, not a production forecasting system. The public dataset is historical UK online retail data and does not represent NIQ internal data.

## Requirements
- Windows 10/11
- Python 3.11 recommended
- VS Code with Python extension
- Git (recommended)
- Docker Desktop (optional until the Docker stage)

## 1. Create and activate a virtual environment (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## 2. Download and prepare the dataset

```powershell
python -m src.download_data
python -m src.prepare_data
```

The download script fetches the UCI Online Retail ZIP file and extracts the workbook into `data/raw/`. If the source URL changes, download the dataset from https://archive.ics.uci.edu/dataset/352/online+retail and place `Online Retail.xlsx` in `data/raw/`.

Preparation removes cancellations, non-positive quantities/prices, and rows without product/date values; it then aggregates quantity by date and product. It keeps the top products by total units to make the starter experiment manageable.

## 3. Train and compare models

```powershell
python -m src.train
```

This creates:
- `artifacts/forecast_model.joblib`
- `artifacts/model_metadata.json`
- `reports/test_predictions.csv`
- `reports/model_comparison.csv`
- local MLflow run data in `mlruns/`

The baseline uses the previous week's sales (`lag_7`). The Random Forest uses lag and calendar features. The split is chronological, not random.

## 4. Open the MLflow UI

In a second terminal, activate the same virtual environment and run:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Select the `retail-sales-forecasting` experiment and compare the logged runs. MLflow tracking records experiment parameters, metrics and model artifacts.

## 5. Start the prediction API

Train the model first, then run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8001
```

The API provides a `/predict` endpoint for requesting forecasts. FastAPI automatically generates interactive API documentation at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs), where you can try `POST /predict`.

Use a product ID that exists in the prepared dataset. Example request:
```json
{
  "product_id": "85123A",
  "date": "2011-12-10",
  "lag_1": 12,
  "lag_7": 10,
  "lag_14": 9,
  "rolling_mean_7": 11.2
}
```

The values above are illustrative, not actual observations. Replace the product code if it isn't among your selected products.

The API uses a small set of lag/calendar features. It is a demonstration endpoint; the caller must supply features for the selected product/date. A future iteration could create a feature store or fetch lag values from a database.

## 6. Run tests

```powershell
python -m pytest -q
```

The starter tests check the feature-generation logic and WAPE calculation. Passing these tests is a useful first quality gate, but it does not prove that the full forecasting system is production-ready.

## 7. Build and run with Docker (optional)

```powershell
docker build -t niq-retail-forecasting .
docker run --rm -p 8000:8000 niq-retail-forecasting
```

The Docker image expects the model artifact to exist. Run training before building the image.

## Repository structure

```text
niq-retail-forecasting/
├── data/raw/                 # downloaded dataset (not committed)
├── data/processed/            # cleaned aggregate data (not committed)
├── artifacts/                 # trained model and metadata (not committed)
├── reports/                   # evaluation reports (not committed)
├── src/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py
│   └── api.py
├── tests/
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Talking points for an interview
1. Why is a chronological split more appropriate than a random split for forecasting?
2. Why do we compare against a naive baseline instead of reporting a model metric alone?
3. What does MAE mean in units sold? What does WAPE communicate?
4. What can MLflow reproduce that a notebook alone may not capture?
5. What additional monitoring would be needed in production (data freshness, drift, latency, failures, and delayed ground-truth model quality)?

## Known limitations and next steps
- The first version models daily sales per product, not all retail demand drivers.
- It does not yet include holidays, promotions, price elasticity, or stock-outs.
- Missing product/date combinations are treated as zero sales when constructing the time series.
- WAPE can be unstable for very low-volume products; inspect metrics alongside per-product performance.
- Add SQL storage, a monitoring dashboard, automated retraining, CI, and model promotion in later portfolio projects.
