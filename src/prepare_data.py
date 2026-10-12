"""Clean UCI Online Retail transactions and create daily product sales."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Online Retail.xlsx"
OUT_FILE = ROOT / "data" / "processed" / "daily_sales.csv"

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_FILE}. Run `python -m src.download_data` first."
        )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("Reading workbook (this may take a little while)...")
    df = pd.read_excel(RAW_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # Credit-note invoices begin with C. Exclude cancellations and invalid sales.
    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C")
        & df["InvoiceDate"].notna()
        & df["StockCode"].notna()
        & (df["StockCode"] != "")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()

    df["date"] = df["InvoiceDate"].dt.normalize()
    daily = (
        df.groupby(["date", "StockCode"], as_index=False)
          .agg(units_sold=("Quantity", "sum"), revenue=("UnitPrice", lambda s: 0.0))
    )
    # Calculate revenue from the cleaned transaction data separately, preserving units.
    df["line_revenue"] = df["Quantity"] * df["UnitPrice"]
    revenue = df.groupby(["date", "StockCode"], as_index=False)["line_revenue"].sum()
    daily = daily.drop(columns=["revenue"]).merge(
        revenue, on=["date", "StockCode"], how="left"
    ).rename(columns={"StockCode": "product_id", "line_revenue": "revenue"})

    # Limit the starter project to the top 10 products by total units sold.
    top_products = (
        daily.groupby("product_id")["units_sold"].sum()
             .nlargest(10).index
    )
    daily = daily[daily["product_id"].isin(top_products)].copy()
    daily = daily.sort_values(["product_id", "date"])
    daily.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(daily):,} daily product rows to {OUT_FILE}")
    print("Top products:", ", ".join(top_products.astype(str)))

if __name__ == "__main__":
    main()
