"""Download the UCI Online Retail dataset."""
from pathlib import Path
import io
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "Online Retail.xlsx"
    if target.exists():
        print(f"Dataset already exists: {target}")
        return
    print("Downloading UCI Online Retail dataset...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        candidates = [n for n in archive.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not candidates:
            raise RuntimeError(f"No Excel workbook found in archive. Contents: {archive.namelist()}")
        # Use the first workbook in the archive.
        target.write_bytes(archive.read(candidates[0]))
    print(f"Saved dataset to {target}")

if __name__ == "__main__":
    main()
