"""Schedule and monitor the NIQ ML monitoring pipeline."""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
REPORT_DIR = PROJECT_DIR / "reports"

LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "monitoring_pipeline.log"
STATUS_FILE = REPORT_DIR / "pipeline_status.json"

INTERVAL_SECONDS = 3600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)


def run_pipeline():
    logging.info("Starting scheduled monitoring pipeline")
    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.monitoring_pipeline"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.stdout:
            logging.info("\n%s", result.stdout)

        if result.stderr:
            logging.warning("\n%s", result.stderr)

        success = result.returncode == 0

    except Exception:
        logging.exception("Failed to execute monitoring pipeline")
        success = False

    duration = time.time() - start

    status = {
        "success": int(success),
        "last_run_timestamp": time.time(),
        "duration_seconds": duration,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write atomically so the metrics exporter never reads
    # a partially written JSON file.
    temp_file = STATUS_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(STATUS_FILE)

    if success:
        logging.info(
            "Monitoring pipeline completed successfully in %.2f seconds",
            duration,
        )
    else:
        logging.error(
            "Monitoring pipeline failed after %.2f seconds",
            duration,
        )


def main():
    logging.info("NIQ monitoring scheduler started")
    logging.info("Execution interval: %s seconds", INTERVAL_SECONDS)

    while True:
        run_pipeline()
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
