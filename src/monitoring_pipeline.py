"""Run model evaluation and data drift monitoring."""

import subprocess
import sys
from datetime import datetime, timezone


def run_step(module_name):
    print(f"\nStarting {module_name}", flush=True)

    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True,
    )

    print(f"Completed {module_name}", flush=True)


def main():
    started = datetime.now(timezone.utc)
    print(f"Monitoring pipeline started: {started.isoformat()}")

    try:
        run_step("src.monitor_model")
        run_step("src.monitor_drift")
    except subprocess.CalledProcessError as error:
        print(f"Pipeline failed: {error}")
        sys.exit(1)

    finished = datetime.now(timezone.utc)
    print(f"\nPipeline completed: {finished.isoformat()}")


if __name__ == "__main__":
    main()
