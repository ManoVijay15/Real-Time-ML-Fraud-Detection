import subprocess
import sys

from src.monitoring.drift_detection import detect_drift


DRIFT_THRESHOLD = 0.3  # Retrain if more than 30% of features have drifted


def check_and_trigger_retrain(
    reference_path="monitoring/reference_data.csv",
    current_path="monitoring/current_data.csv",
):
    print("Running drift detection...")
    result = detect_drift(reference_path=reference_path, current_path=current_path)

    drift_ratio = result["drifted_columns"] / max(result["total_columns"], 1)

    if result["dataset_drift"] or drift_ratio >= DRIFT_THRESHOLD:
        print(
            f"Drift ratio {drift_ratio:.2%} exceeds threshold {DRIFT_THRESHOLD:.2%}. "
            "Triggering retraining..."
        )
        _trigger_retrain()
    else:
        print(
            f"Drift ratio {drift_ratio:.2%} is within acceptable range. No retraining needed."
        )


def _trigger_retrain():
    subprocess.run(
        [sys.executable, "src/train.py"],
        check=True,
    )
    print("Retraining complete.")


if __name__ == "__main__":
    check_and_trigger_retrain()
