import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def detect_drift(
    reference_path="monitoring/reference_data.csv",
    current_path="monitoring/current_data.csv",
    report_path="monitoring/drift_report.html",
):
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    # Drop target column from reference if present — drift is measured on features only
    reference_features = reference_data.drop(columns=["target"], errors="ignore")
    current_features = current_data.copy()

    # Align columns: only compare columns present in both datasets
    shared_cols = [c for c in reference_features.columns if c in current_features.columns]
    reference_features = reference_features[shared_cols]
    current_features = current_features[shared_cols]

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_features,
        current_data=current_features,
    )

    report.save_html(report_path)
    print(f"Drift report saved to {report_path}")

    # Return drift summary for programmatic use
    report_dict = report.as_dict()
    drift_results = report_dict["metrics"][0]["result"]

    n_drifted = drift_results.get("number_of_drifted_columns", 0)
    n_total = drift_results.get("number_of_columns", len(shared_cols))
    dataset_drift = drift_results.get("dataset_drift", False)

    print(f"Dataset drift detected: {dataset_drift}")
    print(f"Drifted features: {n_drifted}/{n_total}")

    return {
        "dataset_drift": dataset_drift,
        "drifted_columns": n_drifted,
        "total_columns": n_total,
    }


if __name__ == "__main__":
    detect_drift()
