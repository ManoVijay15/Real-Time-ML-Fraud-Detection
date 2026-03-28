import pandas as pd


def load_reference_data(reference_path="monitoring/reference_data.csv"):
    """Load the baseline reference dataset saved during training."""
    df = pd.read_csv(reference_path)
    print(f"Loaded reference data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_current_data(current_path="monitoring/current_data.csv"):
    """Load the current production data captured from live predictions."""
    df = pd.read_csv(current_path)
    print(f"Loaded current data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
