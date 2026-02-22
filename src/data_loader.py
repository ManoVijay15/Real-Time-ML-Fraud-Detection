import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads dataset from given path.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_csv(path)
    return df
