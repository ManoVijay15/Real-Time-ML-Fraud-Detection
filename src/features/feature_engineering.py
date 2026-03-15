import pandas as pd
import numpy as np

def build_features(df):

    df["amount_log"] = np.log1p(df["Amount"])

    df["hour"] = (df["Time"] // 3600) % 24

    df["amount_to_hour_ratio"] = df["Amount"] / (df["hour"] + 1)

    return df