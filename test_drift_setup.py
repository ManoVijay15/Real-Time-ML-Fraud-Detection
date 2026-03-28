import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

ref = pd.DataFrame({
    "amount": np.random.normal(50, 10, n),
    "hour": np.random.randint(0, 24, n),
    "merchant_category": np.random.choice(["food", "travel", "retail"], n),
    "target": np.random.choice([0, 1], n, p=[0.98, 0.02]),
})
ref.to_csv("monitoring/reference_data.csv", index=False)

cur = pd.DataFrame({
    "amount": np.random.normal(120, 30, n),
    "hour": np.random.randint(0, 24, n),
    "merchant_category": np.random.choice(["food", "travel", "retail", "crypto"], n),
})
cur.to_csv("monitoring/current_data.csv", index=False)

print("CSVs created.")
