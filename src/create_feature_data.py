import pandas as pd

# Load dataset
df = pd.read_csv("/Users/mk004/Documents/GH/creditcard.csv")

# Create entity key required by Feast
df["transaction_id"] = range(len(df))

# Create timestamp column required by Feast
df["event_timestamp"] = pd.Timestamp.now()

# Save parquet for feature store
output_path = "fraud_feature_store/feature_repo/data/processed/transactions.parquet"

df.to_parquet(output_path, index=False)

print("Parquet feature data created successfully!")