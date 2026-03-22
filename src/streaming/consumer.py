from kafka import KafkaConsumer
import json
import pandas as pd
import requests
from feast import FeatureStore

store = FeatureStore(repo_path="fraud_feature_store/feature_repo")

consumer = KafkaConsumer(
    "fraud_transactions",
    bootstrap_servers="localhost:9092",
    group_id="fraud-detector",
    auto_offset_reset="latest",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

for message in consumer:
    transaction_id = message.value["transaction_id"]

    # Fetch features from Feast online store
    feature_vector = store.get_online_features(
        features=[
            "fraud_features:V1", "fraud_features:V2", "fraud_features:V3", "fraud_features:V4",
            "fraud_features:V5", "fraud_features:V6", "fraud_features:V7", "fraud_features:V8",
            "fraud_features:V9", "fraud_features:V10", "fraud_features:V11", "fraud_features:V12",
            "fraud_features:V13", "fraud_features:V14", "fraud_features:V15", "fraud_features:V16",
            "fraud_features:V17", "fraud_features:V18", "fraud_features:V19", "fraud_features:V20",
            "fraud_features:V21", "fraud_features:V22", "fraud_features:V23", "fraud_features:V24",
            "fraud_features:V25", "fraud_features:V26", "fraud_features:V27", "fraud_features:V28",
            "fraud_features:Amount",
        ],
        entity_rows=[{"transaction_id": transaction_id}],
    ).to_dict()

    # Remove the entity key and unwrap single-element lists Feast returns
    feature_vector.pop("transaction_id", None)
    feature_vector = {k: v[0] for k, v in feature_vector.items()}

    response = requests.post(
        "http://localhost:8000/predict",
        json=feature_vector
    )

    if response.ok:
        print(f"transaction_id={transaction_id} → {response.json()}")
    else:
        print(f"transaction_id={transaction_id} → HTTP {response.status_code}: {response.text}")
