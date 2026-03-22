from kafka import KafkaProducer
import json
import time
import pandas as pd

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv("/Users/mk004/Documents/GH/creditcard.csv")

for transaction_id, _ in df.iterrows():
    producer.send("fraud_transactions", {"transaction_id": transaction_id})

    print(f"Sent transaction_id: {transaction_id}")

    time.sleep(1)
