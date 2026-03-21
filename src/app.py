import os
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import mlflow.pyfunc

THRESHOLD=0.7
prediction_count=0
app = FastAPI()

# Load trained model — use MODEL_PATH env var if set, otherwise fall back to registry
_model_path = os.getenv(
    "MODEL_PATH",
    "models:/FraudDetectionModel@champion"
)
_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _tracking_uri:
    mlflow.set_tracking_uri(_tracking_uri)

model = mlflow.pyfunc.load_model(_model_path)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: dict):
    
    global prediction_count
    prediction_count +=1
    # Convert input to DataFrame
    input_df = pd.DataFrame([transaction])

    # Drop Time column if present — model was trained without it
    input_df = input_df.drop(columns=["Time"], errors="ignore")

    # Predict probability using underlying sklearn model
    probability = model._model_impl.sklearn_model.predict_proba(input_df)[0][1]

    prediction = int(probability > THRESHOLD)

    return {
        "fraud_probability": float(probability),
        "prediction": prediction
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "RandomForest Fraud Detector",
        "version": "1.0",
        "threshold": THRESHOLD
    }


@app.get("/metrics")
def metrics():
    return {
        "total_predictions": prediction_count
    }