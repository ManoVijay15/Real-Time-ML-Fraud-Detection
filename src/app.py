from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import mlflow.pyfunc

THRESHOLD=0.7
prediction_count=0
app = FastAPI()

# Load trained model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/FraudDetectionModel@champion")


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