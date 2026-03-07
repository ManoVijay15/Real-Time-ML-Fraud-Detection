from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

THRESHOLD=0.7
prediction_count=0
app = FastAPI()

# Load trained model
model = joblib.load("models/fraud_model.pkl")


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

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    prediction = int(probability > 0.5)

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