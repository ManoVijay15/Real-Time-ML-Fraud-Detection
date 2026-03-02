from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/fraud_model.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: dict):
    # Convert input to DataFrame
    input_df = pd.DataFrame([transaction])

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    prediction = int(probability > 0.5)

    return {
        "fraud_probability": float(probability),
        "prediction": prediction
    }