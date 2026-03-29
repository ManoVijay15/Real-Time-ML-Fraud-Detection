import logging
import os
import threading
from typing import Optional

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pythonjsonlogger import json as jsonlogger
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger


logger = _build_logger("fraud_api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.7"))
CURRENT_DATA_PATH = "monitoring/current_data.csv"
FEATURE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount"]

_count_lock = threading.Lock()
_prediction_count = 0

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    V1: float = Field(..., ge=-100.0, le=100.0)
    V2: float = Field(..., ge=-100.0, le=100.0)
    V3: float = Field(..., ge=-100.0, le=100.0)
    V4: float = Field(..., ge=-100.0, le=100.0)
    V5: float = Field(..., ge=-100.0, le=100.0)
    V6: float = Field(..., ge=-100.0, le=100.0)
    V7: float = Field(..., ge=-100.0, le=100.0)
    V8: float = Field(..., ge=-100.0, le=100.0)
    V9: float = Field(..., ge=-100.0, le=100.0)
    V10: float = Field(..., ge=-100.0, le=100.0)
    V11: float = Field(..., ge=-100.0, le=100.0)
    V12: float = Field(..., ge=-100.0, le=100.0)
    V13: float = Field(..., ge=-100.0, le=100.0)
    V14: float = Field(..., ge=-100.0, le=100.0)
    V15: float = Field(..., ge=-100.0, le=100.0)
    V16: float = Field(..., ge=-100.0, le=100.0)
    V17: float = Field(..., ge=-100.0, le=100.0)
    V18: float = Field(..., ge=-100.0, le=100.0)
    V19: float = Field(..., ge=-100.0, le=100.0)
    V20: float = Field(..., ge=-100.0, le=100.0)
    V21: float = Field(..., ge=-100.0, le=100.0)
    V22: float = Field(..., ge=-100.0, le=100.0)
    V23: float = Field(..., ge=-100.0, le=100.0)
    V24: float = Field(..., ge=-100.0, le=100.0)
    V25: float = Field(..., ge=-100.0, le=100.0)
    V26: float = Field(..., ge=-100.0, le=100.0)
    V27: float = Field(..., ge=-100.0, le=100.0)
    V28: float = Field(..., ge=-100.0, le=100.0)
    Amount: float = Field(..., ge=0.0, description="Transaction amount in currency units")
    Time: Optional[float] = Field(default=None, description="Seconds elapsed since first transaction (ignored by model)")

    @model_validator(mode="after")
    def check_no_nan(self) -> "TransactionInput":
        for field_name, value in self.__dict__.items():
            if isinstance(value, float) and np.isnan(value):
                raise ValueError(f"Field '{field_name}' must not be NaN")
        return self


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    threshold: float


class MetricsResponse(BaseModel):
    total_predictions: int

# ---------------------------------------------------------------------------
# App + model loading
# ---------------------------------------------------------------------------

app = FastAPI(title="Fraud Detection API")

_model_path = os.getenv("MODEL_PATH", "models:/FraudDetectionModel@champion")
_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _tracking_uri:
    mlflow.set_tracking_uri(_tracking_uri)

logger.info("Loading model", extra={"model_path": _model_path})
try:
    model = mlflow.pyfunc.load_model(_model_path)
    logger.info("Model loaded successfully", extra={"model_path": _model_path})
except Exception as exc:
    logger.exception("Failed to load model", extra={"model_path": _model_path, "error": str(exc)})
    raise

_scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")
logger.info("Loading scaler", extra={"scaler_path": _scaler_path})
try:
    scaler = joblib.load(_scaler_path)
    logger.info("Scaler loaded successfully", extra={"scaler_path": _scaler_path})
except Exception as exc:
    logger.exception("Failed to load scaler", extra={"scaler_path": _scaler_path, "error": str(exc)})
    raise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_prediction_input(data: dict) -> None:
    os.makedirs("monitoring", exist_ok=True)
    df = pd.DataFrame([data])
    write_header = not os.path.exists(CURRENT_DATA_PATH)
    df.to_csv(CURRENT_DATA_PATH, mode="a", header=write_header, index=False)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput) -> PredictionResponse:
    global _prediction_count

    input_data = transaction.model_dump(exclude={"Time"})
    input_df = pd.DataFrame([input_data])[FEATURE_ORDER].astype(float)
    input_df["Amount"] = scaler.transform(input_df[["Amount"]])

    logger.info(
        "Prediction request received",
        extra={"amount": transaction.Amount},
    )

    try:
        probability = float(
            model._model_impl.sklearn_model.predict_proba(input_df)[0][1]
        )
    except Exception as exc:
        logger.exception("Model inference failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    prediction = int(probability > THRESHOLD)

    with _count_lock:
        _prediction_count += 1

    logger.info(
        "Prediction complete",
        extra={
            "fraud_probability": probability,
            "prediction": prediction,
            "total_predictions": _prediction_count,
        },
    )

    try:
        _save_prediction_input(input_df.iloc[0].to_dict())
    except Exception as exc:
        # Non-fatal — monitoring write failure should not fail the request
        logger.warning("Failed to save prediction input for monitoring", extra={"error": str(exc)})

    return PredictionResponse(fraud_probability=probability, prediction=prediction)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="FraudDetectionModel",
        version="champion",
        threshold=THRESHOLD,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    with _count_lock:
        count = _prediction_count
    return MetricsResponse(total_predictions=count)
