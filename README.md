# Real-Time ML Fraud Detection System

A production-grade, end-to-end machine learning system for detecting fraudulent credit card transactions in real time. Built with a streaming-first architecture, automated drift monitoring, and a full MLOps pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.10-blue)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [Kafka Streaming](#kafka-streaming)
- [Monitoring & Auto-Retraining](#monitoring--auto-retraining)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Docker](#docker)
- [Roadmap](#roadmap)

---

## Overview

This system classifies credit card transactions as fraudulent or legitimate in real time using a trained XGBoost model. It is built around three core loops:

1. **Offline Training** — Feature engineering via Feast, SMOTE-balanced XGBoost training, and model versioning via MLflow.
2. **Online Inference** — A FastAPI REST API that serves predictions from the MLflow Model Registry with structured logging and input validation.
3. **Automated Monitoring** — Evidently-based data drift detection that automatically triggers model retraining when distribution shift exceeds a threshold.

The dataset used is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 fraud cases — ~0.17% positive rate).

---

## Architecture

```
┌──────────────────────── OFFLINE TRAINING ───────────────────────────┐
│                                                                      │
│   creditcard.csv ──► Feast Feature Store (V1–V28 PCA + Amount)      │
│                              │                                       │
│                    SMOTE Oversampling (imbalance fix)                │
│                              │                                       │
│         ┌────────────────────┼───────────────────┐                  │
│         ▼                    ▼                   ▼                   │
│  Logistic Regression   Random Forest         XGBoost ◄── Best       │
│         └────────────────────┼───────────────────┘                  │
│                              │                                       │
│                    MLflow Experiment Tracking                        │
│                    MLflow Model Registry                             │
│                    (alias: champion)                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ loads champion model
┌──────────────────────── ONLINE INFERENCE ───────────────────────────┐
│                                                                      │
│   Kafka Producer ──► Topic: fraud_transactions                      │
│                              │                                       │
│                       Kafka Consumer                                 │
│                   (fetches features from Feast)                      │
│                              │                                       │
│   ┌──────────────────────────▼──────────────────────────────────┐   │
│   │               FastAPI  /predict  (src/app.py)               │   │
│   │                                                             │   │
│   │   Input: V1–V28 + Amount    Output: fraud_probability,      │   │
│   │   Scale Amount (StandardScaler)      prediction (0/1)       │   │
│   │   XGBoost inference ──► threshold 0.7 ──► response          │   │
│   │   Log input to monitoring/current_data.csv                  │   │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────── MONITORING & AUTO-RETRAINING ───────────────────────┐
│                                                                      │
│   monitoring/current_data.csv  vs  monitoring/reference_data.csv    │
│                              │                                       │
│                  Evidently DataDriftPreset                           │
│                              │                                       │
│              drift_ratio > 30%? ──► trigger src/train.py            │
│                              │       (updates MLflow registry)      │
│                  monitoring/drift_report.html                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.129, Uvicorn, Pydantic v2 |
| **ML Model** | XGBoost 3.2 (500 estimators, max_depth=6) |
| **Class Imbalance** | SMOTE (imbalanced-learn 0.14) |
| **Feature Store** | Feast 0.61 (local SQLite + Parquet) |
| **Model Registry** | MLflow 3.10 (experiment tracking + versioning) |
| **Streaming** | Apache Kafka (kafka-python 2.3) |
| **Drift Detection** | Evidently 0.7 |
| **Logging** | python-json-logger (structured JSON) |
| **Metrics** | Prometheus client |
| **Containerization** | Docker (python:3.11-slim) |
| **CI/CD** | GitHub Actions → GitHub Container Registry (ghcr.io) |
| **Testing** | pytest, FastAPI TestClient |

---

## Project Structure

```
Real-Time-ML-Fraud-Detection/
├── src/
│   ├── app.py                     # FastAPI inference server
│   ├── train.py                   # ML training pipeline
│   ├── streaming/
│   │   ├── producer.py            # Kafka transaction producer
│   │   └── consumer.py            # Kafka consumer → Feast → /predict
│   ├── monitoring/
│   │   ├── drift_detection.py     # Evidently drift analysis
│   │   ├── retrain_trigger.py     # Auto-retrain on drift > 30%
│   │   └── reference_data_loader.py
│   ├── registry/
│   │   └── mlflow_registry.py     # MLflow model registration
│   └── features/                  # Feature engineering logic
│
├── fraud_feature_store/
│   └── feature_repo/
│       ├── feature_definitions.py # Feast feature views & entities
│       ├── feature_store.yaml     # Feast config (local SQLite)
│       └── data/                  # Parquet feature data
│
├── models/
│   ├── fraud_model.pkl            # Trained XGBoost artifact
│   └── scaler.pkl                 # StandardScaler for Amount
│
├── monitoring/
│   ├── reference_data.csv         # Training baseline for drift
│   ├── current_data.csv           # Live prediction log
│   └── drift_report.html          # Evidently HTML drift report
│
├── tests/
│   └── test_app.py                # 14+ API unit & integration tests
│
├── notebooks/                     # EDA Jupyter notebooks
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci-cd.yml    # GitHub Actions CI/CD
```

---

## Setup & Installation

**Prerequisites:** Python 3.11+, pip, (optional) Kafka broker running locally.

```bash
# Clone the repo
git clone https://github.com/<your-username>/Real-Time-ML-Fraud-Detection.git
cd Real-Time-ML-Fraud-Detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dataset:** Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at the path referenced in `src/train.py`.

---

## Training the Model

```bash
# Materialize features into the Feast online store
cd fraud_feature_store/feature_repo
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
cd ../..

# Run the full training pipeline
python src/train.py
```

This will:
1. Load 284,807 transactions from the Feast feature store
2. Apply SMOTE to balance the ~0.17% fraud class
3. Train Logistic Regression, Random Forest, and XGBoost
4. Log all metrics and parameters to MLflow (`FraudDetectionExperiment`)
5. Save `models/fraud_model.pkl` and `models/scaler.pkl`
6. Register XGBoost as `FraudDetectionModel@champion` in MLflow registry
7. Save `monitoring/reference_data.csv` as the drift baseline

View the MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Running the API

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`. On startup it:
- Loads `models:/FraudDetectionModel@champion` from MLflow
- Loads `models/scaler.pkl` for Amount normalization

---

## Kafka Streaming

Start the producer to emit transaction IDs to the `fraud_transactions` topic:

```bash
python src/streaming/producer.py
```

Start the consumer to read from Kafka, fetch features from Feast, and call `/predict`:

```bash
python src/streaming/consumer.py
```

> Requires a running Kafka broker on `localhost:9092`.

---

## Monitoring & Auto-Retraining

Run drift detection against the prediction log vs. training baseline:

```bash
python src/monitoring/drift_detection.py
```

This generates `monitoring/drift_report.html` with an Evidently drift report.

Trigger the auto-retraining check:

```bash
python src/monitoring/retrain_trigger.py
```

If more than **30%** of features show distribution drift, `src/train.py` is automatically invoked and the MLflow registry is updated with a new champion model.

---

## API Reference

### `GET /health`
Returns service health status.
```json
{ "status": "ok" }
```

### `GET /`
```json
{ "message": "Fraud Detection API is running" }
```

### `POST /predict`
Classify a transaction as fraudulent or legitimate.

**Request body:**
```json
{
  "V1": -1.36,
  "V2": -0.07,
  "V3": 2.54,
  ...
  "V28": 0.02,
  "Amount": 149.62,
  "Time": 0.0
}
```

- `V1`–`V28`: PCA-transformed features, range `[-100, 100]`
- `Amount`: Transaction amount ≥ 0 (scaled internally by the API)
- `Time`: Optional, ignored by the model

**Response:**
```json
{
  "fraud_probability": 0.03,
  "prediction": 0
}
```

- `prediction: 1` = fraud (probability > threshold, default 0.7)
- `prediction: 0` = legitimate

### `GET /model-info`
```json
{
  "model_name": "FraudDetectionModel",
  "version": "champion",
  "threshold": 0.7
}
```

### `GET /metrics`
```json
{ "total_predictions": 1042 }
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models:/FraudDetectionModel@champion` | MLflow model URI |
| `MLFLOW_TRACKING_URI` | _(local)_ | Remote MLflow tracking server URL |
| `SCALER_PATH` | `models/scaler.pkl` | Path to the fitted StandardScaler |
| `FRAUD_THRESHOLD` | `0.7` | Classification probability threshold |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Testing

```bash
pytest tests/ -v --tb=short
```

The test suite covers:
- Health and root endpoints
- Model info and metrics endpoints
- Prediction validation (valid inputs, out-of-range values, NaN checks)
- Fraud vs. legitimate classification based on threshold
- HTTP 422 error handling for malformed input

---

## CI/CD Pipeline

Defined in [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml).

### Job 1 — Lint & Test (every push/PR to `main`)
1. Set up Python 3.11 with pip caching
2. Install all dependencies from `requirements.txt`
3. Lint `src/` and `tests/` with flake8 (max line length 120)
4. Run pytest

### Job 2 — Build & Push Docker Image (push to `main` only)
1. Depends on Job 1 passing
2. Logs into GitHub Container Registry (`ghcr.io`)
3. Builds Docker image
4. Tags: `sha-<commit>` and `latest`
5. Pushes to `ghcr.io/<owner>/real-time-ml-fraud-detection`

---

## Docker

**Build locally:**
```bash
docker build -t fraud-detection-api .
```

**Run:**
```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e FRAUD_THRESHOLD=0.7 \
  fraud-detection-api
```

**Pull from GHCR:**
```bash
docker pull ghcr.io/<owner>/real-time-ml-fraud-detection:latest
```

---

## Roadmap

- [ ] `docker-compose.yml` for local Kafka + MLflow + API orchestration
- [ ] Scheduled drift checks via Airflow or cron
- [ ] Remote feature store (Snowflake / BigQuery) replacing local SQLite
- [ ] Remote MLflow tracking server with PostgreSQL backend
- [ ] Authentication & rate limiting on the `/predict` endpoint
- [ ] Grafana dashboard wired to Prometheus metrics
- [ ] Load testing with Locust

---

## License

MIT
