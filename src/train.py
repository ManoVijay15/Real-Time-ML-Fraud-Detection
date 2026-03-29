
import os
import pandas as pd
import joblib

from feast import FeatureStore

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import numpy as np

import mlflow
import mlflow.sklearn

from registry.mlflow_registry import register_model


# -----------------------------------------
# Load features from Feature Store
# -----------------------------------------
def load_features_from_store():

    print("Loading features from Feature Store...")

    store = FeatureStore(repo_path="fraud_feature_store/feature_repo")

    entity_df = pd.DataFrame({
        "transaction_id": range(284807),
        "event_timestamp": pd.to_datetime("now")
    })

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=[

            "fraud_features:V1","fraud_features:V2","fraud_features:V3","fraud_features:V4",
            "fraud_features:V5","fraud_features:V6","fraud_features:V7","fraud_features:V8",
            "fraud_features:V9","fraud_features:V10","fraud_features:V11","fraud_features:V12",
            "fraud_features:V13","fraud_features:V14","fraud_features:V15","fraud_features:V16",
            "fraud_features:V17","fraud_features:V18","fraud_features:V19","fraud_features:V20",
            "fraud_features:V21","fraud_features:V22","fraud_features:V23","fraud_features:V24",
            "fraud_features:V25","fraud_features:V26","fraud_features:V27","fraud_features:V28",
            "fraud_features:Amount"
        ]
    ).to_df()

    return feature_df


# -----------------------------------------
# Training pipeline
# -----------------------------------------
def train():

    # Load features
    df = load_features_from_store()

    # Remove timestamp
    df = df.drop(columns=["event_timestamp"], errors="ignore")

    print("Feature shape:", df.shape)

    # Load dataset
    data = pd.read_csv("/Users/mk004/Documents/GH/creditcard.csv")

    # Add transaction_id to dataset
    data["transaction_id"] = data.index

    # Merge labels correctly
    df = df.merge(
        data[["transaction_id", "Class"]],
        on="transaction_id"
    )

    # Separate features and label
    X = df.drop(columns=["Class", "transaction_id"])
    y = df["Class"]

    # Scale Amount column and persist scaler for inference
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save reference dataset for drift detection baseline
    os.makedirs("monitoring", exist_ok=True)
    reference_data = X_train.copy()
    reference_data["target"] = y_train.values
    reference_data.to_csv("monitoring/reference_data.csv", index=False)
    print("Reference dataset saved to monitoring/reference_data.csv")

    print("\nOriginal class distribution:")
    print(y_train.value_counts())

    # -----------------------------------------
    # SMOTE for class imbalance
    # -----------------------------------------
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE balancing:")
    print(pd.Series(y_train_resampled).value_counts())

    mlflow.set_experiment("FraudDetectionExperiment")

    with mlflow.start_run():

        # -----------------------------------------
        # Logistic Regression
        # -----------------------------------------
        print("\n===== Logistic Regression =====")

        lr_model = LogisticRegression(max_iter=1000)

        lr_model.fit(X_train_resampled, y_train_resampled)

        lr_pred = lr_model.predict(X_test)
        lr_prob = lr_model.predict_proba(X_test)[:, 1]

        lr_roc_auc = roc_auc_score(y_test, lr_prob)
        lr_pr_auc = average_precision_score(y_test, lr_prob)

        print(classification_report(y_test, lr_pred))
        print("ROC-AUC:", lr_roc_auc)
        print("PR-AUC:", lr_pr_auc)

        mlflow.log_metric("LogisticRegression_roc_auc", lr_roc_auc)
        mlflow.log_metric("LogisticRegression_pr_auc", lr_pr_auc)

        # -----------------------------------------
        # Random Forest
        # -----------------------------------------
        print("\n===== Random Forest =====")

        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            n_jobs=-1,
            random_state=42
        )

        rf_model.fit(X_train_resampled, y_train_resampled)

        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]

        rf_roc_auc = roc_auc_score(y_test, rf_prob)
        rf_pr_auc = average_precision_score(y_test, rf_prob)

        print(classification_report(y_test, rf_pred))
        print("ROC-AUC:", rf_roc_auc)
        print("PR-AUC:", rf_pr_auc)

        mlflow.log_metric("RandomForest_roc_auc", rf_roc_auc)
        mlflow.log_metric("RandomForest_pr_auc", rf_pr_auc)

        # -----------------------------------------
        # XGBoost
        # -----------------------------------------
        print("\n===== XGBoost =====")

        xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )

        xgb_model.fit(X_train_resampled, y_train_resampled)

        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

        xgb_roc_auc = roc_auc_score(y_test, xgb_prob)
        xgb_pr_auc = average_precision_score(y_test, xgb_prob)

        print(classification_report(y_test, xgb_pred))
        print("ROC-AUC:", xgb_roc_auc)
        print("PR-AUC:", xgb_pr_auc)

        mlflow.log_metric("XGBoost_roc_auc", xgb_roc_auc)
        mlflow.log_metric("XGBoost_pr_auc", xgb_pr_auc)

        # log params
        mlflow.log_param("lr_max_iter", 1000)
        mlflow.log_param("rf_n_estimators", 300)
        mlflow.log_param("rf_max_depth", 15)
        mlflow.log_param("xgb_n_estimators", 500)
        mlflow.log_param("xgb_max_depth", 6)
        mlflow.log_param("xgb_learning_rate", 0.05)

        # log best model artifact
        mlflow.sklearn.log_model(xgb_model, "model")

        run_id = mlflow.active_run().info.run_id

        # -----------------------------------------
        # Save model
        # -----------------------------------------
        print("\nSaving best model...")

        os.makedirs("models", exist_ok=True)

        joblib.dump(xgb_model, "models/fraud_model.pkl")

        print("Model saved at models/fraud_model.pkl")

    register_model(run_id)


# -----------------------------------------
# Train pipeline with MLflow experiment tracking
# -----------------------------------------
def train_pipeline(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
    }

    best_model = None
    best_score = 0
    best_model_name = None

    mlflow.set_experiment("FraudDetectionExperiment")

    with mlflow.start_run() as run:

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict_proba(X_test)[:,1]

            roc_auc = roc_auc_score(y_test, preds)
            pr_auc = average_precision_score(y_test, preds)

            print(f"{name} ROC-AUC:", roc_auc)
            print(f"{name} PR-AUC:", pr_auc)

            mlflow.log_metric(f"{name}_roc_auc", roc_auc)
            mlflow.log_metric(f"{name}_pr_auc", pr_auc)

            if pr_auc > best_score:
                best_score = pr_auc
                best_model = model
                best_model_name = name

        print("\nBest Model:", best_model_name)

        mlflow.log_param("best_model", best_model_name)

        mlflow.sklearn.log_model(best_model, "model")

        run_id = run.info.run_id

    register_model(run_id)


# -----------------------------------------
# Run training
# -----------------------------------------
if __name__ == "__main__":
    train()