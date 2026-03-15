
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

    # Scale Amount column
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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

    # -----------------------------------------
    # Logistic Regression
    # -----------------------------------------
    print("\n===== Logistic Regression =====")

    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train_resampled, y_train_resampled)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, lr_pred))
    print("ROC-AUC:", roc_auc_score(y_test, lr_prob))
    print("PR-AUC:", average_precision_score(y_test, lr_prob))

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

    print(classification_report(y_test, rf_pred))
    print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
    print("PR-AUC:", average_precision_score(y_test, rf_prob))

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

    print(classification_report(y_test, xgb_pred))
    print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))
    print("PR-AUC:", average_precision_score(y_test, xgb_prob))

    # -----------------------------------------
    # Save model
    # -----------------------------------------
    print("\nSaving best model...")

    os.makedirs("models", exist_ok=True)

    joblib.dump(xgb_model, "models/fraud_model.pkl")

    print("Model saved at models/fraud_model.pkl")


# -----------------------------------------
# Run training
# -----------------------------------------
if __name__ == "__main__":
    train()