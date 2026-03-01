import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

from data_loader import load_data


def train():
    # Load data
    df = load_data("/Users/mk004/Documents/GH/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split data (important: stratify for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # model --> LR 
    print("\n===== Logistic Regression =====")

    log_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced'))
    ])

    log_model.fit(X_train, y_train)

    y_pred_log = log_model.predict(X_test)
    y_proba_log = log_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_log))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))
    print("PR-AUC:", average_precision_score(y_test, y_proba_log))

    # model --> RF
    
    print("\n===== Random Forest =====")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
    print("PR-AUC:", average_precision_score(y_test, y_proba_rf))

    # Save model
    joblib.dump(rf_model, "models/fraud_model.pkl")


if __name__ == "__main__":
    train()



