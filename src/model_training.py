"""
Healthletics: AI-Powered Member Retention and Churn Prediction System

Reproducible Random Forest training script.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# File paths
DATA_PATH = "../data/gym_churn_us.csv"
MODEL_PATH = "../models/healthletics_random_forest_churn_model.joblib"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Feature engineering
df["frequency_change"] = (
    df["Avg_class_frequency_current_month"]
    - df["Avg_class_frequency_total"]
)

# Separate predictors and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize numerical predictors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Random Forest Performance")
print("-------------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Save model and scaler together
os.makedirs("../models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "features": list(X.columns)
    },
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")
