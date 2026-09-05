from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier



# project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_model_xgb.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"


# create models directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# load dataset
print(f"Loading dataset from: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# prep features and target
if "Class" not in df.columns:
    raise ValueError("The dataset must contain a 'Class' column.")

X = df.drop(columns=["Class"])
y = df["Class"]

feature_columns = list(X.columns)

# calculate class imbalance weight
negative_count = (y == 0).sum()
positive_count = (y == 1).sum()

scale_pos_weight = negative_count / positive_count

print(f"Legitimate transactions: {negative_count}")
print(f"Fraudulent transactions: {positive_count}")
print(f"Scale positive weight: {scale_pos_weight:.2f}")


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# create and train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


print("Training XGBoost model...")
model.fit(X_train, y_train)


# evaluate model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

pr_auc = average_precision_score(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Average Precision (PR-AUC): {pr_auc:.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")


# save model and feature columns
joblib.dump(model, MODEL_PATH)
joblib.dump(feature_columns, FEATURES_PATH)

print("\nFiles saved successfully:")
print(f"Model: {MODEL_PATH}")
print(f"Features: {FEATURES_PATH}")
