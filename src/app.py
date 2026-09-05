from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


# project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model_xgb.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"


# make sure required files are available
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Transaction fraud detection API is running",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "XGBoost"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Request body must contain JSON data"
        }), 400

    # find any features missing from the request
    missing_features = [
        feature for feature in feature_columns
        if feature not in data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing_features
        }), 400

    # arrange the input columns in the same order as during training
    input_df = pd.DataFrame(
        [[data[feature] for feature in feature_columns]],
        columns=feature_columns
    )

    probability = float(model.predict_proba(input_df)[0, 1])
    is_fraud = probability >= 0.5

    return jsonify({
        "fraud_probability": probability,
        "is_fraud": bool(is_fraud),
        "threshold": 0.5
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
