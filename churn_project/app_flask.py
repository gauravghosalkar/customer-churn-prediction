"""
app_flask.py
------------
Optional Flask REST API alternative to the Streamlit app.
Exposes a single POST /predict endpoint that accepts customer JSON and
returns a churn prediction.

Run:    python app_flask.py
Test:   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
"""

import pickle

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = "model/churn_model.pkl"
ENCODERS_PATH = "model/encoders.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model, feature_names = bundle["model"], bundle["feature_names"]

with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)


def encode_input(raw: dict) -> pd.DataFrame:
    row = {}
    for col in feature_names:
        if col in encoders:
            le = encoders[col]
            val = str(raw.get(col, le.classes_[0]))
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = le.transform([val])[0]
        else:
            row[col] = raw.get(col, 0)
    return pd.DataFrame([row], columns=feature_names)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Churn prediction API is running. POST customer JSON to /predict."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X = encode_input(data)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return jsonify({
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(proba, 4),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
