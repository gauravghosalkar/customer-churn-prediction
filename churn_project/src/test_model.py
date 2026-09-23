"""
test_model.py
-------------
Quick sanity-test / validation script for the trained model.
Run this AFTER train.py to confirm the saved model loads correctly
and produces sensible predictions on a few sample customers.

Run: python test_model.py
"""

import pickle

import pandas as pd

MODEL_PATH = "../model/churn_model.pkl"
ENCODERS_PATH = "../model/encoders.pkl"

# A few representative test customers (mirroring real-world churn patterns)
TEST_CUSTOMERS = [
    {
        "label": "New customer, month-to-month, high charges -> expect HIGH churn risk",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
        },
    },
    {
        "label": "Long-tenure, two-year contract -> expect LOW churn risk",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
        },
    },
    {
        "label": "Mid-tenure, one-year contract -> borderline case",
        "data": {
            "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
            "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 1320.0,
        },
    },
]


def encode_input(raw, encoders, feature_names):
    row = {}
    for col in feature_names:
        if col in encoders:
            le = encoders[col]
            val = str(raw[col])
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = le.transform([val])[0]
        else:
            row[col] = raw[col]
    return pd.DataFrame([row], columns=feature_names)


def main():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, feature_names = bundle["model"], bundle["feature_names"]

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    print("Model loaded successfully.")
    print(f"Model expects {len(feature_names)} features.\n")

    for case in TEST_CUSTOMERS:
        X = encode_input(case["data"], encoders, feature_names)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        print("-" * 70)
        print(case["label"])
        print(f"  -> Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}  "
              f"(churn probability: {proba:.2%})")

    print("-" * 70)
    print("\nIf predictions align with the expected risk levels above, "
          "the model is working correctly.")


if __name__ == "__main__":
    main()
