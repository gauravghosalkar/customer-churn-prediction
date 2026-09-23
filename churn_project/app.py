"""
app.py
------
Streamlit web app for real-time customer churn prediction.
Loads the trained XGBoost model + encoders and lets a user enter a
customer's details to get an instant churn prediction + probability.

Run: streamlit run app.py
(Make sure you have already run `python src/train.py` at least once,
so model/churn_model.pkl and model/encoders.pkl exist.)
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

MODEL_PATH = "model/churn_model.pkl"
ENCODERS_PATH = "model/encoders.pkl"


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    # SHAP's TreeExplainer works directly on the trained XGBoost model and
    # is cached alongside it so it's only built once per session, not on
    # every click of "Predict Churn"
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle["model"], bundle["feature_names"], encoders, explainer


def encode_input(raw: dict, encoders: dict, feature_names: list) -> pd.DataFrame:
    row = {}
    for col in feature_names:
        if col in encoders:  # categorical column -> label encode
            le = encoders[col]
            val = str(raw[col])
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = le.transform([val])[0]
        else:  # numeric column, used as-is
            row[col] = raw[col]
    return pd.DataFrame([row], columns=feature_names)


def main():
    st.title("📉 Customer Churn Predictor")
    st.caption("Predicts whether a telecom customer is likely to leave the service, using an XGBoost model.")

    try:
        model, feature_names, encoders, explainer = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run `python src/train.py` first "
            "to train the model and generate model/churn_model.pkl."
        )
        return

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    if st.button("Predict Churn", type="primary"):
        raw = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        X = encode_input(raw, encoders, feature_names)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ This customer is likely to CHURN (probability: {proba:.1%})")
        else:
            st.success(f"✅ This customer is likely to STAY (churn probability: {proba:.1%})")

        st.progress(float(proba))

        # --- SHAP explainability: WHY did the model predict this? ---
        st.subheader("Why did the model predict this?")
        st.caption(
            "Each bar below shows how much a factor pushed this specific "
            "customer's prediction toward CHURN (red, pointing right) or "
            "toward STAY (blue, pointing left), starting from the average "
            "prediction across all customers."
        )
        shap_values = explainer(X)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False, max_display=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
