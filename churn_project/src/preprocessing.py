"""
preprocessing.py
-----------------
Loads the raw Telco churn CSV and returns a fully cleaned, numeric,
model-ready DataFrame. Also saves the fitted LabelEncoders so the
Streamlit app can encode new user input the exact same way.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop identifier column - it has zero predictive value
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2. TotalCharges is stored as text in the raw file and has blank
    #    strings for the ~11 customers with tenure == 0. Coerce to numeric
    #    and fill those with 0 (they haven't been billed yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 3. Drop any fully-empty rows / duplicate rows
    df = df.drop_duplicates()

    # 4. Target column: Yes/No -> 1/0
    #    (checked by value rather than dtype, since pandas may load this
    #    column as object OR as the newer nullable "string" dtype)
    if set(df["Churn"].astype(str).unique()) <= {"Yes", "No"}:
        df["Churn"] = df["Churn"].astype(str).map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encodes every categorical column.
    If fit=True, creates and returns new encoders.
    If fit=False, uses the provided encoders (for inference on new data).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle unseen categories gracefully at inference time
            df[col] = df[col].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def preprocess(path: str, save_encoders_path: str = None):
    """
    Full pipeline: raw CSV -> cleaned -> encoded DataFrame + encoders dict.
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df, encoders = encode_features(df, fit=True)

    if save_encoders_path:
        os.makedirs(os.path.dirname(save_encoders_path), exist_ok=True)
        with open(save_encoders_path, "wb") as f:
            pickle.dump(encoders, f)

    return df, encoders


if __name__ == "__main__":
    df, encoders = preprocess(
        "../data/telco_churn.csv",
        save_encoders_path="../model/encoders.pkl",
    )
    print(df.head())
    print("\nShape after cleaning + encoding:", df.shape)
    print("\nMissing values remaining:\n", df.isnull().sum().sum())
