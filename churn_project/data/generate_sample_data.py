"""
generate_sample_data.py
------------------------
Creates a synthetic dataset that follows the EXACT column schema of the
official IBM Telco Customer Churn dataset.

WHY THIS FILE EXISTS:
This sandbox has no internet access, so the real IBM dataset could not be
downloaded automatically. This script builds a same-shaped, statistically
realistic stand-in (7043 rows, same 21 columns, same category values, and
churn correlated with tenure/contract/charges the same way it is in the
real data) so the whole pipeline runs end-to-end out of the box.

>>> HOW TO USE THE REAL IBM DATASET INSTEAD (recommended before submission) <<<
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   (This is IBM's sample dataset, mirrored on Kaggle.)
2. Download "WA_Fn-UseC_-Telco-Customer-Churn.csv"
3. Place it in this "data/" folder and rename it to: telco_churn.csv
4. Skip running this script — the training script will pick up the real file.

Run this script only if you want a working demo dataset immediately:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043  # same number of rows as the real IBM dataset

def choice(options, size, p=None):
    return np.random.choice(options, size=size, p=p)

gender = choice(["Male", "Female"], N)
senior_citizen = choice([0, 1], N, p=[0.84, 0.16])
partner = choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = np.random.randint(0, 73, N)

phone_service = choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    choice(["Yes", "No"], N, p=[0.42, 0.58])
)

internet_service = choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def dependent_internet_feature(p_yes=0.4):
    out = np.empty(N, dtype=object)
    for i in range(N):
        if internet_service[i] == "No":
            out[i] = "No internet service"
        else:
            out[i] = np.random.choice(["Yes", "No"], p=[p_yes, 1 - p_yes])
    return out

online_security = dependent_internet_feature(0.35)
online_backup = dependent_internet_feature(0.40)
device_protection = dependent_internet_feature(0.40)
tech_support = dependent_internet_feature(0.35)
streaming_tv = dependent_internet_feature(0.45)
streaming_movies = dependent_internet_feature(0.45)

contract = choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
paperless_billing = choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

# Monthly charges: base + add-ons
base = np.where(internet_service == "Fiber optic", 70,
        np.where(internet_service == "DSL", 45, 20)).astype(float)
addons = (
    (online_security == "Yes").astype(int) +
    (online_backup == "Yes").astype(int) +
    (device_protection == "Yes").astype(int) +
    (tech_support == "Yes").astype(int) +
    (streaming_tv == "Yes").astype(int) +
    (streaming_movies == "Yes").astype(int)
) * np.random.uniform(4, 8, N)
monthly_charges = np.round(base + addons + np.random.normal(0, 3, N), 2)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, N), 2)
total_charges = np.clip(total_charges, 0, None)
# Real dataset has blank TotalCharges for the 11 customers with tenure == 0
total_charges_str = total_charges.astype(str)
total_charges_str[tenure == 0] = " "

# ---- Build churn probability so it correlates realistically with known churn drivers ----
churn_score = (
    -0.04 * tenure
    + 0.015 * monthly_charges
    + np.where(contract == "Month-to-month", 1.4, np.where(contract == "One year", 0.3, -0.9))
    + np.where(internet_service == "Fiber optic", 0.5, 0.0)
    + np.where(payment_method == "Electronic check", 0.5, 0.0)
    + np.where(tech_support == "No", 0.3, 0.0)
    + np.where(online_security == "No", 0.3, 0.0)
    + np.where(senior_citizen == 1, 0.2, 0.0)
    + np.random.normal(0, 1.0, N)
)
churn_prob = 1 / (1 + np.exp(-churn_score + 2.2))  # shift to realistic ~26.5% churn rate
churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

customer_id = [f"{np.random.randint(1000,9999)}-{''.join(np.random.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(N)]

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges_str,
    "Churn": churn,
})

out_path = "telco_churn.csv"
df.to_csv(out_path, index=False)
print(f"Synthetic dataset written to {out_path}  |  shape={df.shape}")
print(df["Churn"].value_counts(normalize=True).rename("proportion"))
