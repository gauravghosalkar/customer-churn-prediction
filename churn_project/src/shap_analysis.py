"""
shap_analysis.py
----------------
Adds model explainability on top of the trained XGBoost model using SHAP
(SHapley Additive exPlanations). While train.py already saves XGBoost's
built-in feature_importance.png, that only tells you WHICH features matter
on average. SHAP goes further and tells you:
  - WHICH DIRECTION each feature pushes a prediction (increases or
    decreases churn risk), not just how important it is
  - WHY a SPECIFIC customer was predicted to churn or not (per-customer
    explanation), which is what "explain reasons of churn" really means

Run this AFTER train.py (it loads the saved model + encoders).

Run: python shap_analysis.py
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from preprocessing import preprocess

MODEL_PATH = "../model/churn_model.pkl"
RAW_PATH = "../data/telco_churn.csv"
IMG_DIR = "../images"


def main():
    # 1. Load the trained model + rebuild the same test-style feature matrix
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, feature_names = bundle["model"], bundle["feature_names"]

    df, _ = preprocess(RAW_PATH)
    X = df.drop(columns=["Churn"])[feature_names]

    # Use a sample for speed on the summary plots (SHAP is exact for trees,
    # but explaining thousands of rows for a plot is unnecessary)
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)

    # 2. Build the SHAP explainer for the tree-based XGBoost model
    #    TreeExplainer is exact and fast for tree ensembles like XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # 3. GLOBAL EXPLANATION #1 — mean |SHAP value| bar chart
    #    "Which features matter most, on average, across all customers?"
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. GLOBAL EXPLANATION #2 — beeswarm plot
    #    Shows not just importance but DIRECTION: e.g. does high tenure
    #    push predictions toward "stay" or toward "churn"? Each dot is one
    #    customer; color = feature value (red=high, blue=low); x-position
    #    = impact on the churn prediction for that customer.
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.title("SHAP Summary — Feature Impact & Direction")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. LOCAL EXPLANATION — waterfall plots for a few individual customers
    #    "WHY was THIS specific customer predicted to churn?"
    #    We pick one high-risk and one low-risk customer as examples.
    probs = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = probs.argmax()
    low_risk_idx = probs.argmin()

    for label, idx in [("high_risk_example", high_risk_idx), ("low_risk_example", low_risk_idx)]:
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False, max_display=12)
        plt.title(f"Why this customer's churn prediction — {label.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("SHAP analysis complete. Saved to images/:")
    print("  - shap_bar_importance.png   (which features matter most, overall)")
    print("  - shap_beeswarm.png         (which features matter + which direction)")
    print("  - shap_waterfall_high_risk_example.png  (why one at-risk customer churns)")
    print("  - shap_waterfall_low_risk_example.png   (why one loyal customer stays)")


if __name__ == "__main__":
    main()
