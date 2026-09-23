"""
train.py
--------
Trains an XGBoost classifier to predict customer churn.

Pipeline:
  1. Load + clean + encode data       (preprocessing.py)
  2. 80/20 stratified train-test split
  3. Train XGBoost Classifier
  4. Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
  5. Plot + save feature importance
  6. Save the trained model with pickle for the Streamlit app

Run: python train.py
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocessing import preprocess

RAW_PATH = "../data/telco_churn.csv"
MODEL_PATH = "../model/churn_model.pkl"
ENCODERS_PATH = "../model/encoders.pkl"
IMG_DIR = "../images"


def main():
    # 1. Preprocess
    df, encoders = preprocess(RAW_PATH, save_encoders_path=ENCODERS_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    # 2. Train/test split — 80% train, 20% test, stratified on the target
    #    so both splits keep the same ~26.5% churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")

    # 3. Train XGBoost
    #    scale_pos_weight helps with the class imbalance (~70/30 split)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    with open("../model/metrics.txt", "w") as f:
        f.write("===== MODEL EVALUATION =====\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # 5. Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})", color="#C62828")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # 6. Feature importance
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", hue="feature",
                palette="viridis", legend=False)
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/feature_importance.png", dpi=150)
    plt.close()

    print("\nTop 5 most important features:")
    print(imp_df.head(5).to_string(index=False))

    # 7. Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Encoders saved to {ENCODERS_PATH}")
    print(f"Plots saved to {IMG_DIR}/")


if __name__ == "__main__":
    main()
