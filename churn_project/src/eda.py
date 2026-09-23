"""
eda.py
------
Exploratory Data Analysis for the Telco churn dataset.
Generates and saves plots to ../images/ so they can be dropped straight
into a report or presentation.

Run: python eda.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / no display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

RAW_PATH = "../data/telco_churn.csv"
IMG_DIR = "../images"


def load_for_eda(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def plot_churn_distribution(df):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="Churn", hue="Churn", palette=["#2E7D32", "#C62828"], legend=False)
    ax.set_title("Overall Churn Distribution")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/churn_distribution.png", dpi=150)
    plt.close()


def plot_churn_by_contract(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", palette=["#2E7D32", "#C62828"])
    plt.title("Churn by Contract Type")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/churn_by_contract.png", dpi=150)
    plt.close()


def plot_tenure_vs_churn(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
                 palette=["#2E7D32", "#C62828"], bins=30)
    plt.title("Tenure Distribution by Churn")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/tenure_vs_churn.png", dpi=150)
    plt.close()


def plot_monthly_charges_vs_churn(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=["#2E7D32", "#C62828"], legend=False)
    plt.title("Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/monthly_charges_vs_churn.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    numeric_df = df.copy()
    for col in numeric_df.select_dtypes(include=["object","str"]).columns:
        numeric_df[col] = numeric_df[col].astype("category").cat.codes
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()


def run_eda():
    df = load_for_eda(RAW_PATH)
    print("Dataset shape:", df.shape)
    print("\nChurn rate:\n", df["Churn"].value_counts(normalize=True))
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_tenure_vs_churn(df)
    plot_monthly_charges_vs_churn(df)
    plot_correlation_heatmap(df)
    print(f"\nPlots saved to {IMG_DIR}/")


if __name__ == "__main__":
    run_eda()
