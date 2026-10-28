# ============================
# Customer Churn Prediction
# ============================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def save_or_show_plot(fig, filename, title):
    output_dir = project_root() / "plots"
    output_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def main():
    project_dir = project_root()
    dataset_path = project_dir / "customer_churn.csv"

    df = pd.read_csv(dataset_path)

    print("First 5 Rows:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    df = df.drop_duplicates().copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Churn Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Churn", data=df, ax=ax)
    ax.set_title("Customer Churn Distribution")
    save_or_show_plot(fig, "churn_distribution.png", "Customer Churn Distribution")

    # Monthly Charges vs Churn
    if "MonthlyCharge" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="Churn", y="MonthlyCharge", data=df, ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        save_or_show_plot(fig, "monthly_charges_vs_churn.png", "Monthly Charges vs Churn")

    # Contract Type vs Churn
    if "ContractRenewal" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(x="ContractRenewal", hue="Churn", data=df, ax=ax)
        ax.set_title("Contract Renewal vs Churn")
        save_or_show_plot(fig, "contract_vs_churn.png", "Contract Renewal vs Churn")

    # Tenure vs Churn
    if "AccountWeeks" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="Churn", y="AccountWeeks", data=df, ax=ax)
        ax.set_title("Account Weeks vs Churn")
        save_or_show_plot(fig, "account_weeks_vs_churn.png", "Account Weeks vs Churn")

    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = label_encoder.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    def print_model_metrics(name, y_true, y_pred):
        print(f"\n========== {name} ==========")
        print("Accuracy :", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred, zero_division=0))
        print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    print_model_metrics("Logistic Regression", y_test, pred_lr)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    print_model_metrics("Random Forest", y_test, pred_rf)

    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    print_model_metrics("XGBoost", y_test, pred_xgb)

    corr_fig, corr_ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=corr_ax)
    corr_ax.set_title("Correlation Heatmap")
    save_or_show_plot(corr_fig, "correlation_heatmap.png", "Correlation Heatmap")

    print("\nProject Completed Successfully!")


if __name__ == "__main__":
    main()

# ==========================================
# GRAPHS AND TABLES FOR CUSTOMER CHURN PROJECT
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a folder to save outputs
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("customer_churn.csv")

sns.set_theme(style="whitegrid")

# 1. TABLE: First 10 customers
print("\n===== CUSTOMER DATA TABLE =====")
print(df.head(10).to_string(index=False))
df.head(10).to_csv("outputs/customer_data_table.csv", index=False)

# 2. GRAPH: Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/churn_distribution.png", dpi=300)
plt.show()

# 3. TABLE: Model performance comparison
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy (%)": [86.06, 92.65, 92.65],
    "Precision (%)": [64.29, 86.11, 85.14],
    "Recall (%)": [17.82, 61.39, 62.38]
})

print("\n===== MODEL PERFORMANCE TABLE =====")
print(results.to_string(index=False))
results.to_csv("outputs/model_performance.csv", index=False)

# 4. GRAPH: Model accuracy comparison
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results, x="Model", y="Accuracy (%)")
ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 105)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_accuracy.png", dpi=300)
plt.show()

# 5. GRAPH: Precision and recall comparison
metrics = results.melt(
    id_vars="Model",
    value_vars=["Precision (%)", "Recall (%)"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(9, 5))
sns.barplot(data=metrics, x="Model", y="Score", hue="Metric")
plt.title("Model Precision and Recall Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("outputs/precision_recall.png", dpi=300)
plt.show()

# 6. GRAPH: Churn by contract renewal
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="ContractRenewal", hue="Churn")
plt.title("Customer Churn by Contract Renewal")
plt.xlabel("Contract Renewal (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/churn_by_contract.png", dpi=300)
plt.show()

print("\nAll graphs and tables saved in the outputs folder!")
