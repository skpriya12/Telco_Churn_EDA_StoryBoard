# Telco Customer Churn — EDA Storyboard (12 Insights + 3 Hypotheses)


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

# ----------------------------
# Settings
# ----------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)

# ----------------------------
# Load data
# ----------------------------
# Update path to where you saved the Kaggle file
# Typical Kaggle file name: "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# ----------------------------
# Basic cleaning
# ----------------------------
# Strip whitespace in column names (just in case)
df.columns = [c.strip() for c in df.columns]

# Convert TotalCharges to numeric (it sometimes has blanks/spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Convert SeniorCitizen to category labels for nicer plots
if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Ensure Churn is consistent
df["Churn"] = df["Churn"].astype(str)


# ----------------------------
# Helper functions
# ----------------------------
def churn_rate_by(group_col: str) -> pd.DataFrame:
    """Return churn rate table for a categorical column."""
    tab = (
        df.groupby(group_col)["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
        .rename("ChurnRate")
        .reset_index()
    )
    tab["ChurnRate"] = (tab["ChurnRate"] * 100).round(2)
    return tab

def show():
    plt.tight_layout()
    plt.show()
# ----------------------------
# Insight 1 — Data overview & missingness
# ----------------------------
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))
print("\nDtypes:\n", df.dtypes)

sns.heatmap(df.isnull(), cbar=False)
plt.title("Insight 1: Missing Values Heatmap")
show()

# ----------------------------
# Insight 2 — Distribution of MonthlyCharges
# ----------------------------
sns.histplot(df["MonthlyCharges"], kde=True)
plt.title("Insight 2: Distribution of Monthly Charges")
plt.xlabel("MonthlyCharges")
show()
# ----------------------------
# Insight 3 — Distribution of TotalCharges (with NaNs)
# ----------------------------
sns.histplot(df["TotalCharges"].dropna(), kde=True)
plt.title("Insight 3: Distribution of Total Charges")
plt.xlabel("TotalCharges")
show()

# ----------------------------
# Insight 4 — Overall churn breakdown
# ----------------------------
sns.countplot(x="Churn", data=df)
plt.title("Insight 4: Churn Distribution")
show()

# ----------------------------
# Insight 5 — Churn by Contract Type
# ----------------------------
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Insight 5: Churn by Contract Type")
plt.xticks(rotation=15)
show()

print("\nChurn rate by Contract:\n", churn_rate_by("Contract"))
# ----------------------------
# Insight 6 — Tenure distribution + churn separation
# ----------------------------
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, element="step")
plt.title("Insight 6: Tenure Distribution by Churn")
plt.xlabel("Tenure (months)")
show()

# Create tenure buckets for segmented view
df["TenureBucket"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"],
    include_lowest=True
)

sns.countplot(x="TenureBucket", hue="Churn", data=df)
plt.title("Insight 6b: Churn by Tenure Bucket")
show()

# ----------------------------
# Insight 7 — MonthlyCharges vs Churn
# ----------------------------
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Insight 7: Monthly Charges by Churn")
show()

# ----------------------------
# Insight 8 — Payment method risk
# ----------------------------
sns.countplot(y="PaymentMethod", hue="Churn", data=df)
plt.title("Insight 8: Churn by Payment Method")
show()

print("\nChurn rate by PaymentMethod:\n", churn_rate_by("PaymentMethod"))

# ----------------------------
# Insight 9 — InternetService segments & churn
# ----------------------------
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Insight 9: Churn by Internet Service Type")
plt.xticks(rotation=15)
show()

print("\nChurn rate by InternetService:\n", churn_rate_by("InternetService"))

# ----------------------------
# Insight 10 — Support features & churn (TechSupport, OnlineSecurity)
# ----------------------------
for col in ["TechSupport", "OnlineSecurity"]:
    if col in df.columns:
        sns.countplot(x=col, hue="Churn", data=df)
        plt.title(f"Insight 10: Churn by {col}")
        plt.xticks(rotation=15)
        show()
        print(f"\nChurn rate by {col}:\n", churn_rate_by(col))

# ----------------------------
# Insight 11 — PaperlessBilling and churn
# ----------------------------
sns.countplot(x="PaperlessBilling", hue="Churn", data=df)
plt.title("Insight 11: Churn by Paperless Billing")
plt.xticks(rotation=15)
show()

print("\nChurn rate by PaperlessBilling:\n", churn_rate_by("PaperlessBilling"))

# ----------------------------
# Insight 12 — Correlation snapshot for numeric fields
# ----------------------------
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
corr = df[num_cols].corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Insight 12: Correlation Heatmap (Numeric Features)")
show()


# ============================================================
# Hypotheses (2–3)
# ============================================================

# ----------------------------
# Hypothesis 1: Customers using more services churn less
# Build a simple ServiceCount from service columns (Yes/No)
# ----------------------------
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

# Some columns contain 'No internet service' or 'No phone service' → treat as 0
def yes_to_one(x):
    return 1 if x == "Yes" else 0

service_matrix = pd.DataFrame()
for c in service_cols:
    if c in df.columns:
        service_matrix[c] = df[c].apply(yes_to_one)

df["ServiceCount"] = service_matrix.sum(axis=1)

sns.boxplot(x="Churn", y="ServiceCount", data=df)
plt.title("Hypothesis 1: Service Count vs Churn")
show()

retained = df.loc[df["Churn"] == "No", "ServiceCount"].dropna()
churned = df.loc[df["Churn"] == "Yes", "ServiceCount"].dropna()
t1 = stats.ttest_ind(retained, churned, equal_var=False)
print("\nHypothesis 1 t-test (ServiceCount retained vs churned):", t1)

# ----------------------------
# Hypothesis 2: Month-to-month contracts have higher churn
# Chi-square test on Contract vs Churn
# ----------------------------
contingency = pd.crosstab(df["Contract"], df["Churn"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\nHypothesis 2 Chi-square (Contract vs Churn):")
print("chi2:", chi2, "p:", p, "dof:", dof)
print("contingency:\n", contingency)

# ----------------------------
# Hypothesis 3: High monthly charges increase early churn risk (tenure <= 6)
# KDE distribution by churn for early-tenure customers
# ----------------------------
early = df[df["tenure"] <= 6].copy()

sns.kdeplot(data=early, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False)
plt.title("Hypothesis 3: Early Tenure (<=6m) MonthlyCharges by Churn")
show()

# Optional: compare early churned vs retained monthly charges (t-test)
early_retained = early.loc[early["Churn"] == "No", "MonthlyCharges"].dropna()
early_churned = early.loc[early["Churn"] == "Yes", "MonthlyCharges"].dropna()
t3 = stats.ttest_ind(early_retained, early_churned, equal_var=False)
print("\nHypothesis 3 t-test (Early MonthlyCharges retained vs churned):", t3)

# ----------------------------
# Summary tables (optional)
# ----------------------------
print("\nTop churn rates by segment (Contract, InternetService, PaymentMethod):")
print("\nContract:\n", churn_rate_by("Contract"))
print("\nInternetService:\n", churn_rate_by("InternetService"))
print("\nPaymentMethod:\n", churn_rate_by("PaymentMethod"))