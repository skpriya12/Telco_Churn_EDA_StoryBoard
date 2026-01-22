# Telco Customer Churn — EDA Storyboard (12 Insights + 3 Hypotheses)
# This version SAVES all charts to /visuals for GitHub Actions + portfolio use

import os
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
# Output directory for charts
# ----------------------------
OUTPUT_DIR = "visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_show(filename):
    """Save figure to visuals/ and close plot (CI-safe)."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ----------------------------
# Load data
# ----------------------------
CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_PATH)

# ----------------------------
# Basic cleaning
# ----------------------------
df.columns = [c.strip() for c in df.columns]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

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

# ----------------------------
# Insight 1 — Data overview & missingness
# ----------------------------
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))
print("\nDtypes:\n", df.dtypes)

sns.heatmap(df.isnull(), cbar=False)
plt.title("Insight 1: Missing Values Heatmap")
save_show("01_missing_values.png")

# ----------------------------
# Insight 2 — Distribution of MonthlyCharges
# ----------------------------
sns.histplot(df["MonthlyCharges"], kde=True)
plt.title("Insight 2: Distribution of Monthly Charges")
plt.xlabel("MonthlyCharges")
save_show("02_monthly_charges_dist.png")

# ----------------------------
# Insight 3 — Distribution of TotalCharges
# ----------------------------
sns.histplot(df["TotalCharges"].dropna(), kde=True)
plt.title("Insight 3: Distribution of Total Charges")
plt.xlabel("TotalCharges")
save_show("03_total_charges_dist.png")

# ----------------------------
# Insight 4 — Overall churn breakdown
# ----------------------------
sns.countplot(x="Churn", data=df)
plt.title("Insight 4: Churn Distribution")
save_show("04_churn_distribution.png")

# ----------------------------
# Insight 5 — Churn by Contract Type
# ----------------------------
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Insight 5: Churn by Contract Type")
plt.xticks(rotation=15)
save_show("05_churn_by_contract.png")

print("\nChurn rate by Contract:\n", churn_rate_by("Contract"))

# ----------------------------
# Insight 6 — Tenure distribution + churn separation
# ----------------------------
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, element="step")
plt.title("Insight 6: Tenure Distribution by Churn")
plt.xlabel("Tenure (months)")
save_show("06_tenure_distribution.png")

# Tenure buckets
df["TenureBucket"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"],
    include_lowest=True
)

sns.countplot(x="TenureBucket", hue="Churn", data=df)
plt.title("Insight 6b: Churn by Tenure Bucket")
save_show("07_churn_by_tenure_bucket.png")

# ----------------------------
# Insight 7 — MonthlyCharges vs Churn
# ----------------------------
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Insight 7: Monthly Charges by Churn")
save_show("08_monthly_charges_vs_churn.png")

# ----------------------------
# Insight 8 — Payment method risk
# ----------------------------
sns.countplot(y="PaymentMethod", hue="Churn", data=df)
plt.title("Insight 8: Churn by Payment Method")
save_show("09_churn_by_payment_method.png")

print("\nChurn rate by PaymentMethod:\n", churn_rate_by("PaymentMethod"))

# ----------------------------
# Insight 9 — InternetService segments & churn
# ----------------------------
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Insight 9: Churn by Internet Service Type")
plt.xticks(rotation=15)
save_show("10_churn_by_internet_service.png")

print("\nChurn rate by InternetService:\n", churn_rate_by("InternetService"))

# ----------------------------
# Insight 10 — Support features & churn
# ----------------------------
for col in ["TechSupport", "OnlineSecurity"]:
    if col in df.columns:
        sns.countplot(x=col, hue="Churn", data=df)
        plt.title(f"Insight 10: Churn by {col}")
        plt.xticks(rotation=15)
        save_show(f"11_churn_by_{col.lower()}.png")
        print(f"\nChurn rate by {col}:\n", churn_rate_by(col))

# ----------------------------
# Insight 11 — PaperlessBilling and churn
# ----------------------------
sns.countplot(x="PaperlessBilling", hue="Churn", data=df)
plt.title("Insight 11: Churn by Paperless Billing")
plt.xticks(rotation=15)
save_show("12_churn_by_paperless_billing.png")

print("\nChurn rate by PaperlessBilling:\n", churn_rate_by("PaperlessBilling"))

# ----------------------------
# Insight 12 — Correlation snapshot for numeric fields
# ----------------------------
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
corr = df[num_cols].corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Insight 12: Correlation Heatmap (Numeric Features)")
save_show("13_correlation_heatmap.png")

# ============================================================
# Hypotheses
# ============================================================

# ----------------------------
# Hypothesis 1: More services → lower churn
# ----------------------------
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

def yes_to_one(x):
    return 1 if x == "Yes" else 0

service_matrix = pd.DataFrame()
for c in service_cols:
    if c in df.columns:
        service_matrix[c] = df[c].apply(yes_to_one)

df["ServiceCount"] = service_matrix.sum(axis=1)

sns.boxplot(x="Churn", y="ServiceCount", data=df)
plt.title("Hypothesis 1: Service Count vs Churn")
save_show("14_service_count_vs_churn.png")

retained = df.loc[df["Churn"] == "No", "ServiceCount"].dropna()
churned = df.loc[df["Churn"] == "Yes", "ServiceCount"].dropna()
t1 = stats.ttest_ind(retained, churned, equal_var=False)
print("\nHypothesis 1 t-test (ServiceCount retained vs churned):", t1)

# ----------------------------
# Hypothesis 2: Contract type ↔ churn
# ----------------------------
contingency = pd.crosstab(df["Contract"], df["Churn"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\nHypothesis 2 Chi-square (Contract vs Churn):")
print("chi2:", chi2, "p:", p, "dof:", dof)
print("contingency:\n", contingency)

# ----------------------------
# Hypothesis 3: High charges → early churn
# ----------------------------
early = df[df["tenure"] <= 6].copy()

sns.kdeplot(data=early, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False)
plt.title("Hypothesis 3: Early Tenure MonthlyCharges by Churn")
save_show("15_early_tenure_pricing_risk.png")

early_retained = early.loc[early["Churn"] == "No", "MonthlyCharges"].dropna()
early_churned = early.loc[early["Churn"] == "Yes", "MonthlyCharges"].dropna()
t3 = stats.ttest_ind(early_retained, early_churned, equal_var=False)
print("\nHypothesis 3 t-test (Early MonthlyCharges retained vs churned):", t3)

# ----------------------------
# Summary tables
# ----------------------------
print("\nTop churn rates by segment (Contract, InternetService, PaymentMethod):")
print("\nContract:\n", churn_rate_by("Contract"))
print("\nInternetService:\n", churn_rate_by("InternetService"))
print("\nPaymentMethod:\n", churn_rate_by("PaymentMethod"))

print("\nEDA completed. Charts saved to /visuals")
