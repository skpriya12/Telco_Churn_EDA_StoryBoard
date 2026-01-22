# Telco Customer Churn — EDA Storyboard

## Overview
Business-focused exploratory data analysis (EDA) project designed to identify churn drivers, high-risk revenue segments, and actionable retention strategies for a telecom subscription business. The analysis combines visual storytelling with statistical hypothesis testing to simulate an executive-level analytics workflow.

---

## Business Objective
- Identify customer segments most likely to churn  
- Quantify revenue risk drivers  
- Recommend product, pricing, and operational levers to improve retention  

---

## Dataset
- Source: Kaggle — `blastchar/telco-customer-churn`  
- Size: 7,043 customers, 21 features  
- Note: 11 missing values in `TotalCharges` (new customers), excluded from lifetime revenue analysis

---

## Tech Stack
- Python
- Pandas
- Seaborn / Matplotlib
- SciPy
- Git / GitHub

---

## Repository Structure

---

## Key Insights
- Month-to-month customers churn at **42.7%** vs **2.8%** for two-year contracts  
- Electronic check users churn at **45.3%**, ~3× higher than autopay users  
- Fiber optic customers churn at **41.9%**, more than double DSL customers  
- Customers without TechSupport or OnlineSecurity churn at ~**42%** vs ~15% for users with these services  
- Early-tenure churned customers show significantly higher monthly charges (p < 1e-40)

---

## Hypotheses Validated
1. Customers who use more services churn less (p ≈ 5e-10)  
2. Churn is strongly associated with contract type (p ≈ 6e-258)  
3. Higher monthly charges increase early churn risk (p ≈ 1e-47)

---

## Business Recommendations
- Migrate month-to-month customers to annual plans  
- Promote autopay adoption to reduce billing friction  
- Bundle support and security features during onboarding  
- Target high-priced early-tenure customers with proactive retention offers  

---

## How to Run
```bash
pip install pandas seaborn matplotlib scipy kagglehub
python scripts/eda_storyboard.py
