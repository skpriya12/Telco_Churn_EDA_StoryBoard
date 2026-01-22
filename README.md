
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
