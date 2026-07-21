# Customer Churn Prediction — Working with a Business Problem

## Overview
This project predicts customer churn using the **Telco Customer Churn**
dataset (IBM Sample Dataset, 7,043 customers, 21 features). Two models
were trained and compared: **Logistic Regression** and a **Decision
Tree Classifier**.

## Dataset
- Source: [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 rows, 21 columns
- Target: `Churn` (Yes/No — whether the customer left)

## Exploratory Data Analysis (Key Findings)

- **Overall churn rate:** 26.5% of customers churned, 73.5% stayed.
- **Contract type matters a lot:** Month-to-month customers churn at
  42.7%, compared to 11.3% for one-year contracts and just 2.8% for
  two-year contracts.
- **Tenure matters:** New customers (0-12 months) churn at 47.7%,
  while long-term customers (49-72 months) churn at only 9.5%.
- **Pricing:** Churned customers pay higher average monthly charges
  ($74.44) than customers who stayed ($61.27).

## Handling Categorical Variables & Class Imbalance

- All categorical columns (Contract, InternetService, PaymentMethod,
  etc.) were encoded using `pd.get_dummies()` with `drop_first=True`.
- `TotalCharges` had a few blank values (new customers with 0 months
  tenure) — converted to numeric and filled with the median.
- **Class imbalance:** The target is imbalanced (~73% stayed vs ~27%
  churned). This was **not fully corrected** in this task (e.g. no
  SMOTE or oversampling applied) — it's flagged here as a known
  limitation. Because of the imbalance, accuracy alone would
  overstate how good the model really is, so Precision, Recall, and
  F1-score are reported for the minority ("Churn") class.

## Model Comparison

| Model                | Accuracy | Precision | Recall | F1-score |
|-----------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression   | 80.5%    | 65.5%     | 55.9%  | 60.3%    |
| Decision Tree         | 79.4%    | 63.0%     | 54.5%  | 58.5%    |

Logistic Regression slightly outperforms the Decision Tree across all
metrics in this case. Both models catch a bit over half of actual
churners (Recall ~55-56%) — there's clear room for improvement (e.g.
addressing class imbalance, trying ensemble models like Random
Forest).

## Top 3 Features Driving Churn (from Decision Tree `.feature_importances_`)

1. **Tenure** (42.1% importance) — how long the customer has been
   with the company. Newer customers are far more likely to churn.
2. **Internet Service: Fiber optic** (35.8% importance) — customers
   with fiber optic internet churn more than DSL customers.
3. **Total Charges** (4.7% importance) — total amount billed to date.

## Business Summary

*(As if presenting to a non-technical manager)*

Our analysis shows that customer churn is heavily driven by contract
length and how new the customer is — customers on month-to-month
plans are roughly 15 times more likely to leave than those on
two-year contracts, and churn is highest in a customer's first year.
Customers with fiber optic internet also churn noticeably more,
which may point to service quality or pricing concerns worth
investigating. Our predictive model correctly flags actual churners
about 66% of the time when it raises a flag, and catches around 56%
of all customers who eventually leave — a solid starting point, though
not yet a complete solution, since churners are a minority of our
customer base and harder to detect. We recommend prioritizing
retention outreach for new, month-to-month, fiber optic customers,
and digging deeper into why fiber optic service correlates with
higher churn. With further refinement, this model can become a
practical, proactive tool for identifying at-risk customers before
they leave.

## Files
- `churn_prediction.ipynb` — full pipeline: EDA, cleaning, encoding,
  model training, comparison, feature importance.
- `telco_churn.csv` — dataset used.
- `README.md` — this file.
