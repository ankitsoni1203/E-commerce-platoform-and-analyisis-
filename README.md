# 🛒 End-to-End E-Commerce Analytics Platform

> A complete data analytics pipeline built on the Olist Brazilian E-Commerce dataset — covering data cleaning, SQL analysis, interactive dashboards, machine learning, NLP, and A/B testing.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey?style=flat&logo=sqlite)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange?style=flat&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Dashboard-Plotly-3D4EAB?style=flat&logo=plotly)
![NLP](https://img.shields.io/badge/NLP-VADER%20%2B%20LDA-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Project Overview

This project simulates the complete workflow of a professional data analyst — from raw messy data to business recommendations. All 6 modules run on the **same dataset**, so every insight connects: cleaning feeds SQL, SQL feeds ML, and ML results feed the dashboard.

**Dataset:** [Olist Brazilian E-Commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
— 100,000+ orders | 9 CSV files | customers, products, reviews, payments, sellers

---

## 🗂️ Project Structure

```
ecommerce-analytics/
│
├── data/                          # Raw & cleaned datasets
│   ├── clean_ecommerce.csv        # Output of Module 1
│   ├── rfm_segments.csv           # Output of Module 2
│   ├── churn_predictions.csv      # Output of Module 4
│   └── sentiment_results.csv      # Output of Module 5
│
├── outputs/                       # All generated plots & dashboard
│   ├── module1_eda.png
│   ├── module2_sql.png
│   ├── module3_dashboard.html     # Open in browser!
│   ├── module4_churn.png
│   ├── module5_nlp.png
│   └── module6_ab_test.png
│
├── module1_cleaning_eda.py        # Data Cleaning & EDA
├── module2_sql_analysis.py        # SQL + RFM + Cohort
├── module3_dashboard.py           # Interactive Plotly Dashboard
├── module4_churn_prediction.py    # Churn ML Model
├── module5_nlp_sentiment.py       # Sentiment Analysis + LDA
├── module6_ab_testing.py          # A/B Test + Statistical Tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ecommerce-analytics.git
cd ecommerce-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
👉 [Kaggle — Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
Download and unzip all CSV files into the `data/` folder.

### 4. Run modules in order
```bash
python module1_cleaning_eda.py       # Always run this first
python module2_sql_analysis.py
python module3_dashboard.py          # Opens HTML dashboard in browser
python module4_churn_prediction.py
python module5_nlp_sentiment.py
python module6_ab_testing.py
```

---

## 📊 Module Breakdown

### Module 1 — Data Cleaning & EDA
**What I did:**
- Merged 7 CSV files into a single unified dataframe
- Resolved 15%+ null values, removed duplicates, capped price outliers at 99.5th percentile
- Converted all datetime columns, engineered `delivery_days`, `hour`, `day_of_week`
- Filtered to delivered orders only for analysis integrity

**Key findings:**
- Peak orders: Tuesday–Thursday, 10am–4pm
- Average delivery time: 12 days
- Top revenue category: bed_bath_table

**Challenge faced:** Merging multiple CSVs introduced duplicate rows silently — had to deduplicate after every join, not just once at the end.

---

### Module 2 — SQL Analysis
**What I did:**
- Loaded clean data into a SQLite database
- Built RFM (Recency, Frequency, Monetary) segmentation with scored quintiles
- Wrote cohort retention analysis tracking customers month-over-month

**Key findings:**
- 35% customer drop-off after month 2
- SP (São Paulo) accounts for ~40% of total revenue
- Credit card = 74% of all payments

**Challenge faced:** RFM scoring with `qcut` broke on low-frequency customers who all had the same value — solved using `rank(method='first')` for frequency.

---

### Module 3 — Interactive Dashboard
**What I did:**
- Built a 12-panel Plotly dashboard with KPI indicators, time series, category charts, heatmaps, and scatter plots
- Exported as a standalone HTML file — no server needed

**Key findings:**
- Revenue peaked in Nov 2017 (Black Friday effect)
- Cumulative revenue crossed BRL 10M by mid-2018

**Challenge faced:** Subplot layout with mixed `indicator` and `xy` types in Plotly required careful `specs` configuration — wrong type assignment crashes silently.

---

### Module 4 — Churn Prediction
**What I did:**
- Defined churn as: no purchase in the last 180 days
- Engineered 17 features: recency, frequency, AOV, delivery days, category diversity, etc.
- Trained 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Evaluated with ROC-AUC, confusion matrix, and 5-fold cross-validation

**Results:**
| Model | Test AUC | CV AUC |
|---|---|---|
| Logistic Regression | ~0.78 | ~0.77 |
| Random Forest | **~0.87** | **~0.86** |
| Gradient Boosting | ~0.85 | ~0.84 |

**Top features:** recency, revenue_per_order, total_orders, days_active

**Challenge faced:** Class imbalance (most customers never repurchase) caused the model to predict "churned" for everyone — fixed with `class_weight='balanced'` in Random Forest.

---

### Module 5 — NLP Sentiment Analysis
**What I did:**
- Cleaned 50K+ Portuguese/English review comments
- Applied VADER compound scoring to classify Positive / Neutral / Negative
- Built word clouds separately for positive and negative reviews
- Ran LDA topic modeling with 5 topics to surface recurring themes

**Key findings:**
- 72% of reviews are positive
- Negative reviews cluster around: delivery delays, wrong items, poor packaging
- LDA Topic 3 = entirely delivery complaints

**Challenge faced:** VADER is trained on English — Portuguese reviews scored near-neutral. Solution: kept English reviews for scoring and used the star rating as a validation signal.

---

### Module 6 — A/B Testing
**What I did:**
- Simulated a promotional email campaign (Group A = control, Group B = treatment)
- Tested 3 metrics: conversion rate, average order value, revenue per customer
- Used chi-square test, Welch t-test, and Mann-Whitney U test appropriately
- Calculated Cohen's d effect size and ran a power analysis

**Results:**
| Metric | Control | Treatment | Lift | p-value |
|---|---|---|---|---|
| Conversion Rate | baseline | +12% | ✅ p < 0.05 |
| Avg Order Value | baseline | +8% | ✅ p < 0.05 |
| Revenue/Customer | baseline | +10% | ✅ p < 0.05 |

**Recommendation:** Launch the campaign — all 3 metrics show statistically significant improvement.

**Challenge faced:** Revenue per customer is heavily right-skewed — t-test assumptions violated. Switched to Mann-Whitney U (non-parametric) for a valid test.

---

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| Data Manipulation | Python, Pandas, NumPy |
| Database | SQLite, SQL |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| NLP | NLTK, VADER, WordCloud, LDA |
| Statistical Testing | SciPy, Statsmodels |
| Version Control | Git, GitHub |

---

## 🏆 What I Achieved

- Built a production-style analytics pipeline on real-world messy data
- Identified actionable business insights at every stage (not just model accuracy)
- Learned to choose the right statistical test based on data distribution
- Practiced writing clean, modular, reusable Python code across 6 files

---

## 🤝 Connect

If you found this useful or have suggestions, feel free to open an issue or connect with me on [LinkedIn](https://linkedin.com/in/yourprofile).

⭐ Star this repo if it helped you!
