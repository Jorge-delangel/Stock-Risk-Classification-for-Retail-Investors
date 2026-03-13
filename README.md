# 📈 Stock Risk Classification for Retail Investors

Machine learning project that classifies **short‑term stock risk (Low / Medium / High)** using historical market data and interpretable ML models, designed for a **retail investment platform targeting non‑expert users**.

---

## 🧠 Project Overview

Retail investors often struggle to assess short‑term market risk due to noisy and complex financial data.  
This project builds an **end‑to‑end machine learning pipeline** that predicts **next‑day stock risk levels** and explains *why* a stock is considered risky using **explainable AI (SHAP)**.

The goal is not only prediction, but **transparency and trust**, making the outputs suitable for real‑world fintech applications.

---

## 📊 Data

- **Source:** Yahoo Finance (via `yfinance` API)
- **Scope:**  
  - 10 years of daily data  
  - 6 large‑cap equities across multiple sectors:
    - Apple (AAPL)
    - Microsoft (MSFT)
    - Tesla (TSLA)
    - NVIDIA (NVDA)
    - Amazon (AMZN)
    - Alphabet (GOOGL)
- **Raw fields:** Open, High, Low, Volume, Date
- Data reshaped into a **panel format** with a `Ticker` column to enable cross‑asset feature engineering.

> Raw data is not committed to the repository. It can be reproduced by running the data collection scripts.

---

## 🧩 Feature Engineering

Features commonly used in **financial risk and momentum modeling**, including:

- **Returns**
  - Daily returns
  - Next‑day absolute return (used as risk proxy)
- **Volatility**
  - Rolling standard deviation (7, 14, 30 days)
  - Average True Range (ATR)
- **Momentum**
  - Moving averages (7, 15, 30 days)
  - RSI‑14 (Relative Strength Index)

### 🎯 Target Variable
- **Next‑day absolute return**, discretized into three classes using quantiles:
  - Low Risk (bottom 33%)
  - Medium Risk (middle 33%)
  - High Risk (top 33%)

---

## 🤖 Models

- **Baseline Model:** Logistic Regression  
  Chosen for simplicity and interpretability.

- **Primary Model:** XGBoost Classifier  
  Selected for strong performance on tabular data and compatibility with SHAP explainability.

### Training Strategy
- **80/20 time‑aware train/test split** to respect temporal ordering and avoid data leakage.

---

## 📈 Evaluation

Models were evaluated using **multi‑class classification metrics**:

- Accuracy  
- Macro Precision  
- Macro Recall  
- Macro F1 Score  

Since ROC curves are binary, a **One‑vs‑Rest (OvR)** strategy was used to evaluate class separability.

### Summary
- Both models achieved similar performance (accuracy ≈ 0.30–0.31)
- Results highlight the inherent difficulty of short‑term risk prediction in financial markets
- Performance is comparable to random guessing, emphasizing realistic limitations

---

## 🔍 Explainability (SHAP)

To ensure transparency and user trust:

- **Global explainability:**  
  SHAP bar plots identify the most influential features for **High Risk** predictions.
- **Local explainability:**  
  SHAP force plots explain individual predictions, showing how specific features push risk higher or lower.

**Key drivers of high risk predictions:**
- Trading volume  
- RSI‑14  
- ATR (30‑day)

---

## 🧪 Repository Structure

```text
stock-risk-classification/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ README.md
├─ notebooks/
│  ├─ 01_data_collection_yfinance.ipynb
│  ├─ 02_cleaning_and_panel_build.ipynb
│  ├─ 03_feature_engineering.ipynb
│  ├─ 04_modeling_baseline_logreg.ipynb
│  ├─ 05_modeling_xgboost.ipynb
│  ├─ 06_evaluation_roc_ovr.ipynb
│  └─ 07_explainability_shap.ipynb
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ explainability/
│  └─ utils/
├─ reports/
│  ├─ figures/
│  └─ final_report/
├─ scripts/
│  ├─ run_pipeline.py
│  └─ run_inference_demo.py
├─ requirements.txt
└─ README.md
