> Portfolio project by Jorge Luis Del Angel Maldonado – MSc Data Analytics

<p align="center">
  <img src="https://github.com/Jorge-delangel/Stock-Risk-Classification-for-Retail-Investors/blob/main/assets/Copilot_20260313_173601.png" alt="Stock Risk Classification Cover" width="100%">
</p>

# 📈 Stock Risk Classification for Retail Investors  
**End‑to‑End Machine Learning Pipeline for Short‑Term Risk Prediction**

This project implements a complete, reproducible machine‑learning workflow that predicts **short‑term stock risk levels** (Low / Medium / High) using historical market data, engineered financial features, and interpretable models. It is designed as a professional portfolio project demonstrating data engineering, modeling, evaluation, and explainability.

---

## 🧠 Project Summary  
Retail investors often face uncertainty when evaluating short‑term market risk. This project provides a transparent ML pipeline that:

- Downloads historical stock data  
- Engineers quantitative financial features  
- Labels future risk levels  
- Trains baseline and advanced ML models  
- Evaluates performance with multi‑class metrics  
- Generates explainability visuals (SHAP)  
- Saves models and outputs in a clean, production‑style structure  

The entire workflow is automated through `main.py`.

---

## 🗂️ Repository Structure  
```text
Stock-Risk-Classification-for-Retail-Investors/
│
├── main.py
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── risk_labeling.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── explainability.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── reports/
└── figures/
```
Each component has a clear responsibility, making the project easy to navigate and extend.

---

## 📥 Data Collection  
Data is sourced from **Yahoo Finance** using the `yfinance` API.

- Universe: A small set of large‑cap equities  
- Frequency: Daily  
- Fields: OHLCV + Adjusted Close  
- Storage:
  - `data/raw/` — raw downloaded data  
  - `data/processed/` — feature matrices and labeled datasets  

The `data/` folder is intentionally empty in the repository and populated automatically when running the pipeline.

---

## 🧮 Feature Engineering  
Implemented in `src/feature_engineering.py`, including:

- Daily and multi‑day returns  
- Rolling volatility  
- Moving averages (10, 50, 21 days)  
- Momentum ratios  
- Price‑over‑moving‑average indicators  

These features capture short‑term behavior relevant to risk.

---

## 🏷️ Risk Labeling  
Implemented in `src/risk_labeling.py`.

The target variable is **forward return over 5 days**, bucketed into:

- **High Risk** — large negative forward return  
- **Medium Risk** — moderate movement  
- **Low Risk** — strong positive forward return  

Thresholds are configurable.

---

## 🤖 Models  
Two models are trained:

### **1. Logistic Regression (Baseline)**
- Simple, interpretable  
- Establishes a benchmark  

### **2. XGBoost Classifier (Primary Model)**
- Handles non‑linear relationships  
- Strong performance on tabular data  
- Compatible with SHAP explainability  

Models are saved to: `models/logistic_regression.pkl` and
`models/xgboost_model.json`.

---

## 📊 Evaluation  
Evaluation includes:

- Classification report  
- Confusion matrices  
- Class distribution  
- SHAP summary plot  

All visuals are saved to: `reports/figures/`.

These files are generated automatically after running the pipeline.

---

## 🔍 Explainability  
Using SHAP:

- **Global importance** — which features drive risk predictions  
- **Local explanations** — why a specific prediction was made  

This ensures transparency and interpretability.

---

## ▶️ Running the Pipeline  

### **1. Create the environment**
```bash
conda env create -f environment.yml
conda activate stock-risk-classification
```
### **2. Run the full workflow**
```bash
python main.py
```
### **3. View outputs**
- Processed datasets → `data/processed/`
- Saved models → `models/`
- Visuals → `reports/figures/`

---

## 🧭 Next Steps (Optional Enhancements)

- Add more tickers or sectors
- Incorporate macroeconomic indicators
- Build a Streamlit dashboard
- Add backtesting for strategy evaluation
- Deploy as an API or web app

---

## 👤 Author
**Jorge Luis Del Angel Maldonado**  
MSc Science in Data Analytics – Machine Learning  
Analytics & ML Portfolio Project

---

## 📄 Disclaimer
This project is a portfolio and academic demonstration.
It is not intended as financial advice or a production‑grade trading system.
All results are for educational purposes only.

