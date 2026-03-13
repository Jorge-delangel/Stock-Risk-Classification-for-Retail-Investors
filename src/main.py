"""
main.py

End-to-end pipeline for:
1) Downloading data
2) Feature engineering
3) Risk labeling
4) Train/test split
5) Training models (LogReg + XGBoost)
6) Evaluation + confusion matrices
7) SHAP explainability

Run:
    python main.py
"""

import os

import pandas as pd

from src.data_collection import get_dataset
from src.feature_engineering import build_feature_matrix
from src.risk_labeling import add_risk_labels
from src.train_models import (
    train_test_split_data,
    train_logistic_regression,
    train_xgboost_classifier,
    evaluate_model,
    save_models,
)
from src.evaluate_models import evaluate_and_plot
from src.explainability import compute_shap_values, plot_shap_summary


def ensure_directories() -> None:
    """Create standard project directories if they don't exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)


def run_pipeline() -> None:
    # -------------------------------------------------------------------------
    # 1. Data collection
    # -------------------------------------------------------------------------
    print("📥 Downloading raw price data...")
    df_raw = get_dataset(
        start="2018-01-01",
        end="2024-01-01",
        interval="1d",
    )
    df_raw.to_csv("data/raw/raw_prices.csv")
    print(f"Raw data shape: {df_raw.shape}")

    # -------------------------------------------------------------------------
    # 2. Feature engineering
    # -------------------------------------------------------------------------
    print("🧮 Building feature matrix...")
    df_features = build_feature_matrix(df_raw)
    df_features.to_csv("data/processed/features_no_labels.csv")
    print(f"Feature matrix shape (no labels): {df_features.shape}")

    # -------------------------------------------------------------------------
    # 3. Risk labeling
    # -------------------------------------------------------------------------
    print("🏷️ Adding risk labels...")
    df_labeled = add_risk_labels(
        df_features,
        price_col="Adj Close",
        horizon_days=5,       # adjust to match your notebook
        low_threshold=-0.05,  # adjust thresholds as needed
        high_threshold=0.05,
    )
    df_labeled.to_csv("data/processed/features_with_labels.csv")
    print(f"Labeled dataset shape: {df_labeled.shape}")
    print("Label distribution:")
    print(df_labeled["risk_label"].value_counts(normalize=True))

    # -------------------------------------------------------------------------
    # 4. Train/test split
    # -------------------------------------------------------------------------
    print("✂️ Splitting into train and test...")
    X_train, X_test, y_train, y_test = train_test_split_data(df_labeled)

    # -------------------------------------------------------------------------
    # 5. Train models
    # -------------------------------------------------------------------------
    print("🤖 Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    print("🌲 Training XGBoost...")
    xgb_model = train_xgboost_classifier(X_train, y_train)

    # -------------------------------------------------------------------------
    # 6. Evaluation + confusion matrices
    # -------------------------------------------------------------------------
    print("📊 Evaluating models...")
    evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")

    print("🧩 Generating confusion matrices...")
    evaluate_and_plot(
        lr_model,
        X_test,
        y_test,
        model_name="Logistic Regression",
        figures_dir="reports/figures",
    )
    evaluate_and_plot(
        xgb_model,
        X_test,
        y_test,
        model_name="XGBoost",
        figures_dir="reports/figures",
    )

    # -------------------------------------------------------------------------
    # 7. Save models
    # -------------------------------------------------------------------------
    print("💾 Saving models...")
    save_models(lr_model, xgb_model, output_dir="models")

    # -------------------------------------------------------------------------
    # 8. SHAP explainability (on a sample)
    # -------------------------------------------------------------------------
    print("🔍 Computing SHAP values for XGBoost...")
    # Use a manageable sample for SHAP
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = compute_shap_values(xgb_model, X_sample)

    print("📈 Plotting SHAP summary...")
    plot_shap_summary(
        shap_values,
        X_sample,
        save_path="reports/figures/shap_summary.png",
    )

    print("\n✅ Pipeline completed successfully.")
    print("Check:")
    print("  - data/raw/ and data/processed/ for datasets")
    print("  - models/ for saved models")
    print("  - reports/figures/ for confusion matrices and SHAP plots")


if __name__ == "__main__":
    ensure_directories()
    run_pipeline()
