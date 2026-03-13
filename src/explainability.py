import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import os


def compute_shap_values(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame
) -> shap.Explanation:
    """
    Compute SHAP values for a trained XGBoost model.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost model.
    X_sample : pd.DataFrame
        Sample of feature matrix.

    Returns
    -------
    shap.Explanation
        SHAP values object.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    save_path: str = "reports/figures/shap_summary.png"
) -> None:
    """
    Plot SHAP summary plot and save to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values.values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
