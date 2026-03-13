import pandas as pd
from typing import Dict

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import os


def load_models(models_dir: str = "models") -> Dict[str, object]:
    """
    Load trained models from disk.
    """
    lr_path = os.path.join(models_dir, "logistic_regression.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.json")

    lr_model = joblib.load(lr_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    return {"logistic_regression": lr_model, "xgboost": xgb_model}


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title: str,
    save_path: str | None = None
) -> None:
    """
    Plot and optionally save a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_and_plot(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    figures_dir: str = "reports/figures"
) -> None:
    """
    Generate classification report and confusion matrix plot.
    """
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, digits=3))

    labels = sorted(y_test.unique())
    save_path = f"{figures_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        title=f"{model_name} - Confusion Matrix",
        save_path=save_path,
    )
