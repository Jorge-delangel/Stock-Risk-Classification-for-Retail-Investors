import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import xgboost as xgb


FEATURE_COLS_DEFAULT = [
    "return_1d",
    "return_5d",
    "return_21d",
    "vol_5d",
    "vol_21d",
    "ma_10",
    "ma_50",
    "ma_ratio_10_50",
    "price_over_ma_21",
]


def train_test_split_data(
    df: pd.DataFrame,
    feature_cols=None,
    label_col: str = "risk_label",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix into train and test sets.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_DEFAULT

    X = df[feature_cols]
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build a simple baseline pipeline with scaling + logistic regression.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial")),
        ]
    )
    return pipe


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train logistic regression model.
    """
    pipe = build_logistic_regression_pipeline()
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] | None = None
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier for multi-class risk prediction.
    """
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
        }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> str:
    """
    Print and return a classification report.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print(f"\n=== {model_name} ===")
    print(report)
    return report


def save_models(
    lr_model: Pipeline,
    xgb_model: xgb.XGBClassifier,
    output_dir: str = "models"
) -> None:
    """
    Save trained models to disk.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr_model, f"{output_dir}/logistic_regression.pkl")
    xgb_model.save_model(f"{output_dir}/xgboost_model.json")
