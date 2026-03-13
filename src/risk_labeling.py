import pandas as pd
import numpy as np


def compute_forward_return(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    horizon_days: int = 5
) -> pd.Series:
    """
    Compute forward return over a given horizon.

    Returns
    -------
    pd.Series
        Forward return aligned with current date.
    """
    df = df.copy()
    fwd_price = df.groupby("Ticker")[price_col].shift(-horizon_days)
    fwd_ret = (fwd_price - df[price_col]) / df[price_col]
    return fwd_ret


def label_risk_bucket(
    forward_return: pd.Series,
    low_threshold: float = -0.05,
    high_threshold: float = 0.05
) -> pd.Series:
    """
    Map forward returns into discrete risk buckets:
    - Low   (0): drawdown smaller than low_threshold
    - Medium(1): between low_threshold and high_threshold
    - High  (2): above high_threshold

    You can adjust thresholds to match your notebook.
    """
    labels = pd.Series(index=forward_return.index, dtype="Int64")

    labels[forward_return <= low_threshold] = 2  # High risk (big negative move)
    labels[(forward_return > low_threshold) & (forward_return < high_threshold)] = 1
    labels[forward_return >= high_threshold] = 0  # Low risk (strong positive)

    return labels


def add_risk_labels(
    df_features: pd.DataFrame,
    price_col: str = "Adj Close",
    horizon_days: int = 5,
    low_threshold: float = -0.05,
    high_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Add forward return and risk bucket labels to the feature matrix.
    """
    df = df_features.copy()
    fwd_ret = compute_forward_return(df, price_col=price_col, horizon_days=horizon_days)
    df["forward_return"] = fwd_ret
    df["risk_label"] = label_risk_bucket(fwd_ret, low_threshold, high_threshold)
    df = df.dropna(subset=["risk_label"])
    return df
