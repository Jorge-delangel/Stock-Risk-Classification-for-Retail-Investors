import pandas as pd
import numpy as np


def add_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """
    Add daily returns and rolling returns.

    Assumes df is indexed by (Date, Ticker).
    """
    df = df.copy()
    df["return_1d"] = df.groupby("Ticker")[price_col].pct_change()
    df["return_5d"] = df.groupby("Ticker")[price_col].pct_change(5)
    df["return_21d"] = df.groupby("Ticker")[price_col].pct_change(21)
    return df


def add_volatility_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """
    Add rolling volatility features.
    """
    df = df.copy()
    group = df.groupby("Ticker")[price_col]

    df["vol_5d"] = group.pct_change().rolling(5).std()
    df["vol_21d"] = group.pct_change().rolling(21).std()
    return df


def add_technical_indicators(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """
    Add simple technical indicators (moving averages, RSI-like proxy, etc.).
    """
    df = df.copy()
    group = df.groupby("Ticker")[price_col]

    df["ma_10"] = group.transform(lambda x: x.rolling(10).mean())
    df["ma_50"] = group.transform(lambda x: x.rolling(50).mean())
    df["ma_ratio_10_50"] = df["ma_10"] / df["ma_50"]

    # Simple momentum proxy: price / 21-day rolling mean
    df["price_over_ma_21"] = group.transform(lambda x: x / x.rolling(21).mean())
    return df


def build_feature_matrix(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw OHLCV data indexed by (Date, Ticker).

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """
    df = df_raw.copy()
    df = add_returns(df)
    df = add_volatility_features(df)
    df = add_technical_indicators(df)

    # Drop rows with NaNs from rolling windows
    df = df.dropna()
    return df
