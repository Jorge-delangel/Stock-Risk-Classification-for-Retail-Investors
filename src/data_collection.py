import pandas as pd
import yfinance as yf
from typing import List, Tuple


def download_price_data(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download OHLCV data for a list of tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        Stock tickers to download.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    interval : str
        Data frequency, e.g. '1d', '1h'.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame with (Date, Ticker) index.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    # Normalize to long format: index = Date, columns = [Ticker, Feature]
    frames = []
    for ticker in tickers:
        df_t = data[ticker].copy()
        df_t["Ticker"] = ticker
        df_t.index.name = "Date"
        frames.append(df_t.reset_index())

    full = pd.concat(frames, axis=0, ignore_index=True)
    full.set_index(["Date", "Ticker"], inplace=True)
    return full


def load_sample_universe() -> List[str]:
    """
    Return a small sample universe of tickers.
    Replace with your own universe if needed.
    """
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def get_dataset(
    start: str = "2018-01-01",
    end: str = "2024-01-01",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    High-level function to get the full raw dataset.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV data indexed by (Date, Ticker).
    """
    tickers = load_sample_universe()
    df = download_price_data(tickers, start=start, end=end, interval=interval)
    return df
