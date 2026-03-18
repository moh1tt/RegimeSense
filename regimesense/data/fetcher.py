"""
fetcher.py
----------
One job: download OHLCV data from Yahoo Finance and return it clean.
No transformations, no feature engineering — just raw data in, clean DataFrame out.
"""

import yfinance as yf
import pandas as pd
import yaml
import logging
from pathlib import Path

# Set up a logger so we can see what's happening when we run it
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    """
    Reads config/config.yaml and returns it as a Python dictionary.
    We use a config file so we never hardcode dates or tickers in the code.
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_ohlcv(ticker: str,
                start: str,
                end: str) -> pd.DataFrame:
    """
    Download daily OHLCV (Open/High/Low/Close/Volume) from Yahoo Finance.

    Parameters
    ----------
    ticker : str   e.g. "SPY"
    start  : str   e.g. "2010-01-01"
    end    : str   e.g. "2024-12-31"

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    Index is DatetimeIndex (trading days only).
    """
    log.info(f"Fetching {ticker} from {start} to {end} ...")

    raw = yf.download(ticker,
                      start=start,
                      end=end,
                      auto_adjust=True,   # adjusts for splits + dividends
                      progress=False)

    # yfinance sometimes returns multi-level columns — flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    # Keep only the columns we care about
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Drop any rows where Close is NaN (market holidays sometimes slip in)
    before = len(df)
    df = df.dropna(subset=["Close"])
    dropped = before - len(df)
    if dropped > 0:
        log.warning(f"Dropped {dropped} rows with missing Close price.")

    log.info(f"Fetched {len(df)} trading days for {ticker}.")
    return df


def load_data_from_config() -> pd.DataFrame:
    """
    Convenience function: reads ticker/dates from config and calls fetch_ohlcv.
    This is what every other module in the project will call.

    Usage:
        from regimesense.data.fetcher import load_data_from_config
        df = load_data_from_config()
    """
    cfg = load_config()
    return fetch_ohlcv(
        ticker=cfg["data"]["ticker"],
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
    )


# ── Quick self-test: run this file directly to make sure it works ──
if __name__ == "__main__":
    df = load_data_from_config()
    print("\nShape:", df.shape)
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nAny nulls?", df.isnull().sum().sum())