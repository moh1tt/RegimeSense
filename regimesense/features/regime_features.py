"""
regime_features.py
------------------
Takes raw OHLCV data → returns a clean DataFrame of 5 regime features.

Why 5 features?
  Each one captures a DIFFERENT dimension of market behaviour.
  Together they give the HMM enough signal to distinguish 4 regime types.
  More is not always better — correlated features confuse the HMM.
"""

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Daily log returns: log(P_t / P_{t-1})

    Why log returns instead of simple (P_t - P_{t-1}) / P_{t-1}?
    Log returns are additive across time (weekly = sum of daily log returns).
    Simple returns are not. For rolling statistics, log returns behave better.
    """
    return np.log(prices / prices.shift(1))


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling standard deviation of returns, annualized.

    window=20 means: use the past 20 trading days (~1 month).
    Multiply by sqrt(252) to convert daily vol to annual vol.
    252 = number of trading days in a year.

    Result: a number like 0.15 means 15% annualized volatility.
    S&P 500 typical range: 0.08 (very calm) to 0.80 (crisis).
    """
    return returns.rolling(window).std() * np.sqrt(252)


def return_autocorrelation(returns: pd.Series,
                           window: int = 40,
                           lag: int = 5) -> pd.Series:
    """
    Rolling autocorrelation at lag=5 days.

    Autocorrelation asks: "if the market was up 5 days ago, does that
    predict what it does today?"

    Positive (>0.1) → trending: momentum is persisting
    Near zero       → random walk: no predictable pattern
    Negative (<-0.1)→ mean-reverting: market tends to reverse

    We use lag=5 (one week) because daily noise is too short
    and monthly is too slow to be actionable.
    """
    return returns.rolling(window).apply(
        lambda x: x.autocorr(lag=lag),
        raw=False  # raw=False means pass a pd.Series (needed for .autocorr())
    )


def rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling Sharpe ratio over 60 trading days (~3 months).

    Sharpe = mean(returns) / std(returns) * sqrt(252)

    This measures: "how well has the market been rewarding risk-taking?"
    High (>1.0)  → smooth uptrend, strategies should be aggressive
    Near zero    → flat/noisy market, reduce exposure
    Negative     → market is falling, go defensive

    60 days is long enough to smooth out noise but short enough
    to respond to changing conditions within a quarter.
    """
    mean = returns.rolling(window).mean()
    std  = returns.rolling(window).std()
    # Avoid division by zero on rare flat periods
    sharpe = mean / std.replace(0, np.nan) * np.sqrt(252)
    return sharpe


def return_skewness(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling skewness of returns over 60 days.

    Skewness measures the SHAPE of the return distribution:
      Symmetric (near 0)  → normal market
      Negative skew (<-1) → fat left tail = crash risk elevated
                            "most days are fine but the bad days are VERY bad"
      Positive skew (>1)  → lottery-ticket market (rare big gains)

    This is your CRISIS EARLY WARNING signal.
    In late 2007, skewness turned sharply negative MONTHS before
    the market peaked. Same in Feb 2020.
    """
    return returns.rolling(window).skew()


def volume_momentum(volume: pd.Series,
                    fast: int = 5,
                    slow: int = 20) -> pd.Series:
    """
    Relative volume: recent volume vs longer-term average.

    vol_momentum = 5-day avg volume / 20-day avg volume

    > 1.0 → volume is ABOVE normal (active market, institutional participation)
    < 1.0 → volume is BELOW normal (quiet market, low conviction)

    Why this matters:
    A price move on HIGH volume = real conviction (trending regime)
    A price move on LOW volume = weak hands, likely to reverse (mean-reverting)
    """
    fast_avg = volume.rolling(fast).mean()
    slow_avg = volume.rolling(slow).mean()
    return fast_avg / slow_avg.replace(0, np.nan)


def build_feature_matrix(df: pd.DataFrame,
                         config: dict = None) -> pd.DataFrame:
    """
    MASTER FUNCTION: takes raw OHLCV DataFrame → returns feature matrix.

    This is the only function other modules need to call.
    Everything else above is a helper.

    Parameters
    ----------
    df     : pd.DataFrame  — OHLCV data from fetcher.py
    config : dict          — optional, uses defaults if None

    Returns
    -------
    pd.DataFrame with 5 columns:
        realized_vol, autocorr_5d, rolling_sharpe, skewness, volume_momentum
    Index = same DatetimeIndex as input, NaN rows at start dropped.
    """
    w = (config or {}).get("features_window", 20)

    log.info("Computing regime features ...")

    close  = df["Close"]
    volume = df["Volume"]
    rets   = compute_returns(close)

    features = pd.DataFrame({
        "realized_vol":    realized_volatility(rets, window=w),
        "autocorr_5d":     return_autocorrelation(rets, window=w*2, lag=5),
        "rolling_sharpe":  rolling_sharpe(rets, window=w*3),
        "skewness":        return_skewness(rets, window=m*3 if (m:=w) else 60),
        "volume_momentum": volume_momentum(volume, fast=5, slow=w),
    }, index=df.index)

    # Drop rows where ANY feature is NaN
    # (this happens at the start because rolling windows need warm-up data)
    n_before = len(features)
    features = features.dropna()
    n_dropped = n_before - len(features)
    log.info(f"Feature matrix: {len(features)} rows "
             f"({n_dropped} dropped for rolling warm-up).")

    return features


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each feature to mean=0, std=1.

    WHY THIS IS CRITICAL:
    The HMM uses a Gaussian distribution to model each feature.
    If realized_vol is in range [0.08, 0.80] and rolling_sharpe is
    in range [-3.0, 4.0], the HMM will be dominated by whichever
    feature has the largest raw scale.

    After normalization, every feature contributes equally.
    This is the same reason you standardize features before running PCA or k-means.
    """
    return (features - features.mean()) / features.std()


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from regimesense.data.fetcher import load_data_from_config

    df       = load_data_from_config()
    features = build_feature_matrix(df)
    normed   = normalize_features(features)

    print("\nRaw features — first 3 rows:")
    print(features.head(3).round(4))

    print("\nNormalized features — first 3 rows:")
    print(normed.head(3).round(4))

    print("\nFeature stats (should all be ~mean=0, std=1 after normalization):")
    print(normed.describe().round(3))