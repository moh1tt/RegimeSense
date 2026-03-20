"""
momentum.py
-----------
12-1 month cross-sectional momentum (simplified to time-series here).

Logic: if SPY's return over the past 12 months (excluding last month)
is positive, go long. If negative, go flat/short.

Why exclude the last month (1-month reversal)?
Short-term return tends to REVERSE (liquidity effects, market maker
inventory). The 12-1 formulation skips that noise and captures
the genuine medium-term trend signal.

Works best in: bull regime, high-vol trending regime.
Fails in:      choppy regime (frequent reversals whipsaw you).
"""

import pandas as pd
from regimesense.strategies.base import Strategy


class MomentumStrategy(Strategy):

    def __init__(self, long_window: int = 252,
                       short_window: int = 21):
        """
        long_window  : lookback for the trend (252 = 12 months)
        short_window : exclusion window (21 = 1 month reversal skip)
        """
        super().__init__(name="momentum")
        self.long_window  = long_window
        self.short_window = short_window

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        # 12-month return (t-252 to t)
        ret_long  = close.pct_change(self.long_window)

        # 1-month return (t-21 to t) — this is the reversal noise to skip
        ret_short = close.pct_change(self.short_window)

        # 12-1 momentum: long-term trend minus short-term reversal noise
        momentum = ret_long - ret_short

        # Signal: +1 if positive momentum, 0 if negative
        # We go flat (not short) because momentum shorting is expensive
        signal = (momentum > 0).astype(float)
        signal.name = self.name
        return signal