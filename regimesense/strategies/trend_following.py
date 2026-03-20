"""
trend_following.py
------------------
Dual moving average crossover (50-day vs 200-day).

Logic: when the fast MA (50d) crosses ABOVE the slow MA (200d),
the trend is turning up — go long. When it crosses below — go flat.

This is the "Golden Cross / Death Cross" — one of the oldest and
most robust technical signals. It's slower than momentum but generates
fewer false signals because the 200-day MA is very hard to fake.

Key difference from momentum:
  Momentum: looks at raw return over 12 months
  Trend following: looks at smoothed price level crossover

Works best in: sustained trends (bull, high-vol trend).
Fails in:      choppy market — the MAs cross back and forth constantly
               (whipsawing you into lots of small losses).
"""

import pandas as pd
from regimesense.strategies.base import Strategy


class TrendFollowingStrategy(Strategy):

    def __init__(self, fast_window: int = 50,
                       slow_window: int = 200):
        super().__init__(name="trend_following")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()

        # +1 when fast MA is above slow MA (uptrend), 0 otherwise
        signal = (fast_ma > slow_ma).astype(float)
        signal.name = self.name
        return signal