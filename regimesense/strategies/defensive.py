"""
defensive.py
------------
Crisis protection strategy — moves to cash when crisis signals fire.

Logic: monitors two danger signals:
  1. Realized vol spike: if current vol is 1.5x its 60-day average,
     something bad is happening — reduce exposure
  2. Price drawdown: if price is more than 8% below its recent 60-day high,
     we're in a downtrend — go flat

When EITHER signal fires → 0 (flat/cash).
When BOTH are calm → 1 (fully invested, earn the equity risk premium).

This strategy never goes short — it just moves to cash.
Why? Because shorting requires borrowing and timing — too many ways to lose.
Being in cash during a crisis is the win. You don't need to profit from it.

Works best in: crisis regime (avoids the drawdowns that kill other strategies).
Fails in:      bull market (you'll be in cash during some of the best days).
"""

import pandas as pd
import numpy as np
from regimesense.strategies.base import Strategy


class DefensiveStrategy(Strategy):

    def __init__(self, vol_window: int = 20,
                       vol_threshold: float = 1.5,
                       drawdown_window: int = 60,
                       drawdown_threshold: float = 0.08):
        """
        vol_threshold      : go flat if vol > vol_threshold × avg vol
        drawdown_threshold : go flat if price > X% below recent high
        """
        super().__init__(name="defensive")
        self.vol_window         = vol_window
        self.vol_threshold      = vol_threshold
        self.drawdown_window    = drawdown_window
        self.drawdown_threshold = drawdown_threshold

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close   = df["Close"]
        returns = close.pct_change()

        # Signal 1: vol spike detector
        current_vol = returns.rolling(self.vol_window).std()
        avg_vol     = current_vol.rolling(60).mean()
        vol_spike   = current_vol > (self.vol_threshold * avg_vol)

        # Signal 2: drawdown detector
        rolling_high = close.rolling(self.drawdown_window).max()
        drawdown     = (close - rolling_high) / rolling_high   # negative values
        in_drawdown  = drawdown < -self.drawdown_threshold

        # Go flat (0) if EITHER danger signal fires, invested (1) otherwise
        danger = vol_spike | in_drawdown
        signal = (~danger).astype(float)
        signal.name = self.name
        return signal