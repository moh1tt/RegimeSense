"""
mean_reversion.py
-----------------
RSI-based mean reversion strategy.

Logic: when price is "oversold" (RSI < 35), it tends to bounce back up.
When "overbought" (RSI > 65), it tends to pull back.

RSI (Relative Strength Index) measures the ratio of recent up-moves
to down-moves over a rolling window. It's bounded 0-100.
  RSI < 30 = extremely oversold (strong buy signal)
  RSI > 70 = extremely overbought (sell signal)

We use 35/65 (less extreme) to get more frequent signals.

Works best in: choppy regime where prices oscillate around a mean.
Fails in:      trending regime (RSI stays oversold during a crash,
               and you keep buying a falling knife).
"""

import pandas as pd
import numpy as np
from regimesense.strategies.base import Strategy


class MeanReversionStrategy(Strategy):

    def __init__(self, rsi_window: int = 14,
                       oversold: float = 35,
                       overbought: float = 65):
        super().__init__(name="mean_reversion")
        self.rsi_window  = rsi_window
        self.oversold    = oversold
        self.overbought  = overbought

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """
        RSI formula:
          delta   = daily price change
          gain    = average of up-days over window
          loss    = average of down-days over window
          RS      = gain / loss
          RSI     = 100 - (100 / (1 + RS))
        """
        delta = close.diff()

        gain = delta.clip(lower=0)   # keep only positive changes
        loss = -delta.clip(upper=0)  # keep only negative changes (flip sign)

        # Exponential weighted average (EWM) — standard RSI uses this
        avg_gain = gain.ewm(span=self.rsi_window, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_window, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._compute_rsi(df["Close"])

        # +1 when oversold (buy the dip), -1 when overbought (sell the peak)
        signal = pd.Series(0.0, index=df.index, name=self.name)
        signal[rsi < self.oversold]  =  1.0
        signal[rsi > self.overbought] = -1.0
        return signal