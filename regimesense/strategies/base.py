"""
base.py
-------
Abstract base class for all strategies.

Why an abstract class?
Every strategy must implement generate_signal().
If a new strategy forgets to implement it, Python raises an error immediately
rather than silently returning wrong results. This is defensive programming.

All strategies take the same input (OHLCV DataFrame) and return
the same output (a pd.Series of daily position sizes: +1=long, -1=short, 0=flat).
The allocator doesn't care HOW each strategy works — only what signal it outputs.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Base class. All strategies inherit from this."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Takes OHLCV DataFrame.
        Returns pd.Series of position signals, indexed by date.
          +1  = go long (buy)
           0  = flat (no position)
          -1  = go short (sell)

        Values between -1 and +1 are allowed for partial positions.
        """
        pass

    def daily_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Convenience: compute strategy daily returns from signal.
        Signal is generated on day t, position entered at close of day t,
        return realized on day t+1.

        This is the correct way to avoid lookahead bias:
        you can't trade on today's close using today's close price.
        """
        signal = self.generate_signal(df)
        price_return = df["Close"].pct_change()

        # shift(1): today's signal → tomorrow's return
        # This is the lookahead-bias-free implementation
        strategy_return = signal.shift(1) * price_return
        strategy_return.name = self.name
        return strategy_return