"""
allocator.py
------------
Meta-allocator: combines regime posteriors + strategy signals
→ final daily portfolio return.

The key insight: we don't hard-switch strategies.
We BLEND them using regime probabilities as weights.

Example on a typical bull day:
  prob_bull=0.75, prob_choppy=0.20, prob_crisis=0.05

  momentum weight    = 0.75×0.8 + 0.20×0.1 + 0.05×0.1 = 0.625
  mean_rev weight    = 0.75×0.3 + 0.20×0.9 + 0.05×0.1 = 0.410
  trend_follow weight= 0.75×0.6 + 0.20×0.1 + 0.05×0.2 = 0.480
  defensive weight   = 0.75×0.1 + 0.20×0.2 + 0.05×0.9 = 0.155

  Normalize so weights sum to 1.0 → final allocation vector.
"""

import numpy as np
import pandas as pd
import logging
from regimesense.strategies.momentum       import MomentumStrategy
from regimesense.strategies.mean_reversion import MeanReversionStrategy
from regimesense.strategies.trend_following import TrendFollowingStrategy
from regimesense.strategies.defensive      import DefensiveStrategy

log = logging.getLogger(__name__)


# ── Affinity matrix ─────────────────────────────────────────────────
# How well does each strategy perform in each regime?
# Rows = strategies, Columns = [bull, choppy, high_vol_trend, crisis]
# Values: 0 (terrible) to 1 (great).
# These are informed starting points — calibrate from your backtest data.

AFFINITY = {
    #                       bull  choppy  hi_vol  crisis
    "momentum":       np.array([0.8,   0.1,    0.6,    0.1]),
    "mean_reversion": np.array([0.3,   0.9,    0.1,    0.1]),
    "trend_following":np.array([0.6,   0.1,    0.8,    0.2]),
    "defensive":      np.array([0.1,   0.2,    0.3,    0.9]),
}

REGIME_ORDER = ["bull", "choppy", "high_vol_trend", "crisis"]


class MetaAllocator:
    """
    Combines 4 strategies using regime posteriors as a soft weighting mechanism.
    """

    def __init__(self):
        # Instantiate all 4 strategies
        self.strategies = {
            "momentum":        MomentumStrategy(),
            "mean_reversion":  MeanReversionStrategy(),
            "trend_following": TrendFollowingStrategy(),
            "defensive":       DefensiveStrategy(),
        }
        self.affinity = AFFINITY

    def compute_strategy_weights(self,
                                  regime_probs: pd.DataFrame) -> pd.DataFrame:
        """
        For each day, compute how much weight to give each strategy.

        Parameters
        ----------
        regime_probs : pd.DataFrame with columns
                       [prob_bull, prob_choppy, prob_high_vol_trend, prob_crisis]

        Returns
        -------
        pd.DataFrame with columns [momentum, mean_reversion,
                                    trend_following, defensive]
        Each row sums to 1.0.
        """
        # Extract regime probability matrix: shape (n_days, 4)
        prob_cols = [f"prob_{r}" for r in REGIME_ORDER]
        P = regime_probs[prob_cols].values   # (n_days, 4)

        weights = {}
        for strat_name, affinity_vec in self.affinity.items():
            # Weight for this strategy on each day:
            # dot product of [prob_bull, prob_choppy, ...] with affinity vector
            # = how much this strategy is "called for" given today's regime mix
            w = P @ affinity_vec   # shape: (n_days,)
            weights[strat_name] = w

        df_weights = pd.DataFrame(weights, index=regime_probs.index)

        # Normalize each row to sum to 1
        # (so total portfolio exposure = 100%, not more)
        row_sums = df_weights.sum(axis=1).replace(0, 1)
        df_weights = df_weights.div(row_sums, axis=0)

        return df_weights

    def compute_portfolio_returns(self,
                                   df: pd.DataFrame,
                                   regime_probs: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: data + regime probs → daily portfolio returns.

        Parameters
        ----------
        df           : OHLCV DataFrame (from fetcher)
        regime_probs : regime posterior DataFrame (from classifier)

        Returns
        -------
        pd.DataFrame with columns:
          - momentum, mean_reversion, trend_following, defensive : individual returns
          - portfolio_return  : weighted combination
          - weight_momentum, weight_mean_reversion, ...          : daily weights
        """
        log.info("Computing strategy signals ...")

        # Step 1: generate signal for every strategy
        strat_returns = {}
        for name, strat in self.strategies.items():
            strat_returns[name] = strat.daily_returns(df)

        ret_df = pd.DataFrame(strat_returns)

        # Step 2: compute strategy weights from regime probs
        weights = self.compute_strategy_weights(regime_probs)

        # Align indices — regime_probs starts later due to HMM warm-up
        common_idx = ret_df.index.intersection(weights.index)
        ret_df  = ret_df.loc[common_idx]
        weights = weights.loc[common_idx]

        # Step 3: portfolio return = weighted sum of strategy returns
        # Multiply element-wise and sum across strategies
        portfolio_ret = (ret_df * weights).sum(axis=1)

        # Step 4: pack everything into one clean output DataFrame
        result = pd.DataFrame(index=common_idx)
        for name in self.strategies:
            result[f"return_{name}"]   = ret_df[name]
            result[f"weight_{name}"]   = weights[name]
        result["portfolio_return"] = portfolio_ret

        return result


# ── Performance metrics helper ───────────────────────────────────────

def performance_metrics(returns: pd.Series,
                         freq: int = 252) -> dict:
    """
    Compute standard quant performance metrics from a daily return series.
    These are the numbers you'll quote in interviews and your README.
    """
    r = returns.dropna()
    ann_ret  = r.mean() * freq
    ann_vol  = r.std()  * np.sqrt(freq)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum      = (1 + r).cumprod()
    peak     = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd   = drawdown.min()

    # Calmar ratio = ann_return / abs(max_drawdown)
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    hit_rate = (r > 0).mean()

    return {
        "ann_return":  round(ann_ret,  4),
        "ann_vol":     round(ann_vol,  4),
        "sharpe":      round(sharpe,   3),
        "max_drawdown":round(max_dd,   4),
        "calmar":      round(calmar,   3),
        "hit_rate":    round(hit_rate, 3),
        "n_days":      len(r),
    }


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from regimesense.data.fetcher import load_data_from_config
    from regimesense.features.regime_features import (
        build_feature_matrix, normalize_features)
    from regimesense.regime.hmm_classifier import RegimeClassifier

    # Full pipeline
    df       = load_data_from_config()
    features = build_feature_matrix(df)
    normed   = normalize_features(features)

    clf      = RegimeClassifier(n_states=4, n_iter=200, random_state=42)
    clf.fit(normed)
    regimes  = clf.predict(normed)

    allocator = MetaAllocator()
    results   = allocator.compute_portfolio_returns(df, regimes)

    # Print metrics for each strategy AND the portfolio
    print("\n" + "="*55)
    print(f"{'Strategy':<20} {'Sharpe':>7} {'Ann Ret':>8} {'Max DD':>8}")
    print("="*55)

    for col in ["return_momentum", "return_mean_reversion",
                "return_trend_following", "return_defensive",
                "portfolio_return"]:
        m    = performance_metrics(results[col])
        name = col.replace("return_","").replace("_"," ")
        print(f"{name:<20} {m['sharpe']:>7.3f} "
              f"{m['ann_return']:>7.1%} {m['max_drawdown']:>8.1%}")
    print("="*55)