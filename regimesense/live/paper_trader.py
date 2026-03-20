"""
paper_trader.py
---------------
Live paper trading loop using Alpaca.
Runs every Friday at 3:50 PM ET (10 mins before close).

What it does:
  1. Pulls last 300 days of SPY data from Alpaca
  2. Computes regime features
  3. Runs HMM to detect current regime
  4. Computes target strategy weights
  5. Maps weights to ETF positions (SPY, QQQ, BIL)
  6. Submits rebalance orders
  7. Logs regime + weights + P&L to CSV

Why ETFs instead of the raw strategies?
  In live trading we buy/sell actual securities.
  We map strategy weights → ETF proxies:
    momentum + trend  → QQQ (growth/momentum exposure)
    mean_reversion    → SPY (broad market, lower momentum)
    defensive         → BIL (T-bill ETF, essentially cash)
"""

import os
import csv
import logging
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
import schedule
import time
from dotenv import load_dotenv

load_dotenv()  # reads .env file for API keys

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "live_log.csv"

# ── ETF proxy map ────────────────────────────────────────────────────
# How we translate abstract strategy weights into real tradeable ETFs
ETF_MAP = {
    "growth":    "QQQ",   # momentum + trend_following signal
    "broad":     "SPY",   # mean_reversion signal
    "defensive": "BIL",   # defensive signal (T-bills, ~cash)
}


def get_alpaca_api():
    """Connect to Alpaca paper trading API."""
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        base_url="https://paper-api.alpaca.markets",
        api_version="v2",
    )


def fetch_live_data(api, ticker: str = "SPY",
                    lookback_days: int = 300) -> pd.DataFrame:
    """
    Pull recent daily bars from Alpaca.
    We need 300 days for the 200-day MA in the trend strategy
    and 252 days for the momentum signal.
    """
    bars = api.get_bars(
        ticker,
        "1Day",
        limit=lookback_days,
        adjustment="all",    # split + dividend adjusted
    ).df

    # Alpaca returns MultiIndex sometimes — flatten
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level=1)

    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars = bars.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",   "close": "Close",
        "volume": "Volume"
    })
    return bars[["Open", "High", "Low", "Close", "Volume"]]


def weights_to_etf_allocation(strategy_weights: dict) -> dict:
    """
    Convert strategy weights → ETF target allocations.

    Mapping logic:
      growth ETF (QQQ) weight  = momentum_w + trend_following_w
      broad ETF (SPY) weight   = mean_reversion_w
      defensive ETF (BIL) weight = defensive_w

    Normalize to sum to 1.0 (fully invested, no leverage).
    """
    growth    = (strategy_weights.get("momentum", 0)
               + strategy_weights.get("trend_following", 0))
    broad     = strategy_weights.get("mean_reversion", 0)
    defensive = strategy_weights.get("defensive", 0)

    total = growth + broad + defensive
    if total == 0:
        return {"QQQ": 0.33, "SPY": 0.33, "BIL": 0.34}

    return {
        "QQQ": round(growth    / total, 4),
        "SPY": round(broad     / total, 4),
        "BIL": round(defensive / total, 4),
    }


def rebalance(api, target_weights: dict, portfolio_value: float):
    """
    Submit market orders to reach target ETF weights.

    Steps:
      1. Get current positions
      2. Compute current weights
      3. Compute delta (target - current)
      4. Submit orders for significant changes only (>2% threshold)
         to avoid excessive trading on tiny weight drifts
    """
    # Current positions
    positions = {p.symbol: float(p.market_value)
                 for p in api.list_positions()}

    orders_submitted = []

    for etf, target_w in target_weights.items():
        target_value  = target_w * portfolio_value
        current_value = positions.get(etf, 0.0)
        delta_value   = target_value - current_value

        # Only rebalance if the difference exceeds 2% of portfolio
        if abs(delta_value) < 0.02 * portfolio_value:
            continue

        # Get current price to compute share count
        quote = api.get_latest_trade(etf)
        price = float(quote.price)
        shares = int(abs(delta_value) / price)

        if shares == 0:
            continue

        side = "buy" if delta_value > 0 else "sell"
        try:
            api.submit_order(
                symbol=etf,
                qty=shares,
                side=side,
                type="market",
                time_in_force="day",
            )
            orders_submitted.append(f"{side} {shares} {etf}")
            log.info(f"  Order: {side} {shares} {etf} @ ~${price:.2f}")
        except Exception as e:
            log.error(f"  Order failed for {etf}: {e}")

    return orders_submitted


def log_to_csv(regime_label: str,
               regime_probs: dict,
               strategy_weights: dict,
               etf_weights: dict,
               portfolio_value: float):
    """Append today's state to the live log CSV."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    write_header = not LOG_PATH.exists()

    row = {
        "date":             date.today().isoformat(),
        "regime_label":     regime_label,
        "prob_bull":        round(regime_probs.get("bull", 0), 4),
        "prob_choppy":      round(regime_probs.get("choppy", 0), 4),
        "prob_high_vol":    round(regime_probs.get("high_vol_trend", 0), 4),
        "prob_crisis":      round(regime_probs.get("crisis", 0), 4),
        "w_momentum":       round(strategy_weights.get("momentum", 0), 4),
        "w_mean_rev":       round(strategy_weights.get("mean_reversion", 0), 4),
        "w_trend":          round(strategy_weights.get("trend_following", 0), 4),
        "w_defensive":      round(strategy_weights.get("defensive", 0), 4),
        "etf_QQQ":          etf_weights.get("QQQ", 0),
        "etf_SPY":          etf_weights.get("SPY", 0),
        "etf_BIL":          etf_weights.get("BIL", 0),
        "portfolio_value":  round(portfolio_value, 2),
    }

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    log.info(f"Logged to {LOG_PATH}")


def run_weekly_rebalance():
    """
    Main function — runs every Friday at 3:50 PM ET.
    Full pipeline: fetch → features → regime → weights → orders → log.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from regimesense.features.regime_features import (
        build_feature_matrix, normalize_features)
    from regimesense.regime.hmm_classifier  import RegimeClassifier
    from regimesense.portfolio.allocator    import MetaAllocator

    log.info("=" * 50)
    log.info(f"RegimeSense weekly rebalance — {date.today()}")

    api = get_alpaca_api()

    # Portfolio value
    account        = api.get_account()
    portfolio_value = float(account.portfolio_value)
    log.info(f"Portfolio value: ${portfolio_value:,.2f}")

    # Step 1: fetch live data
    df = fetch_live_data(api, ticker="SPY", lookback_days=300)
    log.info(f"Fetched {len(df)} days of live SPY data")

    # Step 2: features
    features = build_feature_matrix(df)
    normed   = normalize_features(features)

    # Step 3: load trained HMM (trained in backtest, not retrained live)
    clf = RegimeClassifier.load()   # loads from logs/hmm_model.pkl

    # Step 4: predict today's regime (use only last row)
    regimes        = clf.predict(normed)
    today_regime   = regimes.iloc[-1]
    regime_label   = today_regime["regime_label"]
    regime_probs   = {
        "bull":           today_regime["prob_bull"],
        "choppy":         today_regime["prob_choppy"],
        "high_vol_trend": today_regime["prob_high_vol_trend"],
        "crisis":         today_regime["prob_crisis"],
    }

    log.info(f"Today's regime : {regime_label}")
    log.info(f"Probabilities  : {regime_probs}")

    # Step 5: strategy weights from today's regime
    allocator = MetaAllocator()
    weights_df = allocator.compute_strategy_weights(regimes.tail(1))
    strategy_weights = weights_df.iloc[-1].to_dict()
    log.info(f"Strategy weights: {strategy_weights}")

    # Step 6: map to ETF allocations
    etf_weights = weights_to_etf_allocation(strategy_weights)
    log.info(f"ETF targets    : {etf_weights}")

    # Step 7: submit rebalance orders
    orders = rebalance(api, etf_weights, portfolio_value)
    log.info(f"Orders submitted: {orders if orders else 'none (within threshold)'}")

    # Step 8: log everything
    log_to_csv(regime_label, regime_probs, strategy_weights,
               etf_weights, portfolio_value)

    log.info("Rebalance complete.")
    log.info("=" * 50)


# ── Scheduler ─────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("RegimeSense live paper trader starting ...")
    log.info("Scheduled: every Friday at 15:50 ET")

    # Run immediately on startup for testing
    run_weekly_rebalance()

    # Then schedule weekly
    schedule.every().friday.at("15:50").do(run_weekly_rebalance)

    while True:
        schedule.run_pending()
        time.sleep(60)