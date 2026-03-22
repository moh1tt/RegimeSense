"""
paper_trader.py
---------------
Live paper trading loop using Alpaca (alpaca-py SDK).
Runs every Friday at 3:50 PM ET (10 mins before close).

What it does:
  1. Pulls last 300 days of SPY data from Alpaca
  2. Computes regime features
  3. Runs HMM to detect current regime
  4. Computes target strategy weights
  5. Maps weights to ETF positions (QQQ, SPY, BIL)
  6. Submits rebalance orders
  7. Logs regime + weights + P&L to CSV

Install: pip install alpaca-py python-dotenv schedule
"""

import os
import csv
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import numpy as np
import schedule
import time
from dotenv import load_dotenv

# ── alpaca-py imports (new SDK) ─────────────────────────────────────
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums    import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical  import StockHistoricalDataClient
from alpaca.data.requests    import StockBarsRequest
from alpaca.data.timeframe   import TimeFrame

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)

LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "live_log.csv"


# ── Client factory ──────────────────────────────────────────────────

def get_clients():
    """
    Returns two clients:
      trading_client → submit orders, read positions, read account
      data_client    → pull historical + live market data

    paper=True routes to the paper trading endpoint automatically.
    No need to hardcode the base_url.
    """
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY or ALPACA_SECRET_KEY not found in environment. "
            "Check your .env file is in the project root and load_dotenv() ran."
        )

    trading = TradingClient(
        api_key    = api_key,
        secret_key = secret_key,
        paper      = True,          # True = paper trading endpoint
    )
    data = StockHistoricalDataClient(
        api_key    = api_key,
        secret_key = secret_key,
    )
    return trading, data


# ── Data fetching ───────────────────────────────────────────────────

def fetch_live_data(data_client: StockHistoricalDataClient,
                    ticker: str = "SPY",
                    lookback_days: int = 300) -> pd.DataFrame:
    """
    Pull recent daily bars from Alpaca.
    We need 300 days because:
      - 200-day MA in trend_following strategy needs 200 days minimum
      - 252-day momentum signal needs ~1 year

    end = yesterday because today's bar isn't finalized until
    after market close — requesting today can return incomplete data.
    """
    end   = datetime.now() - timedelta(days=1)
    start = datetime.now() - timedelta(days=lookback_days + 10)
    # +10 buffer for weekends and market holidays

    request = StockBarsRequest(
        symbol_or_symbols = ticker,
        timeframe         = TimeFrame.Day,
        start             = start,
        end               = end,
        adjustment        = "all",   # split + dividend adjusted
    )

    raw = data_client.get_stock_bars(request).df

    # alpaca-py returns a MultiIndex: (symbol, timestamp)
    # Drop the symbol level — we only asked for one ticker
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    # Convert tz-aware UTC timestamps → tz-naive (rest of pipeline expects this)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    # alpaca-py uses lowercase column names — rename to match our pipeline
    raw = raw.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    })

    # Keep only OHLCV — alpaca-py also returns vwap, trade_count, etc.
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in raw.columns]
    df = raw[cols].copy()

    # Drop any rows with missing Close (shouldn't happen but defensive)
    df = df.dropna(subset=["Close"])

    log.info(f"Fetched {len(df)} live bars for {ticker} "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── ETF weight mapping ──────────────────────────────────────────────

def weights_to_etf_allocation(strategy_weights: dict) -> dict:
    """
    Map abstract strategy weights → concrete ETF allocations.

    QQQ = momentum + trend_following  (growth/momentum factor)
    SPY = mean_reversion              (broad market)
    BIL = defensive                   (3-month T-bills, ~cash)

    Normalizes so all weights sum exactly to 1.0.
    """
    growth    = (strategy_weights.get("momentum", 0)
               + strategy_weights.get("trend_following", 0))
    broad     = strategy_weights.get("mean_reversion", 0)
    defensive = strategy_weights.get("defensive", 0)

    total = growth + broad + defensive
    if total < 1e-6:
        # Fallback: equal weight if something is wrong
        log.warning("Strategy weights sum to zero — using equal ETF weight")
        return {"QQQ": 0.33, "SPY": 0.33, "BIL": 0.34}

    return {
        "QQQ": round(growth    / total, 4),
        "SPY": round(broad     / total, 4),
        "BIL": round(defensive / total, 4),
    }


# ── Order execution ─────────────────────────────────────────────────

def get_latest_price(data_client: StockHistoricalDataClient,
                     ticker: str) -> float:
    """
    Fetch the most recent closing price for a ticker.
    Used to convert dollar amount → share count for orders.
    """
    request = StockBarsRequest(
        symbol_or_symbols = ticker,
        timeframe         = TimeFrame.Day,
        start             = datetime.now() - timedelta(days=5),
        end               = datetime.now() - timedelta(days=1),
        limit             = 1,
    )
    bars = data_client.get_stock_bars(request).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index(level=0, drop=True)

    if bars.empty:
        raise ValueError(f"No price data returned for {ticker}")

    return float(bars["close"].iloc[-1])


def rebalance(trading_client: TradingClient,
              data_client:    StockHistoricalDataClient,
              target_weights: dict,
              portfolio_value: float) -> list:
    """
    Rebalance portfolio to target ETF weights.

    Logic:
      1. Read current positions from Alpaca
      2. For each ETF, compute (target value - current value)
      3. Skip if delta < 2% of portfolio (avoid trivial trades)
      4. Convert dollar delta → share count using latest price
      5. Submit market order

    Returns list of submitted order descriptions for logging.
    """
    # Current positions: {symbol: market_value}
    try:
        positions = {
            p.symbol: float(p.market_value)
            for p in trading_client.get_all_positions()
        }
    except Exception as e:
        log.error(f"Failed to fetch positions: {e}")
        positions = {}

    orders_submitted = []

    for etf, target_w in target_weights.items():
        target_value  = target_w * portfolio_value
        current_value = positions.get(etf, 0.0)
        delta_value   = target_value - current_value

        # Skip small rebalances — not worth the transaction cost
        if abs(delta_value) < 0.02 * portfolio_value:
            log.info(f"  {etf}: delta ${delta_value:+.0f} < 2% threshold, skipping")
            continue

        # Get price to compute share count
        try:
            price = get_latest_price(data_client, etf)
        except Exception as e:
            log.error(f"  Could not get price for {etf}: {e}")
            continue

        shares = int(abs(delta_value) / price)
        if shares == 0:
            log.info(f"  {etf}: 0 shares after rounding, skipping")
            continue

        side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL

        order_req = MarketOrderRequest(
            symbol        = etf,
            qty           = shares,
            side          = side,
            time_in_force = TimeInForce.DAY,
        )

        try:
            trading_client.submit_order(order_req)
            desc = f"{side.value} {shares} {etf} @ ~${price:.2f}"
            orders_submitted.append(desc)
            log.info(f"  Order submitted: {desc}")
        except Exception as e:
            log.error(f"  Order failed for {etf}: {e}")

    return orders_submitted


# ── CSV logging ─────────────────────────────────────────────────────

def log_to_csv(regime_label:     str,
               regime_probs:     dict,
               strategy_weights: dict,
               etf_weights:      dict,
               portfolio_value:  float):
    """
    Append one row to the live log CSV.
    After 8-12 weeks this log becomes your live validation dataset —
    you can compute a live IC from it.
    """
    LOG_PATH.parent.mkdir(exist_ok=True)
    write_header = not LOG_PATH.exists()

    row = {
        "date":            date.today().isoformat(),
        "regime_label":    regime_label,
        "prob_bull":       round(regime_probs.get("bull", 0),           4),
        "prob_choppy":     round(regime_probs.get("choppy", 0),         4),
        "prob_high_vol":   round(regime_probs.get("high_vol_trend", 0), 4),
        "prob_crisis":     round(regime_probs.get("crisis", 0),         4),
        "w_momentum":      round(strategy_weights.get("momentum", 0),        4),
        "w_mean_rev":      round(strategy_weights.get("mean_reversion", 0),  4),
        "w_trend":         round(strategy_weights.get("trend_following", 0), 4),
        "w_defensive":     round(strategy_weights.get("defensive", 0),       4),
        "etf_QQQ":         etf_weights.get("QQQ", 0),
        "etf_SPY":         etf_weights.get("SPY", 0),
        "etf_BIL":         etf_weights.get("BIL", 0),
        "portfolio_value": round(portfolio_value, 2),
    }

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    log.info(f"Logged to {LOG_PATH}")


# ── Main rebalance function ─────────────────────────────────────────
from regimesense.features.regime_features import (
    build_feature_matrix, normalize_features
)
from regimesense.regime.hmm_classifier  import RegimeClassifier
from regimesense.portfolio.allocator    import MetaAllocator
def run_weekly_rebalance():
    """
    Full pipeline — runs every Friday at 3:50 PM ET.

    fetch live data
      → compute regime features
        → HMM regime detection
          → strategy weight allocation
            → ETF order execution
              → CSV logging
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from regimesense.features.regime_features import (
        build_feature_matrix, normalize_features
    )
    from regimesense.regime.hmm_classifier import RegimeClassifier
    from regimesense.portfolio.allocator   import MetaAllocator

    log.info("=" * 52)
    log.info(f"RegimeSense weekly rebalance — {date.today()}")

    # ── Connect ─────────────────────────────────────────────────────
    trading_client, data_client = get_clients()

    account         = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    log.info(f"Portfolio value : ${portfolio_value:,.2f}")

    # ── Step 1: fetch live SPY data ─────────────────────────────────
    df = fetch_live_data(data_client, ticker="SPY", lookback_days=300)

    # ── Step 2: compute regime features ────────────────────────────
    features = build_feature_matrix(df)
    normed   = normalize_features(features)
    log.info(f"Feature matrix  : {len(normed)} rows × {normed.shape[1]} features")

    # ── Step 3: load trained HMM ────────────────────────────────────
    # We load, never retrain live — stability over adaptivity
    # clf = RegimeClassifier.load()
    clf = RegimeClassifier(n_states=4, n_iter=200, random_state=42)
    clf.fit(normed)

    # ── Step 4: detect today's regime ───────────────────────────────
    regimes      = clf.predict(normed)
    today        = regimes.iloc[-1]
    regime_label = today["regime_label"]
    regime_probs = {
        "bull":           float(today["prob_bull"]),
        "choppy":         float(today["prob_choppy"]),
        "high_vol_trend": float(today["prob_high_vol_trend"]),
        "crisis":         float(today["prob_crisis"]),
    }

    log.info(f"Today's regime  : {regime_label}")
    for k, v in regime_probs.items():
        log.info(f"  {k:<18}: {v:.3f}")

    # ── Step 5: compute strategy weights ────────────────────────────
    allocator        = MetaAllocator()
    weights_df       = allocator.compute_strategy_weights(regimes.tail(1))
    strategy_weights = weights_df.iloc[-1].to_dict()

    log.info("Strategy weights:")
    for k, v in strategy_weights.items():
        log.info(f"  {k:<20}: {v:.3f}")

    # ── Step 6: map to ETF allocations ─────────────────────────────
    etf_weights = weights_to_etf_allocation(strategy_weights)
    log.info(f"ETF targets     : {etf_weights}")

    # ── Step 7: submit orders ───────────────────────────────────────
    orders = rebalance(trading_client, data_client,
                       etf_weights, portfolio_value)
    if orders:
        log.info(f"Orders submitted: {len(orders)}")
    else:
        log.info("Orders submitted: none (all within 2% threshold)")

    # ── Step 8: log to CSV ──────────────────────────────────────────
    log_to_csv(regime_label, regime_probs, strategy_weights,
               etf_weights, portfolio_value)

    log.info("Rebalance complete.")
    log.info("=" * 52)


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("RegimeSense paper trader starting ...")
    log.info("Will run immediately, then every Friday at 15:50 ET")

    # Run once immediately on startup (for testing + first rebalance)
    run_weekly_rebalance()

    # Schedule weekly thereafter
    schedule.every().friday.at("15:50").do(run_weekly_rebalance)

    while True:
        schedule.run_pending()
        time.sleep(60)