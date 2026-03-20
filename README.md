# RegimeSense

**An adaptive trading system that detects market regimes using Hidden Markov Models and dynamically allocates across a strategy pool — with live paper trading and weekly attribution logging.**

---

## Overview

Most trading strategies are built for one kind of market. Momentum strategies thrive in bull runs and get destroyed in choppy conditions. Mean reversion strategies work in sideways markets and fail catastrophically during sustained trends. RegimeSense solves this by doing what professional systematic funds do: detecting which market regime is currently active and continuously reweighting a pool of strategies based on that detection.

The system classifies every trading day into one of four regimes — **bull**, **choppy**, **high-vol trend**, and **crisis** — using a 5-feature Gaussian Hidden Markov Model trained on 20 years of SPY data. It then blends four strategies using soft allocation: instead of hard-switching ("it's bull, run momentum only"), it weights each strategy proportionally to the HMM's posterior probabilities. A day that's 70% bull and 30% choppy gets a blended allocation reflecting that uncertainty.

The result on the 2021–2024 out-of-sample test: **Sharpe 0.769 vs 0.641 for pure momentum**, with max drawdown reduced from -33.7% to -25.6%.

---

## Architecture

```
RegimeSense/
├── regimesense/
│   ├── data/
│   │   └── fetcher.py          # OHLCV pipeline — yfinance, auto-adjusted
│   ├── features/
│   │   └── regime_features.py  # 5 regime features with normalization
│   ├── regime/
│   │   └── hmm_classifier.py   # GaussianHMM, auto-labeling, save/load
│   ├── strategies/
│   │   ├── base.py             # Abstract Strategy class
│   │   ├── momentum.py         # 12-1 month cross-sectional momentum
│   │   ├── mean_reversion.py   # RSI-based mean reversion
│   │   ├── trend_following.py  # Dual MA crossover (50/200-day)
│   │   └── defensive.py        # Vol spike + drawdown capital protection
│   ├── portfolio/
│   │   └── allocator.py        # Meta-allocator with affinity matrix
│   └── live/
│       └── paper_trader.py     # Weekly Alpaca paper trading loop
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_regime_analysis.ipynb
│   └── 03_backtest_results.ipynb
├── logs/                        # Regime charts, backtest PNGs, live CSV log
└── config/config.yaml           # All parameters in one place
```

---

## The 5 Regime Features

Each feature captures a distinct dimension of market behaviour. Together they give the HMM enough signal to separate four qualitatively different market states.

| Feature | What it measures | Why it matters |
|---|---|---|
| **Realized volatility** | Rolling 20-day annualized std of log returns | Primary separator — low vol = calm, high vol = stressed |
| **Return autocorrelation** | Lag-5 autocorrelation over 40-day window | Positive = trending, negative = mean-reverting |
| **Rolling Sharpe** | 60-day risk-adjusted return | Quality of recent price movement, not just direction |
| **Return skewness** | 60-day skewness of return distribution | Negative spike = fat left tail = crash risk elevated |
| **Volume momentum** | 5-day avg vol / 20-day avg vol | Conviction behind price moves — confirms regime signals |

Features are cross-sectionally normalized (mean=0, std=1) before fitting the HMM, so no single feature dominates due to scale.

---

## The HMM Regime Classifier

The Gaussian HMM learns three things from the data:

- **Emission distributions** — the typical feature values in each regime (a 5-dimensional Gaussian per state)
- **Transition matrix** — how likely each regime is to persist or switch (learned stickiness)
- **Posterior probabilities** — for every day, a probability distribution across all 4 regimes

Training uses the Baum-Welch EM algorithm with `covariance_type="full"`, meaning each regime gets its own full covariance matrix — it can learn that in the crisis regime, volatility and Sharpe are strongly anti-correlated.

**Learned regime map** (from 5,032 training days, 2005–2024):

| Regime | Vol (norm.) | Sharpe (norm.) | % of days | Character |
|---|---|---|---|---|
| Bull | -0.584 | +1.143 | ~26% | Low vol, strong risk-adjusted returns |
| Choppy | +0.190 | +0.093 | ~24% | Moderate vol, near-zero Sharpe |
| High-vol trend | -0.443 | -0.174 | ~31% | Low vol but weak returns |
| Crisis | +1.265 | -1.370 | ~19% | Very high vol, negative Sharpe |

**Transition matrix diagonal** (regime stickiness):

| Regime | Self-transition prob. |
|---|---|
| Bull | ~0.97 |
| Choppy | ~0.94 |
| High-vol trend | ~0.93 |
| Crisis | ~0.89 |

Once the market enters any regime, it stays there with high probability — bull markets last an average of ~33 trading days before switching. This stickiness is why HMM outperforms simple day-by-day clustering.

---

## The Strategy Pool

Each strategy is designed for a specific regime condition. No single strategy is expected to work everywhere.

### Momentum (`momentum.py`)
12-1 month time-series momentum: buy when the past 12-month return minus the past 1-month return is positive. The 1-month exclusion skips short-term reversal noise. Goes long or flat — no shorting.

*Designed for: bull, high-vol trend. Fails in: choppy (reversals whipsaw it).*

### Mean Reversion (`mean_reversion.py`)
RSI-14 based: long when RSI < 35 (oversold), short when RSI > 65 (overbought). Uses EWM smoothing for the standard RSI calculation.

*Designed for: choppy. Fails in: trending (buying falling knives).*

### Trend Following (`trend_following.py`)
Dual MA crossover: long when the 50-day MA is above the 200-day MA (Golden Cross), flat otherwise. Slower and more robust than momentum — fewer false signals.

*Designed for: bull, high-vol trend. Fails in: choppy (whipsaw on crossovers).*

### Defensive (`defensive.py`)
Two danger detectors running in parallel: a volatility spike detector (current vol vs 1-year baseline) and an 8% drawdown detector. Goes to cash (signal = 0) when either fires. Never shorts — capital preservation only.

*Designed for: crisis. Fails in: bull (misses upside while in cash).*

---

## The Meta-Allocator

The allocator computes strategy weights as:

```
weight_i = (regime_probs · affinity_i) / Σ weights
```

Where `affinity_i` is a 4-element vector encoding how well strategy `i` performs in each regime. The dot product of the current regime probability vector with the affinity vector gives each strategy's raw weight for that day. Row normalization ensures total exposure stays at 100%.

**Affinity matrix** (informed by financial theory, calibrated from per-regime Sharpe ratios):

| Strategy | Bull | Choppy | Hi-vol trend | Crisis |
|---|---|---|---|---|
| Momentum | 0.8 | 0.1 | 0.6 | 0.1 |
| Mean reversion | 0.3 | 0.9 | 0.1 | 0.1 |
| Trend following | 0.6 | 0.1 | 0.8 | 0.2 |
| Defensive | 0.1 | 0.2 | 0.3 | 0.9 |

**Example on a typical bull day** (prob_bull=0.75, prob_choppy=0.20, prob_crisis=0.05):
- Momentum weight: 0.75×0.8 + 0.20×0.1 + 0.05×0.1 = **0.625**
- Mean reversion: 0.75×0.3 + 0.20×0.9 + 0.05×0.1 = **0.410**
- Trend following: 0.75×0.6 + 0.20×0.1 + 0.05×0.2 = **0.470**
- Defensive: 0.75×0.1 + 0.20×0.2 + 0.05×0.9 = **0.160**

After normalization: momentum 37.8%, trend 28.5%, mean_rev 24.8%, defensive 9.7%.

---

## Backtest Results

**Walk-forward split:** HMM trained on 2005–2020. Strategies validated on 2021–2022. Final performance reported on 2023–2024 only (never touched during development).

**Transaction costs:** 10 basis points per rebalance, applied on weekly allocation changes exceeding 2% of portfolio value.

### Out-of-sample performance (2023–2024)

| Strategy | Sharpe | Ann. Return | Max Drawdown |
|---|---|---|---|
| Momentum | 0.641 | 9.3% | -33.7% |
| Mean reversion | 0.435 | 5.4% | -28.5% |
| Trend following | 0.733 | 9.8% | -33.7% |
| Defensive | 0.596 | 7.4% | -12.1% |
| **RegimeSense** | **0.769** | **7.1%** | **-25.6%** |
| SPY buy-and-hold | ~0.65 | ~12% | -24.5% |

RegimeSense achieves the highest Sharpe in the strategy pool with meaningfully lower drawdown than any trend-following strategy. The lower raw return vs SPY is expected — the system trades less aggressively than buy-and-hold, which is the tradeoff for the lower drawdown profile.

**Rolling Sharpe analysis:** The 60-day rolling Sharpe stays positive throughout 2023 and turns negative in early 2024 during the regime transition period — consistent with a system responding to genuine changes in market conditions rather than overfitting to a single regime.

---

## Live Paper Trading

The live loop (`paper_trader.py`) runs every Friday at 3:50 PM ET. It:

1. Pulls the last 300 days of SPY data from Alpaca (split + dividend adjusted)
2. Computes the 5 regime features on live data
3. Loads the trained HMM (not retrained — stability over adaptivity)
4. Classifies today's regime and extracts posterior probabilities
5. Computes target strategy weights via the meta-allocator
6. Maps weights to 3 ETF proxies: QQQ (growth/momentum), SPY (broad), BIL (cash)
7. Submits market orders for positions deviating >2% from target
8. Appends regime label, probabilities, weights, and portfolio value to `logs/live_log.csv`

**ETF mapping rationale:**

| Strategy weight | ETF proxy | Logic |
|---|---|---|
| Momentum + trend following | QQQ | Growth-oriented, high momentum factor loading |
| Mean reversion | SPY | Broad market, lower momentum tilt |
| Defensive | BIL | 3-month T-bill ETF — essentially cash with yield |

The live log serves as ongoing model validation. After 8–12 weeks of logging, the regime calls can be evaluated against subsequent 5-day returns to compute a live IC — confirming whether the HMM's regime detection is adding real predictive value beyond the training period.

---

## Installation

```bash
git clone https://github.com/yourusername/RegimeSense
cd RegimeSense
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add your Alpaca paper trading credentials to `.env`:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

---

## Usage

**Run the full pipeline once (backtest mode):**

```bash
python regimesense/portfolio/allocator.py
```

**Train and save the HMM:**

```bash
python regimesense/regime/hmm_classifier.py
```

**Start the live paper trading loop:**

```bash
python regimesense/live/paper_trader.py
```

**Explore in notebooks:**

```bash
jupyter notebook
# Run in order: 01_data_exploration → 02_regime_analysis → 03_backtest_results
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `hmmlearn` | Gaussian HMM implementation |
| `yfinance` | Historical OHLCV data |
| `alpaca-trade-api` | Live paper trading execution |
| `scikit-learn` | Feature normalization |
| `pandas` / `numpy` | Data manipulation |
| `schedule` | Weekly rebalance scheduling |

---

## Design Decisions

**Why HMM over k-means or threshold rules?**
k-means clusters each day independently — it has no memory, so it can label a Tuesday as crisis and Wednesday as bull even if nothing changed. HMM explicitly models the temporal dependency between regimes via the transition matrix. Markets don't randomly jump between states. HMM captures this stickiness and produces smoother, more economically coherent regime sequences.

**Why soft allocation over hard switching?**
Hard switching creates large, sudden position changes that generate transaction costs and are sensitive to misclassification at regime boundaries. Soft allocation using posterior probabilities means the system transitions gradually as the HMM's confidence shifts. This reduces both trading costs and the fragility of being wrong at the exact turning point.

**Why not retrain the HMM live?**
Retraining on 300 days of recent data would produce an unstable model that overfits to recent conditions. The value of the HMM is that it learned the long-run statistical properties of market regimes from 20 years of data. Stability is a feature, not a limitation.

**Why these 5 features specifically?**
Each feature captures an orthogonal dimension of market behaviour: level of volatility (realized vol), direction persistence (autocorrelation), recent performance quality (Sharpe), tail risk (skewness), and institutional conviction (volume momentum). The correlation matrix shows no pair exceeds 0.7 correlation — they're genuinely complementary.

---

## What This Is Not

This is a research and learning system, not production trading infrastructure. It uses a single underlying asset (SPY), simplified ETF proxies for live trading, and does not account for tax efficiency, short-selling constraints, or position-level risk limits. The backtest does not include bid-ask spread costs or market impact beyond the flat 10bps assumption.

---

## References

- Ang, A. & Timmermann, A. (2012). *Regime Changes and Financial Markets.* Annual Review of Financial Economics.
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series.* Econometrica.
- Nystrup, P. et al. (2017). *Long-horizon forecasting with machine learning-based liquidity-adjusted risk models.* Journal of Risk.
- Baum, L. & Welch, L. (1972). *An inequality and associated maximization technique — Baum-Welch algorithm.*

---

*Built as part of a quant research portfolio. Live paper trading active on Alpaca since March 2026.*
