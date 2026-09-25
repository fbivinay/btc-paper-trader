# Progress Report
**AI-Powered Autonomous Bitcoin Trading System (Paper Trading)**

Status as of 25 September 2026 · 22 commits · 3,511 lines of ML code
Repository: https://github.com/fbivinay/btc-paper-trader
Live system: https://btc-paper-trader-fbivinays-projects.vercel.app

---

## 1. Dataset

### Primary: BTC/USDT 5-minute OHLCV

| | |
|---|---|
| Source | `data.binance.vision` monthly archives (official Binance dumps) |
| Rows | **525,848 candles** |
| Period | 2021-09-01 → 2026-08-31 (**5 years**) |
| Interval | 5 minutes |
| Missing | 40 candles (0.0076%), exchange maintenance windows |
| Storage | Parquet, 20 MB compressed |
| Script | `ml/fetch_history.py` (resumable, monthly cache) |

Columns retained: `open_time, open, high, low, close, volume, trades,
taker_buy_base, quote_volume`

`taker_buy_base` is the share of volume that was aggressive buying — order-flow
imbalance. `quote_volume ÷ trades` gives average trade size.

### Auxiliary datasets

| Dataset | Rows | Period | Source |
|---|---|---|---|
| Perpetual funding rate | 7,712 events | 2019-09 → 2026-09 | Binance futures API |
| Crypto Fear & Greed Index | 3,153 days | 2018-02 → 2026-09 | alternative.me (free) |
| Open interest, long/short ratio | recording live | 2026-09 → | Binance futures API |

**Data-availability finding.** Open interest and the long/short ratio are
retained by Binance for only ~30 days (~2 days at 5-minute granularity), so
neither can be back-filled or trained on. They are now recorded every candle into
a database table so a usable history accumulates. Funding rate and Fear & Greed
have full history and *are* used as model inputs.

### Labelled dataset

**523,007 rows × 54 features**, after discarding a 7-day feature warm-up and the
unresolvable tail.

---

## 2. Feature Engineering — 54 features

All features are **scale-free** (ratios, z-scores or bounded oscillators). A raw
price or raw moving average as an input means a model trained at \$40k BTC sees
meaningless values at \$120k. Verified by an automated test: multiplying all
prices by 10 must leave every feature unchanged.

| Group | n | Features |
|---|---|---|
| Trend | 3 | EMA20/EMA50 distance from price, EMA spread |
| Momentum | 3 | MACD, signal, histogram (all normalised by price) |
| Oscillators | 3 | RSI(14), Bollinger %B, Bollinger bandwidth |
| Volatility | 5 | ATR%, 1h/6h realised vol, vol ratio, 7-day vol percentile |
| Volume | 3 | volume z-score, trade-count z-score, VWAP distance |
| Trend strength | 2 | ADX(14), DI differential |
| Returns | 5 | 1, 3, 6, 12, 72-bar returns |
| Candle shape | 4 | high-low range, body, upper wick, lower wick |
| Seasonality | 2 | time-of-day (sin/cos) |
| **Funding / positioning** | **7** | rate, 1d & 7d means, 30d z-score, 30d percentile, 8h settlement phase (sin/cos) |
| **Sentiment** | **4** | F&G level, distance from neutral, 7d change, 30d z-score |
| **Order flow** | **8** | flow imbalance, 1h/6h/24h means, z-score, signed volume, price-flow divergence, trade-size z-score |
| **Path asymmetry** | **5** | up/down semi-variance ratio, return skew, kurtosis, distance to recent high/low in ATR units |

Indicators are implemented directly in pandas rather than via `pandas-ta`, which
is incompatible with NumPy 2.x.

### Lookahead protection

Every feature at candle *t* uses only data up to the close of *t*. Two merges
carry real lookahead risk and both are asserted against in automated tests:

- **Funding rate** settles at 00:00/08:00/16:00 UTC. `merge_asof(direction="backward")`
  guarantees a rate settled at 08:00 is invisible to candles before 08:00. A
  nearest-match merge would be an 8-hour leak.
- **Fear & Greed** publishes daily; same protection, a 1-day leak otherwise.

---

## 3. Labels

Two labelling schemes were implemented and compared.

### (a) Fixed-horizon labels

Predict the 4-hour forward return. Entry at the *next* candle's open (the
earliest realistic fill), exit at the close 48 bars later.

```
return >  threshold  → UP
return < -threshold  → DOWN
otherwise            → NEUTRAL
```

The threshold is **anchored to transaction cost**, not chosen arbitrarily.
Below the round-trip cost, a move labelled UP loses money after fees, so the
model would be trained to chase unprofitable trades.

### (b) Triple-barrier labels (López de Prado)

Label what a real trade would actually have done. For each candle, a
hypothetical long (+1.5% target / −0.5% stop) and short (−1.5% / +0.5%) are run
forward until a barrier is touched.

```
UP      the long reached its target before its stop
DOWN    the short reached its target before its stop
NEUTRAL neither did
```

Class balance: **24.5% DOWN / 50.7% NEUTRAL / 24.8% UP** — matching theory
exactly (at 3:1 reward-to-risk, a driftless walk hits its target 25% of the time).

Implemented as a vectorised sliding-window computation over 523,007 rows.

---

## 4. Model

### Primary: LSTM classifier (PyTorch)

```
Input   120 timesteps × 54 features        (10 hours of context)
          ↓
LSTM    2 layers, hidden 64, dropout 0.3
          ↓
Dense   64 → 32 → ReLU → 3
          ↓
Softmax UP / DOWN / NEUTRAL + confidence
```

- Loss: cross-entropy with inverse-frequency class weights
- Optimiser: AdamW, lr 1e-3, weight decay 1e-4, gradient clipping 1.0
- Epoch selection: **validation trading return**, not loss. Loss rewards being
  confidently NEUTRAL, which earns nothing.
- Normalisation fitted **per training window only**, never across the full file,
  to avoid leaking test-period statistics.

### Secondary: gradient-boosted meta-model (scikit-learn)

`HistGradientBoostingClassifier` used for meta-labelling — given a proposed
trade direction, predict whether *that trade* will reach its target before its
stop. Binary classification.

### Training infrastructure

Kaggle free GPU (Tesla P100). Code is shipped as a private dataset; the kernel
re-downloads and re-derives all data every run, so the weekly retrain and the
research sweep share one code path.

---

## 5. Walk-Forward Validation

Not a single train/test split. Three candidate training windows, four rolling
out-of-sample folds each — **12 independently trained models**.

```
Training window: 180 / 365 / 730 days
Test window:     90 days, strictly after training, never seen

fold 1   train 2023-09 → 2025-09   test 2025-09 → 2025-12
fold 2   train 2023-12 → 2025-12   test 2025-12 → 2026-03
fold 3   train 2024-03 → 2026-03   test 2026-03 → 2026-05
fold 4   train 2024-05 → 2026-05   test 2026-05 → 2026-08
```

Total out-of-sample evaluation: **12 months, 103,680 candles**.

---

## 6. Evaluation Metrics

Classification accuracy alone is **not** used for model selection, for a
documented reason: at the 30-minute horizon 68% of labels were NEUTRAL, so a
model that never trades scores 68% accuracy and earns nothing.

| Metric | Purpose |
|---|---|
| Accuracy, train and test | generalisation check |
| Precision per class (UP/DOWN) | a wrong trade costs the round trip; a missed one costs nothing |
| Win rate | fraction of closed trades in profit |
| **Gross edge per trade** | average PnL **before** fees — the decisive figure |
| Round-trip cost | fees + slippage, both legs |
| Sharpe ratio | annualised, from per-trade returns |
| Maximum drawdown | peak-to-trough, seeded at starting equity |
| Profit factor | gross wins ÷ gross losses |
| Trade count / exposure | feasibility and frequency constraints |
| Directional balance | detects a model that has only learned market drift |
| Folds profitable | consistency, not a single lucky window |

---

## 7. Results

### Generalisation (730-day window, 4h horizon)

| Split | n | Accuracy | Baseline | Directional |
|---|---|---|---|---|
| Train | 30,035 | 42.80% | 34.22% | 40.67% |
| **Test** | **25,920** | **44.27%** | **37.55%** | **40.21%** |

Train–test gap **−1.47%** — the model generalises; it is not memorising.
**Test accuracy exceeds the naive baseline by 6.7 points.**

### Training-window comparison

| Window | Test accuracy | Train–test gap | Verdict |
|---|---|---|---|
| 180 days | 40.75% | **+25.45%** | severely overfitted |
| 365 days | 41.95% | +6.23% | mild overfitting |
| **730 days** | **43.12%** | **+0.37%** | generalises cleanly |

**730 days is the correct training window.** This is exactly the question the
walk-forward experiment was designed to answer.

### Economic evaluation

The model has **measurable predictive skill that is too small to overcome
transaction costs**:

```
gross edge per trade (best confidence gate)   +0.043%
round-trip cost, Binance spot taker            0.250%
```

Same model, same 90-day out-of-sample window, cost assumption varied:

| Execution model | Round trip | Return | Sharpe |
|---|---|---|---|
| Spot taker (0.10%/leg) | 0.250% | −13.54% | −4.39 |
| Spot + BNB discount | 0.200% | −11.26% | −3.60 |
| Futures taker (0.05%/leg) | 0.110% | −7.00% | −2.16 |
| **Futures maker (0.02%/leg)** | **0.060%** | **+9.32%** | **+1.19** |

**The binding constraint is the cost structure, not the model.**

### Feature contribution

Adding funding-rate features raised best gross edge **+0.035% → +0.043%** (+23%)
and made the edge positive at a looser confidence gate, doubling tradeable
frequency. Permutation importance ranks `vwap_dist`, `funding_z`, `fng_z_30d` and
`funding_pctile` highest — the predictive signal is **slow positioning data**,
not fast microstructure.

### Central negative result

| Question the model was asked | Accuracy | Random baseline | Edge |
|---|---|---|---|
| Direction in 4 hours | **44.3%** | 37.6% | **+6.7 pts, real** |
| Which barrier is touched first | 25.8% | 24.8% | +1.0 pt, noise |

**The model predicts direction-at-horizon well and path-order not at all.** A
stop-loss strategy is decided entirely by path order — you never collect the
4-hour move, you collect whichever of your stop or target price reaches first.
This explains every negative economic result above, and it is only visible
because the trade was labelled, not just the price direction.

Meta-labelling recovered a small edge (28.4% win rate against a 27.5% break-even
at 1% risk per trade, +1.3 sd above 20 random control strategies, p ≈ 0.02), but
half of all configurations still lose money.

---

## 8. Deployed System

Beyond the model, the full platform specified is built and running.

| Component | Status |
|---|---|
| 5-minute live inference loop | running, backfills missed candles |
| Market regime classifier | 6 regimes from model-visible features |
| Jev strategy-selection layer | live, with deterministic fallback on failure |
| Deterministic risk engine | 5 modes, hard daily/exposure/leverage limits |
| Paper execution engine | pessimistic fills, gap handling, fees, slippage |
| PostgreSQL (Supabase) | 7 tables, row-level security verified 13/13 |
| Multi-user authentication | email + confirmation |
| Next.js dashboard | live candlestick chart, portfolio, trade history |
| Weekly retraining (Kaggle) | automated |
| Model promotion gate | automated, economic criteria |
| CI self-checks | 7 suites, run on every push |

### Model promotion gate

A model finishing training is not a reason to deploy it. Promotion requires
**all** of: gross edge beats round-trip cost, a majority of folds profitable, both
directions predicted, and beating the incumbent.

**No model has passed this gate.** The model currently in production was
force-promoted so the system can be demonstrated end to end; that override is
recorded in the database and displayed in the user interface.

---

## 9. Reproducibility and Correctness

Every module carries an assert-based self-check; all 7 suites pass. These are not
decoration — they caught:

- a label column leaking into the feature set (96.5% precision, Sharpe 58 — a
  model that was being handed the answer)
- a cost model charging slippage once per round trip instead of once per leg
- a maximum-drawdown calculation reporting zero when the first trade lost
- a sign error scoring every correct short as an equal-sized loss

Archived experiment records: `results/walkforward_h6_30min.json`,
`results/walkforward_h48_4h.json`, `results/meta_labelling.md`.

---

## 10. Remaining Work

1. Ensemble the meta-model across seeds to reduce configuration variance
2. Accumulate open-interest history (recording since 2026-09) and test whether it
   contributes as funding did
3. Model maker-order fill probability explicitly, since profitability depends on
   a ~79% fill rate
4. Final written report

---

## Summary

A complete, deployed, multi-user paper-trading platform with automated weekly
retraining and a validation gate, built entirely on free infrastructure.

The deep-learning model demonstrates **statistically real predictive skill**
(44.3% vs 37.6% baseline, out-of-sample, with healthy generalisation), and the
project's principal finding is that **this skill does not survive transaction
costs at a short horizon** — together with a precise account of why: direction
over a fixed horizon is partly predictable, while the order in which price
touches a stop or a target is not, and only the latter determines what a real
trade earns.
