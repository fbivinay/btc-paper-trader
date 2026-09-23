# BTC Paper Trader

Autonomous paper trading for BTC/USDT. A deep-learning model predicts short-term
direction, a regime classifier describes conditions, [Jev](https://www.jevai.org)
selects a strategy, a deterministic risk engine approves or refuses it, and a
simulated execution engine books the trade. Virtual capital only — no real
orders, no real funds.

**Live:** https://btc-paper-trader-fbivinays-projects.vercel.app
**Code:** https://github.com/fbivinay/btc-paper-trader

---

## The headline result

**The model has real predictive skill and the strategy is still not reliably
profitable.** Both halves of that sentence are load-bearing, and the gap between
them is what this project is actually about.

| Question the model was asked | Accuracy | Random baseline | Verdict |
|---|---|---|---|
| Direction in 4 hours | **44.3%** | 37.6% | real edge, +6.7 points |
| Which barrier is touched first | 25.8% | 24.8% | **no edge** |

A stop-loss strategy lives entirely in the second row. You never collect the
4-hour move — you collect whichever of your stop or target price reaches first.
The model predicts the first question well and the second one not at all.

Most projects in this space never notice, because they never label the trade
they actually take.

---

## What was tested, in order

Each step moved a number for a reason that can be explained. That sequence is
the contribution, more than any single model.

### 1. A 30-minute horizon is structurally impossible

```
horizon   median |move|   vs 0.25% round-trip cost
 30 min      0.152%            0.61x     <- the median move is smaller than the fee
  1 h        0.212%            0.85x
  4 h        0.424%            1.70x
 24 h        1.257%            5.03x
```

At 30 minutes, **a perfect direction oracle still loses money on more than half
its trades.** The walk-forward confirmed it: 0/12 folds profitable, average PnL
per trade −0.2550% against a 0.2500% cost. Gross edge was ≈ 0 and the loss was
exactly the fee.

Worth stating plainly: the model was *measurably better than chance* here
(24% precision against a 16% base rate) and still guaranteed to lose.

→ `results/walkforward_h6_30min.json`

### 2. At 4 hours there is a real edge, and it is too small

Sharpe improved −47 → −11, precision 24% → 38%, trades per fold 1700 → 400.

```
      gate       mean gross edge per trade
 all trades         -0.027%
    top 25%         +0.021%
     top 9%         +0.035%      <- best
   cost to beat:     0.250%
```

Gross edge rises monotonically with model confidence — the ranking is genuinely
informative. It is just ~7× too small to pay retail spot fees.

### 3. The binding constraint was the cost structure, not the model

Same model, same 90 days, only the fee assumption changed:

```
   spot taker 0.10% + slippage    0.250% round trip   -13.54%   sharpe -4.39
               spot + BNB 0.075%  0.200%              -11.26%   sharpe -3.60
             futures taker 0.05%  0.110%               -7.00%   sharpe -2.16
             futures maker 0.02%  0.060%               +9.32%   sharpe +1.19
```

**The same predictions lose 13% on spot and make 9% on futures maker pricing.**
That is the single most important number in the project.

With a conservative fill assumption (60% maker / 40% taker, plus 1bp per leg of
adverse selection) break-even sits at roughly a **79% maker fill rate** — and
even when profitable, max drawdown is about **30%**.

### 4. Funding rate helps; open interest and news cannot be used

| Source | History available | Usable for training |
|---|---|---|
| Funding rate | 2019 → now, 7,712 events | **yes** |
| Fear & Greed index | 2018 → now, 3,153 days | **yes** |
| Open interest | ~30 days (2 days at 5m) | no — recorded going forward |
| Long/short ratio | ~30 days | no — recorded going forward |
| Marketaux news | none on free tier | no |

Adding funding features raised best gross edge **+0.035% → +0.043%**, and turned
the edge positive at a looser gate, meaning it could trade twice as often. A
real contribution from one feature family.

Open interest is captured every candle into `market_context` so that a usable
history exists in a few months. Every day not recorded is lost permanently.

### 5. Win rate is gameable and means nothing on its own

Random entries, **no model at all**:

```
 take profit  stop loss   win rate   total return
        0.3%       3.0%      69.0%       -96.0%
```

Using the real model, an 80% win rate is reachable on request:

```
 take profit  stop loss   win rate   12mo return
       0.20%      5.00%      80.1%       -17.6%
```

Eight trades in ten close green and the account still loses, because each winner
takes 0.20% and each loser gives back 5.00%. This is shipped as the
**Max Win Rate** mode in the app, with those exact numbers displayed whenever it
is selected.

### 6. Triple-barrier labels and meta-labelling

Relabelling so that training matches execution — *did +1.5% arrive before
−0.5%* — produced **no edge at all** (25.8% against a 24.8% random baseline).
Twelve different deterministic entry rules all landed between 24.6% and 25.2%.

A gradient-boosted meta-model found a marginal long-only signal (29.52% vs
25.95% base), but it did not survive honest accounting: 24.37% once trades cannot
overlap, and the best tradeable configuration is 44 trades at 31.82%, which is
z = 1.04, p ≈ 0.15. Not distinguishable from luck.

→ `results/meta_labelling.md`

**A note on the arithmetic of 3:1.** At a 3:1 reward-to-risk ratio a coin flip
wins **25%** of the time, not 50%. Break-even is 25%, or 29.2% after fees.
Measured on five years of data, random entry at R=0.5% with a 72h limit wins
24.8% — matching theory to two decimal places. A target of "60% win rate at 3:1"
is not an ambitious goal; it is a goal that compounds $10,000 into $166bn in two
years, which is about 5% of the entire crypto market.

---

## Architecture

```
                      GitHub Actions (*/5)  +  Supabase pg_cron
                                      |
                       ┌──────────────┴──────────────┐
                       |     ml/live_loop.py         |
                       |                             |
   Binance ──candles──▶│  features (41)              │
   Binance ──funding──▶│    ↓                        │
   alt.me  ──F&G ─────▶│  LSTM  →  UP/DOWN/NEUTRAL   │
                       │    ↓                        │
                       │  regime classifier          │
                       │    ↓                        │
                       │  Jev  ──(429/timeout)──▶ fallback table
                       │    ↓                        │
                       │  risk engine  (per user)    │
                       │    ↓                        │
                       │  paper engine               │
                       └──────────────┬──────────────┘
                                      ▼
                            Supabase (Postgres + RLS)
                                      ▼
                            Next.js on Vercel
```

No always-on server. No Docker. No Redis. Everything runs on free tiers.

### Why each piece is where it is

**Market-wide vs per-user split.** `decisions` and `model_versions` hold one row
per candle, shared by everyone. `profiles`, `positions`, `risk_events` and
`equity_snapshots` are per-user and fenced by row-level security. The loop runs
inference **once** and fans out per user through the risk engine. Without that
split, N users would mean N model inferences and N Jev calls.

**Everything keys off `candle_time`, never wall clock.** A run that fires 20
minutes late processes the candles it missed. Open positions are advanced
through *every* missed candle before any new entry, so a stop that would have
been hit during an outage is honoured at its own price — verified in production
when a position stopped out at 13:30 during a 4.5-hour gap.

**Cash and equity are derived from the position ledger on every read**, never
stored as a running balance. A balance column drifts the moment a write is
retried.

**Jev is called at most once per candle**, and only when some active user's
confidence gate could pass the signal — roughly 1–2 calls/day instead of 288.
The free tier rate-limits hard, and a strategy label cannot change a decision
that is already HOLD.

---

## Repository layout

```
ml/
  config.py          horizon, costs, barrier and label settings (env-overridable)
  fetch_history.py   5y of 5m candles from data.binance.vision
  fetch_funding.py   funding-rate history back to 2019
  fetch_sentiment.py Fear & Greed index back to 2018
  features.py        41 features + horizon and triple-barrier labels
  train.py           LSTM + walk-forward across candidate training windows
  metrics.py         non-overlapping trade simulation and trading metrics
  regime.py          market regime from the same features the model sees
  jev_client.py      Jev decision layer + deterministic fallback
  risk_engine.py     position sizing and hard limits. AI proposes, this approves
  paper_engine.py    simulated execution: fills, stops, targets, fees
  live_loop.py       one pass of the whole pipeline
  promote_model.py   the validation gate that decides what goes to production
  db.py              dependency-free PostgREST client
  test_rls.py        proves row-level security actually isolates users
  market_context.py  records open interest and long/short for future training

supabase/schema.sql  6 tables, RLS policies, signup trigger
kaggle/              GPU kernel that rebuilds data and runs the sweep
web/                 Next.js dashboard
.github/workflows/   5-minute loop, weekly retrain, self-checks on every push
results/             archived walk-forward runs and experiment write-ups
```

Every module ships an assert-based self-check. Run them all:

```bash
cd ml
python metrics.py && python features.py --check && python risk_engine.py \
  && python paper_engine.py && python regime.py && python jev_client.py && python db.py
```

These are not decoration. They caught a label column leaking into the feature
set (96.5% precision, Sharpe 58 — a model that knew the answer), a cost model
that charged slippage once instead of twice, and a drawdown calculation that
reported zero when the first trade lost.

---

## Running it

### Local

```bash
pip install -r requirements.txt

python ml/fetch_history.py        # ~160MB, resumable
python ml/fetch_funding.py
python ml/fetch_sentiment.py
python ml/features.py
python ml/train.py --smoke        # pipeline check, ~1 min on CPU
```

Create `.env` in the project root (gitignored):

```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...       # bypasses RLS — never expose to a browser
JEV_API_KEY=jev_...
```

### Training on Kaggle

The heavy walk-forward runs on Kaggle's free GPU. Code ships as a small private
dataset; the kernel re-downloads and re-derives everything so the weekly retrain
and the sweep share one code path.

```bash
python ml/push_kaggle.py --run
python ml/pull_kaggle.py
python ml/promote_model.py --register artifacts/artifacts/walkforward_*.json
```

### The promotion gate

A model finishing training is not a reason to deploy it. `promote_model.py`
promotes only if **all** of these hold:

- gross edge per trade beats the round-trip cost
- a majority of walk-forward folds were profitable
- the model predicts both directions (it has not just learned the drift)
- it beats the incumbent

**No model has ever passed this gate.** The one running in production was forced
through with `--force-promote` so the system can be demonstrated end to end, and
that override is recorded permanently in its database row and shown in the UI.

---

## Trading modes

| Mode | User sets | System controls |
|---|---|---|
| 🛡️ Conservative | capital, allocation | strategy and trades within low-risk limits |
| ⚖️ Balanced | capital, allocation | strategy and trades within medium-risk limits |
| 🚀 Aggressive | capital, allocation | strategy and trades within high-risk limits |
| 🤖 AI Autonomous | capital, max daily loss | allocation, strategy, entry, exit, size |
| ⚠️ Max Win Rate | capital, allocation | demonstration only — 80% win rate, −17.6% return |

Every mode passes through the same risk engine. `ai_autonomous` has the widest
mandate, not an unlimited one: the daily loss limit, the exposure ceiling and the
no-leverage rule apply to all of them.

Confidence thresholds are **percentiles, not absolute values**, resolved against
each promoted model's own confidence distribution. Across the walk-forward the
top-9% cutoff ranged 0.43 to 0.95 by fold — a hardcoded 0.5 would have traded a
third of all candles in one quarter and none the next.

---

## Things that only appear in production

Collected because each cost real time to diagnose.

| Symptom | Cause |
|---|---|
| `HTTP 451` on every CI run, works locally | Binance geo-blocks US IPs; GitHub runners are in the US. Fixed with `data-api.binance.vision`. |
| Scheduled workflow never fires | GitHub's `*/5` cron is best-effort and was ignored for 4.5 hours. Replaced with Supabase `pg_cron` calling the workflow_dispatch API. |
| `403` from Python, `429` from curl, same request | jevai.org rejects the default `python-urllib` User-Agent. |
| `400 Bad Request` from Jev | Model id must be `typesafe-ai/jev`; the community site's example (`jev-1`) is wrong. |
| Model file missing in CI | An absolute Windows path was stored in the database. Paths are now relative to the repo root. |
| Supabase advisor warning | PostgREST exposes every public function as an RPC endpoint, so a `SECURITY DEFINER` signup trigger was callable by anonymous users. |
| Auth redirects to localhost | Supabase Site URL default, plus a missing PKCE code exchange at `/auth/callback`. |

---

## Limitations

- **Paper trading only.** No real orders are placed and no real funds are held.
- **The production model did not pass validation.** It is expected to decline
  most signals and to lose slowly.
- **Profitability depends on execution quality.** Break-even needs roughly a 79%
  maker fill rate; below that the strategy loses.
- **Max drawdown is ~30%** even in the profitable configuration.
- **Open interest and long/short ratio are display-only** until enough history
  accumulates to train on.
- **The 5-minute cadence is best-effort.** Backfill makes results correct
  regardless, but freshness is not guaranteed.

Nothing here is investment advice.

---

## Stack

Binance (data) · Parquet (archive) · PyTorch (model) · Kaggle (GPU) ·
Supabase (Postgres, auth, RLS, cron) · GitHub Actions (CI, retrain) ·
Next.js + Tailwind + TradingView lightweight-charts (dashboard) · Vercel (hosting) ·
Jev (strategy selection)

Every component is on a free tier.
