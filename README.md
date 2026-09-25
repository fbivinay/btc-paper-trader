# Bitcoin ETF Trend Model

A daily trend model that decides how much to hold in a US spot Bitcoin ETF
(IBIT), a gold ETF (GLD) and T-bills, built for an Indian resident investing
under the Liberalised Remittance Scheme. It runs by itself every US trading day
as a paper portfolio on **real ETF prices**, after **Indian tax and all charges**,
next to buy & hold. Going live with real money is a configuration change: add
broker keys and switch orders on.

**Live dashboard:** https://btc-paper-trader-fbivinays-projects.vercel.app
**Going live with real money:** https://btc-paper-trader-fbivinays-projects.vercel.app/go-live

---

## Results on IBIT's real prices

$10,000 from IBIT's launch (11 Jan 2024) to 24 Sep 2026. "After" means Indian
capital-gains tax at the top slab (31.2%, 13% after 24 months), 1.5% forex each
way, 0.05% per trade, T-bill income taxed at slab. Equity is liquidation value:
what you would get back if you sold that day.

| | per year | worst drop | 2024 | 2025 | 2026 (to Sep) |
|---|---|---|---|---|---|
| **Model, after tax & charges** | **+26.4%** | **−15%** | +55% | +14% | +6% |
| Model, before tax & charges | +42.8% | −19% | +94% | +23% | +9% |
| Buy & hold IBIT, after | +20.5% | −43% | +64% | −5% | +6% |
| Buy & hold IBIT, before | +24.6% | −53% | +101% | −6% | −4% |

On real prices the model returned more than holding, with about a third of the
worst drop, and no losing year. It does not reach 25% in every year.

The top slab is the conservative case. Tax is the biggest single cost, so the
dashboard lets you pick your own slab. At the 20% slab the model made +30.6% a
year on real prices; at 10%, +34.9%. Buy & hold barely moves, because gains on
an ETF held over 24 months are taxed at a flat 12.5% plus cess, whatever the slab.

## How it decides

Once a day, after the US close, from finished daily bars; traded at the next open.

1. **Eight trend signals vote on Bitcoin**: is the price above its 20, 50, 100
   and 200-day average, and higher than 1, 3, 6 and 12 months ago? The share of
   "yes" votes is the Bitcoin weight.
2. **Volatility sizing**: when Bitcoin swings harder than its own past-year
   norm, the weight is scaled down.
3. **Gold fills what Bitcoin leaves**, sized by the same eight votes on gold.
4. **The rest earns T-bill interest.** No leverage, no short selling.

Nothing is fitted to data: every lookback is a standard one, chosen in advance.

## How it was chosen

`ml/etf_research.py` judges 15 strategies the same way. They are chosen on
**2019–2023 only**, using Bitcoin's own real price in place of the ETF, which did
not exist yet. They are then tested once on **IBIT's real prices from 2024**, a
period never used to choose anything.

| Strategy | 2019–23 per year / worst drop | Real IBIT per year / worst drop |
|---|---|---|
| **The model** | +33.1% / −34% | **+26.4% / −15%** |
| Bitcoin only (no gold) | +33.1% / −31% | +20.7% / −14% |
| No volatility sizing | +41.5% / −41% | +26.3% / −18% |
| AI re-picks 5 rules every January | +27.8% / −51% | +24.1% / −19% |
| Machine learning (gradient boosting) | +1.7% / −67% | +27.0% / −22% |
| Buy & hold | +56.5% / −76% | +20.5% / −43% |

The top three were essentially tied on 2019–23. The model was kept for its
second asset. Letting the computer re-pick rules each year, and the
gradient-boosting model, both did worse on the period used to choose. Other
variants tried were faster and slower signals, SMA or momentum votes only,
all-or-nothing, weekly decisions, gold first, and adding Nasdaq-100 and long
bonds.

## The automation

```
GitHub Actions (free)                 Supabase (free Postgres)        Vercel (free)
  22:30 UTC Mon-Fri  decide  ────────►  etf_decisions (write-once)  ─►  dashboard
    real prices (Yahoo) → checks          etf_equity, etf_trades          read-only,
    → model → ledger                      etf_runs (heartbeat)            no login
  14:45 UTC Mon-Fri  execute ─► broker   etf_orders (private)
    only if keys + SEND_ORDERS=on
```

- **Decide** refuses stale, missing or absurd data. It records each decision
  once, and a decision can never be edited, so the live record cannot be
  rewritten afterwards. It then rebuilds the paper portfolio after and before
  tax and charges, with buy & hold alongside.
- **Execute** is off unless broker keys exist and the `SEND_ORDERS` variable is
  `on`. When it runs it enforces these rules:
  - it sells first, waits for the fills, then buys **from cash only** (no leverage)
  - a sell never exceeds the shares held (no shorting)
  - each order is capped at `MAX_ORDER_USD`
  - every order carries an ID built from the decision date, so the broker rejects duplicates
- **Alert** goes out when the split moves enough to be worth a trade, about once
  a week. It is a free phone push via ntfy.sh, so the model works with **any**
  broker app, even ones with no API (INDmoney, Vested and others): you place the
  2–4 orders by hand.
- Every run writes to `etf_runs`, which the dashboard shows as its heartbeat. A
  failed run also makes GitHub email the owner.

## Repository

| Path | What |
|---|---|
| `ml/etf_model.py` | the model: votes, volatility sizing, weights |
| `ml/etf_data.py` | real daily prices (IBIT, GLD, T-bill, USD/INR) and data checks |
| `ml/btc_before_ibit.csv` | Bitcoin's real price at US market hours before IBIT existed |
| `ml/etf_tax_sim.py` | Indian tax simulator: FIFO lots, 24-month rule, loss set-off and 8-year carry-forward |
| `ml/etf_daily.py` | the daily decide / execute job |
| `ml/broker.py` | Alpaca client and the cash-only rebalance |
| `ml/etf_research.py` | all 15 strategies, chosen on 2019–23, tested on real 2024+ prices |
| `ml/db.py` | minimal Supabase REST client |
| `supabase/schema.sql` | tables (decisions, equity by tax slab, trades, alerts, runs, orders) and row-level security |
| `web/` | Next.js dashboard and the go-live guide |

Every module has an assert-based self-check (`python ml/<module>.py`). CI runs
them on every push.

## Running it yourself

```bash
pip install -r requirements.txt
python ml/etf_daily.py decide --dry-run      # today's decision and the portfolio, nothing written
pip install scikit-learn && python ml/etf_research.py   # the full comparison
```

To write to your own database, put `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in
`.env` locally, and in GitHub Actions secrets. The service key must never reach
the browser; the dashboard uses the anon key and read-only row-level security.

## Honest limits

- The real ETF record is 2.7 years long.
- Gold's 2024–25 rally helped the model. Without gold it made +20.7% a year on
  real prices.
- Tax rules are as understood at the time of writing; confirm with a Chartered
  Accountant. Results are in US dollar terms, and rupee moves affect every
  strategy alike.
- Prices come from Yahoo's free chart API. If it fails, the job stops rather
  than trading on bad data.

Paper trading. Not investment advice.
