"""ETF model: when to hold a US spot Bitcoin ETF -- and gold when Bitcoin is not trending.

The model, decided after the close each US trading day, traded at the next open:
  1. a team of 8 standard trend signals votes on Bitcoin; its weight is the share
     voting "hold" (0, 1/8, ... 1)
  2. that weight shrinks when Bitcoin's volatility runs above its past-year norm
  3. whatever Bitcoin does not use goes to gold, sized by the same team on gold
  4. the rest sits in T-bills
Nothing in it is fitted to data. The AI alternative -- re-picking the best rules
every January from past results -- is kept for comparison: it did worse.

Judged twice, after every Indian charge and tax (see etf_tax_sim.py):
  simulated  BTC's own price at US market hours stands in for the ETF, 2019 onward
  real       IBIT's actual prices since its launch on 2024-01-11

    python ml/fetch_etf.py      # ETF prices and T-bill rates
    python ml/etf_model.py
"""

import numpy as np
import pandas as pd

from etf_tax_sim import simulate
from hindsight_search import candidates
from strategies import grid

DATA = "../data"
BTC_FEE = 0.0025             # spot Bitcoin ETF expense ratio, charged on simulated holdings
YEARS = range(2019, 2027)
REAL_START = "2024-01-11"    # IBIT's first trading day
TOP = 5                      # rules in the committee
TARGET = 0.25


def load():
    """Open/close per US trading day: BTC at US hours (the simulated ETF), real IBIT, GLD."""
    gld = pd.read_parquet(f"{DATA}/etf_GLD.parquet").set_index("date")
    ibit = pd.read_parquet(f"{DATA}/etf_IBIT.parquet").set_index("date")
    rate = pd.read_parquet(f"{DATA}/tbill.parquet").set_index("date")["rate"]
    b = pd.read_parquet(f"{DATA}/BTCUSDT_4h.parquet")
    t = pd.to_datetime(b["open_time"]).dt.tz_localize(None)

    def at(hour, col):
        m = (t.dt.hour == hour).to_numpy()
        return pd.Series(b[col].to_numpy()[m], index=t[m].dt.normalize().to_numpy())

    cal = gld.index[gld.index >= t.iloc[0].normalize()]
    btc = pd.DataFrame({"open": at(12, "open"),     # 12:00 UTC, just before the US open
                        "close": at(16, "close")})  # 20:00 UTC, the US close
    return btc.reindex(cal).ffill(), ibit[["open", "close"]], gld.loc[cal, ["open", "close"]], rate


def rules(close: pd.Series) -> dict:
    """Every long-or-cash trend rule tested (short legs become cash under LRS), deduplicated."""
    r = {n: fn(close, **kw) for n, kw, fn in grid()}
    r |= {n: np.clip(v, 0, 1) for n, v in candidates(close).items()}
    return dict({v.tobytes(): (n, v) for n, v in reversed(r.items())}.values())


def backtest(prices: dict, weights: dict, rate, start=None, end=None, fees=None):
    o = pd.DataFrame({k: p["open"] for k, p in prices.items()}).loc[start:end].dropna()
    c = pd.DataFrame({k: p["close"] for k, p in prices.items()}).loc[o.index]
    return simulate(o, c, pd.DataFrame(weights).reindex(o.index), rate=rate, expense=fees)


def yearly_scores(px, sigs, rate, fee) -> dict:
    """For each January: every rule's after-tax CAGR on all data BEFORE that year."""
    return {y: pd.Series({n: backtest({"X": px}, {"X": pd.Series(s, px.index)}, rate,
                                      end=f"{y - 1}-12-31", fees={"X": fee}).cagr()
                          for n, s in sigs.items()}) for y in YEARS}


def committee(px, sigs, scores, k) -> pd.Series:
    """The k best rules so far vote; the weight is the share voting 'hold'."""
    w = pd.Series(0.0, index=px.index)
    for y in YEARS:
        now = px.index.year == y
        w[now] = np.mean([sigs[n][now] for n in scores[y].nlargest(k).index], axis=0)
    return w


def vol_scale(close: pd.Series) -> pd.Series:
    """Shrink the position when volatility runs above its own past-year norm."""
    v = np.log(close).diff().rolling(20).std()
    return (v.rolling(252, min_periods=60).median() / v).clip(upper=1.0).fillna(0.0)


def team(close: pd.Series) -> pd.Series:
    """8 standard trend signals vote: price above its 1/2.5/5/10-month average, and
    up over the last 1/3/6/12 months. Returns the share voting 'hold'."""
    votes = [close > close.rolling(n).mean() for n in (20, 50, 100, 200)]
    votes += [close > close.shift(n) for n in (21, 63, 126, 252)]
    return pd.concat(votes, axis=1).mean(axis=1)


def weights(btc: pd.DataFrame, gld: pd.DataFrame) -> dict:
    """The model: Bitcoin by its team vote and volatility, gold fills what Bitcoin leaves."""
    wb = team(btc["close"]) * vol_scale(btc["close"])
    return {"BTC": wb, "GLD": (1 - wb) * team(gld["close"]) * vol_scale(gld["close"])}


def _self_check(btc: pd.DataFrame, gld: pd.DataFrame) -> None:
    """A weight decided on day t must not change when LATER prices change."""
    cut = btc.index[len(btc) // 2]
    b2, g2 = btc.copy(), gld.copy()
    b2.loc[cut:, "close"] *= 3.0; g2.loc[cut:, "close"] *= 0.5
    a, b = pd.DataFrame(weights(btc, gld)), pd.DataFrame(weights(b2, g2))
    assert np.allclose(a[a.index < cut], b[b.index < cut]), "the model reads the future"


def main() -> None:
    btc, ibit, gld, rate = load()
    _self_check(btc, gld)
    sb, sg = rules(btc["close"]), rules(gld["close"])
    print(f"scoring {len(sb)} Bitcoin and {len(sg)} gold rules for each year {YEARS[0]}-{YEARS[-1]}...",
          flush=True)
    score_b = yearly_scores(btc, sb, rate, BTC_FEE)
    score_g = yearly_scores(gld, sg, rate, 0.0)     # real GLD prices already net its fee

    ai_b = committee(btc, sb, score_b, TOP) * vol_scale(btc["close"])
    ai_g = (1 - ai_b) * committee(gld, sg, score_g, TOP) * vol_scale(gld["close"])
    new = weights(btc, gld)
    models = {                                   # name: (weights, idle cash earns T-bills)
        "buy & hold": ({"BTC": pd.Series(1.0, btc.index)}, True),
        "old: AI picks 1 rule a year": ({"BTC": committee(btc, sb, score_b, 1)}, False),
        f"AI picks {TOP} rules + gold": ({"BTC": ai_b, "GLD": ai_g}, True),
        "NEW: 8-signal team, BTC only": ({"BTC": new["BTC"]}, True),
        "NEW: 8-signal team + gold": (new, True),
    }

    sim = {n: backtest({"BTC": btc, "GLD": gld}, w, rate if cash else None, start="2019-01-01",
                       fees={"BTC": BTC_FEE}) for n, (w, cash) in models.items()}
    real = {n: backtest({"BTC": ibit, "GLD": gld}, w, rate if cash else None, start=REAL_START)
            for n, (w, cash) in models.items()}

    end = btc.index[-1]
    print(f"\nSIMULATED ETF (BTC price), 2019-01-01..{end:%Y-%m-%d}, after all charges and tax")
    print(f"{'':<30} {'CAGR':>6} {'19-23':>6} {'maxDD':>6}  "
          + " ".join(f"{y:>5}" for y in YEARS) + "  lose  >=25%")
    for n, r in sim.items():
        y = r.yearly()
        dev = np.prod(1 + y.loc[2019:2023]) ** (1 / 5) - 1
        print(f"{n:<30} {r.cagr():>+6.1%} {dev:>+6.1%} {r.max_dd():>6.0%}  "
              + " ".join(f"{v:>+5.0%}" for v in y) + f"  {(y < 0).sum():>4}  {(y >= TARGET).sum():>4}")

    print(f"\nREAL ETF (IBIT + GLD prices), {REAL_START}..{end:%Y-%m-%d}   -- never used for choosing")
    print(f"{'':<30} {'CAGR':>6} {'maxDD':>6}  {'2024':>5} {'2025':>5} {'2026':>5}   trades")
    for n, r in real.items():
        print(f"{n:<30} {r.cagr():>+6.1%} {r.max_dd():>6.0%}  " + " ".join(f"{v:>+5.0%}" for v in r.yearly())
              + f"   {len(r.trades):>6}")

    a = ibit["close"].pct_change(); b = btc["close"].reindex(ibit.index).pct_change()
    print(f"\nsimulated vs real daily moves 2024+: correlation {a.corr(b):.3f}")
    r = real["NEW: 8-signal team + gold"]
    yrs = (r.equity.index[-1] - r.equity.index[0]).days / 365.25
    w = pd.DataFrame(new).loc[REAL_START:]
    print(f"NEW model on real ETFs: {len(r.trades) / yrs:.0f} trades a year; average money in "
          f"Bitcoin {w['BTC'].mean():.0%}, gold {w['GLD'].mean():.0%}, T-bills {1 - w.sum(axis=1).mean():.0%}")


if __name__ == "__main__":
    main()
