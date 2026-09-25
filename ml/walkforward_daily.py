"""Walk-forward selection of a low-turnover strategy, with every Indian charge.

At the start of each test year the best candidate is chosen using ONLY the data
before that year, then traded for the year. The chained result is genuinely
out-of-sample: no parameter in it was chosen by looking at the period it is
judged on.

    python ml/walkforward_daily.py
"""

import numpy as np
import pandas as pd

from india_tax_sim import Costs, simulate
from strategies import grid

DATA = "../data/BTCUSDT_1d.parquet"


def load():
    d = pd.read_parquet(DATA)
    d["open_time"] = pd.to_datetime(d["open_time"])
    return d


def metric(res, kind: str) -> float:
    e = res.equity
    if len(e) < 60:
        return -np.inf
    if kind == "cagr":
        return res.cagr()
    if kind == "calmar":
        dd = res.max_dd()
        return res.cagr() / abs(dd) if dd < 0 else res.cagr()
    if kind == "worst_year":
        # maximise the WORST rolling 1-year return: aimed squarely at "never lose"
        r = (e / e.shift(365) - 1).dropna()
        return float(r.min()) if len(r) else -np.inf
    raise ValueError(kind)


def walk_forward(d, signals, years, kind, costs, sim=simulate):
    t = d["open_time"].dt.tz_localize(None)
    combined = np.zeros(len(d)); chosen = {}
    for y in years:
        past = (t < pd.Timestamp(f"{y}-01-01")).to_numpy()
        now = ((t >= pd.Timestamp(f"{y}-01-01")) & (t < pd.Timestamp(f"{y+1}-01-01"))).to_numpy()
        best, best_score = None, -np.inf
        for name, sig in signals.items():
            r = sim(d[past].reset_index(drop=True), sig[past], costs=costs)
            s = metric(r, kind)
            if s > best_score:
                best, best_score = name, s
        combined[now] = signals[best][now]
        chosen[y] = best
    start = (t >= pd.Timestamp(f"{years[0]}-01-01")).to_numpy()
    return combined, chosen, start


def main() -> None:
    d = load()
    close = d["close"]
    signals = {name: fn(close, **kw) for name, kw, fn in grid()}
    costs = Costs()
    years = list(range(2019, 2027))

    t = d["open_time"].dt.tz_localize(None)
    start = (t >= pd.Timestamp(f"{years[0]}-01-01")).to_numpy()
    oos = d[start].reset_index(drop=True)
    bh = simulate(oos, np.ones(len(oos)), costs=costs)

    print(f"OUT-OF-SAMPLE {years[0]}-01-01 .. {t.iloc[-1]:%Y-%m-%d}")
    print(f"charges: fee {costs.fee:.2%}+GST, slip {costs.slip:.2%}, TDS 1%, tax 31.2% no offset\n")
    print(f"{'strategy':<32} {'CAGR':>7} {'maxDD':>7} {'sells/yr':>9}  per-year net returns")
    yrs_bh = bh.yearly()
    print(f"{'buy & hold':<32} {bh.cagr():>+7.1%} {bh.max_dd():>7.1%} {0:>9.1f}  "
          + " ".join(f"{v:>+6.0%}" for v in yrs_bh.values))

    for kind in ("cagr", "calmar", "worst_year"):
        sig, chosen, st = walk_forward(d, signals, years, kind, costs)
        r = simulate(oos, sig[st], costs=costs)
        yrs = (r.equity.index[-1] - r.equity.index[0]).days / 365.25
        sells = (r.trades.side == "SELL").sum()
        print(f"{'walk-forward, pick by ' + kind:<32} {r.cagr():>+7.1%} {r.max_dd():>7.1%} "
              f"{sells/yrs:>9.1f}  " + " ".join(f"{v:>+6.0%}" for v in r.yearly().values))
        print(f"{'':<32} chosen: " + ", ".join(f"{y}:{n}" for y, n in chosen.items()))
    print(f"\nyears: {' '.join(f'{y:>6}' for y in yrs_bh.index)}")


if __name__ == "__main__":
    main()
