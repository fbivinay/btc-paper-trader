"""Can ANY strategy hit >=25% net in EVERY year, even with perfect hindsight?

This is an upper bound, not a strategy. Every candidate is scored on the same
years it is judged on, and the best is picked by looking at the answer. That is
exactly the cheating a real system must never do -- which is the point: if even
the best-of-hundreds-in-hindsight fails the target, no model trained only on
the past can meet it going forward, because hindsight is strictly more
information than any model will ever have.

    python ml/hindsight_search.py
"""

import itertools

import numpy as np
import pandas as pd

from india_tax_sim import Costs, simulate_ls

TARGET = 0.25
START = "2019-01-01"


def load():
    d = pd.read_parquet("../data/BTCUSDT_1d.parquet")
    d["open_time"] = pd.to_datetime(d["open_time"])
    f = pd.read_parquet("../data/BTCUSDT_funding.parquet")
    f["day"] = pd.to_datetime(f["funding_time"]).dt.tz_localize(None).dt.floor("D")
    daily = f.groupby("day")["funding_rate"].sum()
    days = d["open_time"].dt.tz_localize(None).dt.floor("D")
    d["funding"] = days.map(daily).fillna(0.0).to_numpy()
    return d


def hysteresis(enter_long, enter_short, exit_long, exit_short):
    """+1 / -1 / 0 state machine: enter on the enter condition, leave on the exit."""
    n = len(enter_long); out = np.zeros(n); pos = 0
    for i in range(n):
        if pos == 0:
            if enter_long[i]: pos = 1
            elif enter_short[i]: pos = -1
        elif pos == 1 and exit_long[i]:
            pos = -1 if enter_short[i] else 0
        elif pos == -1 and exit_short[i]:
            pos = 1 if enter_long[i] else 0
        out[i] = pos
    return out


def candidates(c: pd.Series):
    b = lambda s: s.fillna(False).to_numpy()
    out = {}
    # 1. price vs moving average, long above / short below, with a neutral band
    for n, band, allow_flat in itertools.product((20, 50, 100, 150, 200), (0.0, 0.03, 0.06, 0.1),
                                                 (True, False)):
        s = c.rolling(n).mean()
        up, dn = b(c > s * (1 + band)), b(c < s * (1 - band))
        if allow_flat:
            out[f"sma{n}_b{band}_flat"] = hysteresis(up, dn, b(c < s), b(c > s))
        else:
            out[f"sma{n}_b{band}_ls"] = hysteresis(up, dn, dn, up)
    # 2. time-series momentum
    for lb, band in itertools.product((7, 14, 30, 60, 90, 180), (0.0, 0.05, 0.1, 0.2)):
        r = c.pct_change(lb)
        up, dn = b(r > band), b(r < -band)
        out[f"mom{lb}_b{band}"] = hysteresis(up, dn, b(r < 0), b(r > 0))
    # 3. moving-average crossovers, always in the market
    for f, s in ((5, 20), (10, 30), (20, 50), (20, 100), (50, 100), (50, 200), (100, 200)):
        fm, sm = c.rolling(f).mean(), c.rolling(s).mean()
        out[f"x{f}_{s}"] = np.where(b(fm > sm), 1.0, np.where(b(fm < sm), -1.0, 0.0))
    # 4. breakouts
    for e, x in ((10, 5), (20, 10), (55, 20), (100, 50)):
        hi, lo = c.rolling(e).max().shift(1), c.rolling(e).min().shift(1)
        xl, xh = c.rolling(x).min().shift(1), c.rolling(x).max().shift(1)
        out[f"brk{e}_{x}"] = hysteresis(b(c > hi), b(c < lo), b(c < xl), b(c > xh))
    # 5. mean reversion: fade stretched moves
    for n, z in itertools.product((10, 20, 50), (1.5, 2.0, 2.5)):
        m, sd = c.rolling(n).mean(), c.rolling(n).std()
        zz = (c - m) / sd
        out[f"mr{n}_z{z}"] = hysteresis(b(zz < -z), b(zz > z), b(zz > 0), b(zz < 0))
    return out


def main() -> None:
    d = load(); c = d["close"]
    t = d["open_time"].dt.tz_localize(None)
    m = (t >= START).to_numpy()
    sub = d[m].reset_index(drop=True)
    costs = Costs()

    rows = []
    cands = candidates(c)
    print(f"searching {len(cands)} strategies with PERFECT HINDSIGHT, {START} onward\n", flush=True)
    for name, sig in cands.items():
        r = simulate_ls(sub, sig[m], funding_daily=sub["funding"].to_numpy(), costs=costs)
        y = r.yearly()
        rows.append((name, r.cagr(), y.min(), (y >= TARGET).sum(), len(y), y))

    rows.sort(key=lambda z: z[2], reverse=True)
    years = rows[0][5].index
    perfect = [z for z in rows if z[3] == z[4]]
    print(f"strategies with >=25% net in EVERY year: {len(perfect)} of {len(rows)}\n")
    print(f"TOP 12 BY WORST YEAR (the number that has to clear 25%)")
    print(f"{'strategy':<22} {'CAGR':>7} {'worst yr':>9} {'yrs>=25%':>9}  "
          + " ".join(f"{y:>6}" for y in years))
    for name, cg, worst, hit, n, y in rows[:12]:
        print(f"{name:<22} {cg:>+7.1%} {worst:>+9.1%} {hit:>6}/{n}  "
              + " ".join(f"{v:>+6.0%}" for v in y.values))
    best_hits = max(z[3] for z in rows)
    print(f"\nmost years any single strategy cleared 25%: {best_hits} of {len(years)}")
    by_year = pd.DataFrame({z[0]: z[5] for z in rows}).T
    print(f"\nBEST ANY STRATEGY ACHIEVED, EACH YEAR (different strategy per year, pure hindsight)")
    print("  " + "  ".join(f"{y}: {by_year[y].max():+.0%}" for y in years))


if __name__ == "__main__":
    main()
