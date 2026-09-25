"""Regime-switching meta-strategy: use whatever worked before in situations like this.

Every month the system reads the current market situation, looks back at past
months that looked the same, finds which strategy did best in them, and runs
that strategy for the coming month.

The rule that makes this honest: when deciding at the start of month k, it may
only learn from months whose outcome was FULLY KNOWN by then -- months that
ended at or before the decision. It never sees how the month it is choosing for
turns out. That is the only position a live system is ever in.

    python ml/regime_switch.py
"""

import numpy as np
import pandas as pd

from hindsight_search import hysteresis
from india_tax_sim import Costs, simulate_ls

COST_PER_SWITCH = 0.003     # rough in-month friction for choosing between strategies
MIN_OBS = 3                 # past months of the same regime needed to trust it
SWITCH_MARGIN = 0.01        # a new strategy must beat the current one by 1%/month


def load():
    d = pd.read_parquet("../data/BTCUSDT_4h.parquet")
    d["open_time"] = pd.to_datetime(d["open_time"])
    f = pd.read_parquet("../data/BTCUSDT_funding.parquet")
    f["bar"] = pd.to_datetime(f["funding_time"]).dt.tz_localize(None).dt.floor("4h")
    fb = f.groupby("bar")["funding_rate"].sum()
    d["funding"] = d["open_time"].dt.tz_localize(None).dt.floor("4h").map(fb).fillna(0.0).to_numpy()
    return d


def library(c: pd.Series) -> dict[str, np.ndarray]:
    """The strategies the system can choose between. Each position series uses
    only past closes; a signal at bar i is executed at bar i+1."""
    b = lambda s: s.fillna(False).to_numpy()
    n = len(c)
    lib = {"cash": np.zeros(n), "hold": np.ones(n), "short": -np.ones(n)}
    for tag, n_, band, mode in (("trend_slow", 800, 0.02, "flat"), ("trend_slow_ls", 800, 0.02, "ls"),
                                ("trend_mid", 300, 0.07, "flat"), ("trend_mid_ls", 400, 0.07, "ls")):
        s = c.rolling(n_).mean(); up, dn = b(c > s*(1+band)), b(c < s*(1-band))
        lib[tag] = hysteresis(up, dn, b(c < s), b(c > s)) if mode == "flat" else hysteresis(up, dn, dn, up)
    r = c.pct_change(360)
    lib["momentum"] = hysteresis(b(r > 0.08), b(r < -0.08), b(r < 0), b(r > 0))
    fm, sm = c.rolling(60).mean(), c.rolling(300).mean()
    lib["crossover"] = np.where(b(fm > sm), 1.0, np.where(b(fm < sm), -1.0, 0.0))
    mm, sd = c.rolling(30).mean(), c.rolling(30).std(); zz = (c - mm) / sd
    lib["mean_revert"] = hysteresis(b(zz < -2.0), b(zz > 2.0), b(zz > 0), b(zz < 0))
    return lib


def features(d: pd.DataFrame) -> pd.DataFrame:
    """The 'situation' at each bar, from past data only."""
    c = d["close"]
    sma = c.rolling(600).mean()
    ret = c.pct_change()
    vol = ret.rolling(180).std()
    f = pd.DataFrame(index=d.index)
    f["trend_pos"] = c / sma - 1
    f["trend_slope"] = sma / sma.shift(180) - 1
    f["vol_pct"] = vol.rolling(2190, min_periods=360).rank(pct=True)
    f["mom_30d"] = c.pct_change(180)
    f["dd_1y"] = c / c.rolling(2190, min_periods=360).max() - 1
    fund = d["funding"].rolling(42).sum()
    f["fund_z"] = (fund - fund.rolling(540, min_periods=90).mean()) / fund.rolling(540, min_periods=90).std()
    return f


def regime(row) -> str:
    """Eight situations from three questions."""
    t = "up" if row["trend_pos"] > 0 else "down"
    v = "hivol" if row["vol_pct"] > 0.5 else "lovol"
    m = "mom+" if row["mom_30d"] > 0 else "mom-"
    return f"{t}/{v}/{m}"


def monthly_outcomes(d, lib):
    """Forward return of every strategy over every calendar month, pre-tax, with
    a friction charge for each position change inside the month."""
    c = d["close"].to_numpy()
    t = d["open_time"].dt.tz_localize(None)
    r = np.r_[0.0, c[1:] / c[:-1] - 1]
    starts = np.flatnonzero((t.dt.month != t.shift(1).dt.month).to_numpy())
    rows = []
    for a, b_ in zip(starts[:-1], starts[1:]):
        out = {"k0": a, "k1": b_, "time": t.iloc[a]}
        for name, pos in lib.items():
            p = pos[a:b_]                       # position held going into each bar
            gross = np.prod(1 + p[:-1] * r[a + 1:b_]) - 1
            flips = np.count_nonzero(np.diff(p))
            out[name] = gross - flips * COST_PER_SWITCH
        rows.append(out)
    return pd.DataFrame(rows)


def walk_forward(d, lib, mode="table"):
    """Choose a strategy for each month using only fully finished past months."""
    feats = features(d)
    mo = monthly_outcomes(d, lib)
    mo["regime"] = [regime(feats.iloc[k]) for k in mo["k0"]]
    names = list(lib)
    combined = np.zeros(len(d)); choices = []; current = "cash"

    if mode == "ml":
        from sklearn.ensemble import HistGradientBoostingRegressor
        fcols = list(feats.columns)
        mo_X = feats.iloc[mo["k0"]].to_numpy()

    for i, row in mo.iterrows():
        known = mo[(mo["k1"] <= row["k0"])]          # months already finished
        if mode == "table":
            same = known[known["regime"] == row["regime"]]
            pool = same if len(same) >= MIN_OBS else known
            scores = pool[names].mean() if len(pool) else pd.Series(0.0, index=names)
        else:
            if len(known) < 12:
                scores = pd.Series(0.0, index=names)
            else:
                Xk = mo_X[known.index]; xk = mo_X[i:i + 1]
                ok = ~np.isnan(Xk).any(1)
                scores = pd.Series(0.0, index=names)
                if ok.sum() >= 12 and not np.isnan(xk).any():
                    for s in names:
                        m = HistGradientBoostingRegressor(max_iter=60, max_depth=2,
                                                          learning_rate=0.05, random_state=0)
                        m.fit(Xk[ok], known[s].to_numpy()[ok])
                        scores[s] = m.predict(xk)[0]
        best = scores.idxmax()
        if scores[best] <= 0:
            best = "cash"
        if best != current and scores[best] - scores.get(current, 0.0) < SWITCH_MARGIN:
            best = current
        current = best
        combined[row["k0"]:row["k1"]] = lib[best][row["k0"]:row["k1"]]
        choices.append((row["time"], row["regime"], best))
    return combined, pd.DataFrame(choices, columns=["month", "regime", "strategy"])


def _self_check() -> None:
    """A decision for month k must not change when the FUTURE changes."""
    d = load().iloc[:6000].reset_index(drop=True)
    lib = library(d["close"])
    _, ch_a = walk_forward(d, lib, "table")
    d2 = d.copy()
    d2.loc[5000:, ["open", "high", "low", "close"]] *= 3.0
    _, ch_b = walk_forward(d2, library(d2["close"]), "table")
    cut = pd.Timestamp(d["open_time"].dt.tz_localize(None).iloc[4800])
    a = ch_a[ch_a["month"] < cut]["strategy"].tolist()
    b = ch_b[ch_b["month"] < cut]["strategy"].tolist()
    assert a == b, "a past decision changed when future prices changed: lookahead"
    print("regime switch self-check passed: decisions never read the future")


def main() -> None:
    d = load(); lib = library(d["close"])
    t = d["open_time"].dt.tz_localize(None)
    m = (t >= "2018-01-01").to_numpy(); sub = d[m].reset_index(drop=True)
    fund = sub["funding"].to_numpy()
    costs = Costs()

    print("REGIME-SWITCHING AI  |  learns only from finished months  |  every Indian charge\n")
    results = {"buy & hold": lib["hold"], "trend_slow alone": lib["trend_slow"]}
    for mode in ("table", "ml"):
        sig, ch = walk_forward(d, lib, mode)
        results[f"regime AI ({mode})"] = sig
        if mode == "table":
            table_choices = ch

    header = None
    for name, sig in results.items():
        r = simulate_ls(sub, sig[m], funding_daily=fund, costs=costs)
        y = r.yearly()
        if header is None:
            print(f"{'':<22} {'CAGR':>7} {'worst':>7} {'maxDD':>7}  " + " ".join(f"{yr:>6}" for yr in y.index))
            header = 1
        print(f"{name:<22} {r.cagr():>+7.1%} {y.min():>+7.0%} {r.max_dd():>7.0%}  "
              + " ".join(f"{v:>+6.0%}" for v in y.values)
              + f"   loss yrs {(y < 0).sum()}")

    tc = table_choices[table_choices["month"] >= "2018-01-01"]
    print(f"\nwhat the table AI chose, by count: " +
          ", ".join(f"{k} {v}" for k, v in tc["strategy"].value_counts().items()))
    print(f"strategy changes: {(tc['strategy'] != tc['strategy'].shift()).sum() - 1}")


if __name__ == "__main__":
    import sys
    _self_check() if "--check" in sys.argv else main()
