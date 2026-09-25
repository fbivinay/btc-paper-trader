"""Does letting the computer pick rules each January beat the fixed 8-signal team?

18 AI versions -- how rules are ranked each January (all-history CAGR, Sharpe,
last-3-year CAGR) x committee size (3, 5, 10) x decision frequency (daily,
weekly) -- next to the fixed team (etf_model.weights). Sorted by the 2019-2023
simulated result, the only period a real developer could have chosen on; the
real-ETF columns show what each choice then delivered.

    python ml/etf_robustness.py
"""

import itertools

import numpy as np
import pandas as pd

from etf_model import BTC_FEE, REAL_START, YEARS, backtest, load, rules, vol_scale, weights


def scores(px, sigs, rate, fee) -> dict:
    out = {"cagr": {}, "sharpe": {}, "cagr3": {}}
    for y in YEARS:
        c, s, c3 = {}, {}, {}
        for n, sig in sigs.items():
            w = {"X": pd.Series(sig, px.index)}
            r = backtest({"X": px}, w, rate, end=f"{y - 1}-12-31", fees={"X": fee})
            d = r.equity.pct_change().dropna()
            c[n], s[n] = r.cagr(), (d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0.0)
            c3[n] = backtest({"X": px}, w, rate, start=f"{y - 3}-01-01", end=f"{y - 1}-12-31",
                             fees={"X": fee}).cagr()
        out["cagr"][y], out["sharpe"][y], out["cagr3"][y] = pd.Series(c), pd.Series(s), pd.Series(c3)
    return out


def committee(px, sigs, sc, k) -> pd.Series:
    w = pd.Series(0.0, index=px.index)
    for y in YEARS:
        now = px.index.year == y
        w[now] = np.mean([sigs[n][now] for n in sc[y].nlargest(k).index], axis=0)
    return w


def weekly(w: pd.Series) -> pd.Series:
    """Only the last trading day of each week may change the position."""
    last = w.index.to_series().groupby(w.index.to_period("W")).transform("max") == w.index
    return w.where(last).ffill().fillna(0.0)


def main() -> None:
    btc, ibit, gld, rate = load()
    sb, sg = rules(btc["close"]), rules(gld["close"])
    SB, SG = scores(btc, sb, rate, BTC_FEE), scores(gld, sg, rate, 0.0)
    vb, vg = vol_scale(btc["close"]), vol_scale(gld["close"])
    rows = []

    def judge(rank, k, freq, w):
        dev = backtest({"BTC": btc, "GLD": gld}, w, rate, start="2019-01-01", end="2023-12-31",
                       fees={"BTC": BTC_FEE})
        real = backtest({"BTC": ibit, "GLD": gld}, w, rate, start=REAL_START)
        rows.append((rank, k, freq, dev.cagr(), dev.max_dd(), dev.cagr() / abs(dev.max_dd()),
                     real.cagr(), real.max_dd()))

    for rank, k, freq in itertools.product(("cagr", "sharpe", "cagr3"), (3, 5, 10), ("daily", "weekly")):
        wb = committee(btc, sb, SB[rank], k) * vb
        wg = (1 - wb) * committee(gld, sg, SG[rank], k) * vg
        if freq == "weekly":
            wb, wg = weekly(wb), weekly(wg)
        judge(rank, k, freq, {"BTC": wb, "GLD": wg})
    fixed = weights(btc, gld)
    judge("FIXED TEAM", 8, "daily", fixed)
    judge("FIXED, BTC only", 8, "daily", {"BTC": fixed["BTC"]})

    df = pd.DataFrame(rows, columns=["rank", "k", "freq", "2019-23 CAGR", "2019-23 maxDD",
                                     "2019-23 CAGR/DD", "real CAGR", "real maxDD"])
    fmt = {c: "{:+.1%}".format for c in df.columns[3:]} | {"2019-23 CAGR/DD": "{:.2f}".format}
    print(df.sort_values("2019-23 CAGR/DD", ascending=False).to_string(index=False, formatters=fmt))
    ai = df[df["k"] != 8]
    print(f"\nreal ETF CAGR across the 18 AI versions: min {ai['real CAGR'].min():+.1%}, "
          f"median {ai['real CAGR'].median():+.1%}, max {ai['real CAGR'].max():+.1%};  "
          f"worst drop between {ai['real maxDD'].max():.0%} and {ai['real maxDD'].min():.0%}")


if __name__ == "__main__":
    main()
