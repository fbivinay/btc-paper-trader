"""Every strategy tried for the ETF model, judged the same honest way.

Each candidate turns real daily prices into target weights. Two judgements:
  DEV   2019-2023. IBIT did not exist, so Bitcoin's own real price stands in
        (with the ETF's 0.25% fee charged). The ONLY period used to choose.
  REAL  IBIT's actual prices from its launch, 2024-01-11. Never used to choose.
Both after Indian tax and all charges; REAL also before them.

The choice rule, fixed before any result was seen: best DEV return per unit of
worst drop (CAGR / max drawdown).

    python ml/etf_research.py
"""

import numpy as np
import pandas as pd

from etf_data import load, yahoo
from etf_model import MOMENTUM_DAYS, SMA_DAYS, vol_scale
from etf_tax_sim import GROSS, run, simulate

BTC_FEE = {"BTC": 0.0025}      # charged on the pre-launch stand-in only
DEV = ("2019-01-01", "2023-12-31")
REAL = "2024-01-11"
YEARS = range(2019, 2027)


# ------------------------------------------------------------ building blocks
def share(close, sma=SMA_DAYS, mom=MOMENTUM_DAYS, sized=True) -> pd.Series:
    """Share of trend signals voting 'hold', optionally shrunk by volatility."""
    v = [close > close.rolling(n).mean() for n in sma] + [close > close.shift(n) for n in mom]
    f = pd.concat(v, axis=1).mean(axis=1)
    return f * vol_scale(close) if sized else f


def priority(*fs) -> list:
    """Each asset in turn takes its share of whatever the ones before it left."""
    left, out = 1.0, []
    for f in fs:
        out.append(left * f)
        left = left - out[-1]
    return out


def weekly(w: pd.Series) -> pd.Series:
    """Only the last trading day of each week may change the position."""
    last = w.index.to_series().groupby(w.index.to_period("W")).transform("max") == w.index
    return w.where(last).ffill().fillna(0.0)


def _hold(enter, leave) -> np.ndarray:
    out, pos = np.zeros(len(enter)), 0
    for i in range(len(out)):
        pos = 1 if (pos == 0 and enter[i]) else 0 if (pos == 1 and leave[i]) else pos
        out[i] = pos
    return out


def rule_library(c: pd.Series) -> dict:
    """The single long-or-cash rules an AI picker can choose from."""
    b = lambda s: s.fillna(False).to_numpy()
    lib = {}
    for n in (20, 50, 100, 150, 200):
        m = c.rolling(n).mean()
        for band in (0.0, 0.03, 0.06, 0.1):
            lib[f"sma{n}/{band}"] = _hold(b(c > m * (1 + band)), b(c < m * (1 - band)))
    for n in (7, 14, 30, 60, 90, 180):
        r = c.pct_change(n)
        for band in (0.0, 0.05, 0.1, 0.2):
            lib[f"mom{n}/{band}"] = _hold(b(r > band), b(r < -band))
    for f, s in ((5, 20), (10, 30), (20, 50), (20, 100), (50, 100), (50, 200), (100, 200)):
        lib[f"x{f}/{s}"] = b(c.rolling(f).mean() > c.rolling(s).mean()).astype(float)
    for e, x in ((10, 5), (20, 10), (55, 20), (100, 50)):
        lib[f"brk{e}/{x}"] = _hold(b(c > c.rolling(e).max().shift(1)), b(c < c.rolling(x).min().shift(1)))
    return lib


# --------------------------------------------------------------- candidates
def ai_picker(px, rate, k=5) -> dict:
    """Each January, the k rules with the best after-tax CAGR on EARLIER data vote."""
    out = {}
    for name, fee in (("BTC", BTC_FEE["BTC"]), ("GLD", 0.0)):
        p = px[name]; lib = rule_library(p["close"])
        w = pd.Series(0.0, index=p.index)
        for y in YEARS:
            past = p.index < f"{y}-01-01"
            score = {n: simulate(p.loc[past, ["open"]].set_axis(["X"], axis=1),
                                 p.loc[past, ["close"]].set_axis(["X"], axis=1),
                                 pd.DataFrame({"X": s[past]}, index=p.index[past]),
                                 rate, {"X": fee}).cagr() for n, s in lib.items()}
            now = p.index.year == y
            w[now] = np.mean([lib[n][now] for n in sorted(score, key=score.get)[-k:]], axis=0)
        out[name] = w * vol_scale(p["close"])
    out["BTC"], out["GLD"] = priority(out["BTC"], out["GLD"])
    return out


def ml_model(px, rate) -> dict:
    """Gradient boosting predicts whether Bitcoin is higher in 21 days; retrained
    every January on earlier data only, with the last 21 days purged so no
    training label overlaps the year it predicts."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    c, g = px["BTC"]["close"], px["GLD"]["close"]
    X = pd.DataFrame({**{f"sma{n}": c / c.rolling(n).mean() - 1 for n in SMA_DAYS},
                      **{f"mom{n}": c / c.shift(n) - 1 for n in MOMENTUM_DAYS},
                      "vol": np.log(c).diff().rolling(20).std(),
                      "drawdown": c / c.rolling(252).max() - 1,
                      "gold63": g / g.shift(63) - 1,
                      "tbill": rate.reindex(c.index, method="ffill")})
    y = (c.shift(-21) > c).astype(float).where(c.shift(-21).notna())
    ok = X.notna().all(axis=1).to_numpy()
    p = pd.Series(np.nan, index=c.index)
    for yr in YEARS:
        first = int(np.argmax(c.index.year == yr))
        train = ok & (np.arange(len(c)) < first - 21) & y.notna().to_numpy()
        test = ok & (c.index.year == yr)
        if train.sum() < 250:
            continue                                   # not enough history to learn from yet
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           random_state=0).fit(X[train], y[train])
        p[test] = m.predict_proba(X[test])[:, 1]
    f = ((p - 0.45) / 0.2).clip(0, 1).fillna(0.0) * vol_scale(c)
    return dict(zip(("BTC", "GLD"), priority(f, share(g))))


def candidates(px, rate) -> dict:
    b, g = px["BTC"]["close"], px["GLD"]["close"]
    q, t = px["QQQ"]["close"], px["TLT"]["close"]
    two = lambda fb, fg: dict(zip(("BTC", "GLD"), priority(fb, fg)))
    model = two(share(b), share(g))
    both = share(b) + share(g)
    return {
        "MODEL: 8 votes, vol sizing, gold fill": model,
        "Bitcoin only (no gold)": {"BTC": share(b)},
        "no volatility sizing": two(share(b, sized=False), share(g, sized=False)),
        "SMA votes only": two(share(b, mom=()), share(g, mom=())),
        "momentum votes only": two(share(b, sma=()), share(g, sma=())),
        "slower signals": two(share(b, (50, 100, 150, 200), (63, 126, 189, 252)),
                              share(g, (50, 100, 150, 200), (63, 126, 189, 252))),
        "faster signals": two(share(b, (10, 20, 50, 100), (10, 21, 63, 126)),
                              share(g, (10, 20, 50, 100), (10, 21, 63, 126))),
        "all-or-nothing (5+ of 8 votes)": two((share(b, sized=False) >= 5 / 8) * vol_scale(b),
                                              (share(g, sized=False) >= 5 / 8) * vol_scale(g)),
        "weekly decisions": {k: weekly(v) for k, v in model.items()},
        "gold first, then Bitcoin": dict(zip(("GLD", "BTC"), priority(share(g), share(b)))),
        "equal footing (scaled to fit)": {"BTC": share(b) / both.clip(lower=1),
                                          "GLD": share(g) / both.clip(lower=1)},
        "+ Nasdaq-100 (QQQ)": dict(zip(("BTC", "GLD", "QQQ"), priority(share(b), share(g), share(q)))),
        "+ Nasdaq-100 + long bonds (TLT)": dict(zip(("BTC", "GLD", "QQQ", "TLT"),
                                                    priority(share(b), share(g), share(q), share(t)))),
        "AI picks 5 rules a year": ai_picker(px, rate),
        "machine learning (gradient boosting)": ml_model(px, rate),
    }


# ------------------------------------------------------------------ judging
def judge(px, w, rate) -> dict:
    p = {k: px[k] for k in w}
    dev = run(p, w, rate, *DEV, expense=BTC_FEE)
    real = run(p, w, rate, REAL)
    gross = run(p, w, rate, REAL, costs=GROSS)
    yd = dev.yearly()
    return {"dev_cagr": dev.cagr(), "dev_dd": dev.max_dd(), "dev_score": dev.cagr() / abs(dev.max_dd()),
            "dev_losing": int((yd < 0).sum()), "real_cagr": real.cagr(), "real_dd": real.max_dd(),
            "real_years": real.yearly(), "real_gross": gross.cagr(), "trades": len(real.trades)}


def main() -> None:
    btc, gold, rate, _ = load()
    px = {"BTC": btc, "GLD": gold}
    for tk in ("QQQ", "TLT"):
        px[tk] = yahoo(tk).reindex(btc.index).ffill()
    rows = {n: judge(px, w, rate) for n, w in candidates(px, rate).items()}
    rows["buy & hold IBIT"] = judge(px, {"BTC": pd.Series(1.0, btc.index)}, rate)

    t = pd.DataFrame(rows).T.sort_values("dev_score", ascending=False)
    print(f"DEV {DEV[0]}..{DEV[1]} chooses; REAL {REAL}..{btc.index[-1]:%Y-%m-%d} is the exam. "
          "After tax and charges unless marked.\n")
    print(f"{'':<40}{'DEV CAGR':>9}{'maxDD':>7}{'score':>7}{'lose':>5} |{'REAL CAGR':>10}{'maxDD':>7}"
          f"{'2024':>6}{'2025':>6}{'2026':>6}{'trades':>7}{'pre-tax':>9}")
    for n, r in t.iterrows():
        yr = " ".join(f"{v:>+5.0%}" for v in r["real_years"])
        print(f"{n:<40}{r['dev_cagr']:>+9.1%}{r['dev_dd']:>7.0%}{r['dev_score']:>7.2f}{r['dev_losing']:>5} |"
              f"{r['real_cagr']:>+10.1%}{r['real_dd']:>7.0%} {yr}{r['trades']:>7}{r['real_gross']:>+9.1%}")
    best = t.drop("buy & hold IBIT").index[0]
    print(f"\nchosen by the DEV rule: {best}")


if __name__ == "__main__":
    main()
