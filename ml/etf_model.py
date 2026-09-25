"""The ETF model: how much to hold in a spot Bitcoin ETF, a gold ETF and T-bills.

Decided after each US close from finished daily bars, traded at the next open:
  1. Eight trend signals vote on Bitcoin: price above its 20/50/100/200-day
     average, and higher than 21/63/126/252 trading days ago (1/3/6/12 months).
     The Bitcoin weight is the share voting "hold": 0, 1/8, ... 1.
  2. The weight shrinks when Bitcoin's 20-day volatility runs above its own
     past-year median. It never grows above the vote.
  3. Whatever Bitcoin leaves is offered to gold, sized the same way on gold.
  4. The rest sits in T-bills.
Nothing here is fitted to data: every number is a standard lookback, chosen
before testing. See etf_research.py for everything it was compared against.

    python ml/etf_model.py      # self-check
"""

import numpy as np
import pandas as pd

SMA_DAYS = (20, 50, 100, 200)
MOMENTUM_DAYS = (21, 63, 126, 252)
N_VOTES = len(SMA_DAYS) + len(MOMENTUM_DAYS)


def votes(close: pd.Series) -> pd.Series:
    """How many of the 8 trend signals say 'hold', using closes up to each day."""
    v = [close > close.rolling(n).mean() for n in SMA_DAYS]
    v += [close > close.shift(n) for n in MOMENTUM_DAYS]
    return pd.concat(v, axis=1).sum(axis=1)


def vol_scale(close: pd.Series) -> pd.Series:
    """1.0 in normal conditions, below 1 when volatility runs above its past-year norm."""
    v = np.log(close).diff().rolling(20).std()
    return (v.rolling(252, min_periods=60).median() / v).clip(upper=1.0).fillna(0.0)


def signals(btc_close: pd.Series, gold_close: pd.Series) -> pd.DataFrame:
    """Everything the model decides, per day: the votes, the scaling, the weights."""
    s = pd.DataFrame({"btc_votes": votes(btc_close), "btc_vol": vol_scale(btc_close),
                      "gold_votes": votes(gold_close), "gold_vol": vol_scale(gold_close)})
    s["w_btc"] = s["btc_votes"] / N_VOTES * s["btc_vol"]
    s["w_gold"] = (1 - s["w_btc"]) * s["gold_votes"] / N_VOTES * s["gold_vol"]
    s["w_cash"] = 1 - s["w_btc"] - s["w_gold"]
    return s


def weights(btc: pd.DataFrame, gold: pd.DataFrame) -> dict:
    s = signals(btc["close"], gold["close"])
    return {"BTC": s["w_btc"], "GLD": s["w_gold"]}


def _self_check() -> None:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=900)
    btc = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.04, 900))), days)
    gold = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))), days)
    s = signals(btc, gold)

    # A decision on day t must not change when LATER prices change.
    cut = days[600]
    s2 = signals(btc.where(days < cut, btc * 3), gold.where(days < cut, gold * 0.5))
    assert np.allclose(s[days < cut], s2[days < cut]), "the model reads the future"

    # No leverage and no shorting: weights stay in [0, 1] and never sum past 1.
    w = s[["w_btc", "w_gold", "w_cash"]]
    assert (w >= -1e-12).all().all() and (w.sum(axis=1) - 1).abs().max() < 1e-12

    # A steady rise ends fully voted in; a steady fall ends fully out.
    up = pd.Series(np.linspace(100, 300, 400), days[:400])
    down = pd.Series(np.linspace(300, 100, 400), days[:400])
    assert votes(up).iloc[-1] == N_VOTES and votes(down).iloc[-1] == 0
    print("ETF model self-check passed: no lookahead, no leverage")


if __name__ == "__main__":
    _self_check()
