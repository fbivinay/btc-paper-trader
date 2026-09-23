"""Trade simulation and performance metrics.

Shared by train.py (model selection) and backtest.py (reporting) so that the
numbers used to pick a model are produced by exactly the same code that reports
them. Deliberately has no torch dependency.

Accuracy is not in here on purpose. 68% of labels are NEUTRAL, so a model that
never trades scores 68% accuracy and zero return. Selection uses these metrics.
"""

import numpy as np
import pandas as pd

from config import COST, HORIZON, BARS_PER_YEAR

DOWN, NEUTRAL, UP = 0, 1, 2


def simulate(pred, conf, fwd_return, min_conf=0.0, cost=COST, horizon=HORIZON):
    """Walk the bars and take non-overlapping trades.

    One position at a time, held for the full horizon. Overlapping entries would
    inflate the trade count and let the same move be counted several times, which
    is the most common way a crypto backtest lies about its Sharpe.

    Sequential by necessity -- whether bar i is tradeable depends on what was
    entered before it, so this cannot be vectorised.
    """
    pred, conf, fwd_return = np.asarray(pred), np.asarray(conf), np.asarray(fwd_return)
    idx, pnl, side = [], [], []
    next_free = 0

    for i in range(len(pred)):
        if i < next_free or pred[i] == NEUTRAL or conf[i] < min_conf:
            continue
        if not np.isfinite(fwd_return[i]):
            continue
        direction = 1 if pred[i] == UP else -1
        idx.append(i)
        side.append(direction)
        pnl.append(direction * fwd_return[i] - cost)
        next_free = i + horizon + 1

    return pd.DataFrame({"bar": idx, "side": side, "pnl": pnl})


def trade_metrics(trades: pd.DataFrame, n_bars: int) -> dict:
    """Performance of a trade sequence. n_bars sets the annualisation horizon."""
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "total_return": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0, "exposure": 0.0}

    pnl = trades["pnl"].to_numpy()
    # Equity starts at 1.0 before the first trade. Without that leading 1.0 the
    # running peak starts at the post-first-trade value, so an opening loss
    # reports zero drawdown.
    equity = np.concatenate([[1.0], np.cumprod(1 + pnl)])
    peak = np.maximum.accumulate(equity)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    years = n_bars / BARS_PER_YEAR
    trades_per_year = n / years if years > 0 else 0.0
    sd = pnl.std(ddof=1) if n > 1 else 0.0

    return {
        "n_trades": n,
        "win_rate": float((pnl > 0).mean()),
        "total_return": float(equity[-1] - 1),
        "sharpe": float(pnl.mean() / sd * np.sqrt(trades_per_year)) if sd > 0 else 0.0,
        "max_dd": float((equity / peak - 1).min()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        "avg_pnl": float(pnl.mean()),
        "exposure": float(n * (HORIZON + 1) / n_bars),
    }


def class_precision(pred, true) -> dict:
    """Precision on the two classes that cost money. Recall is secondary here:
    a missed trade costs nothing, a wrong trade costs the round trip."""
    pred, true = np.asarray(pred), np.asarray(true)
    out = {}
    for name, cls in (("up", UP), ("down", DOWN)):
        taken = pred == cls
        out[f"prec_{name}"] = float((true[taken] == cls).mean()) if taken.any() else 0.0
        out[f"n_{name}"] = int(taken.sum())
    return out


def evaluate(pred, conf, fwd_return, true_label, min_conf=0.0) -> dict:
    trades = simulate(pred, conf, fwd_return, min_conf=min_conf)
    return {**trade_metrics(trades, len(pred)), **class_precision(pred, true_label),
            "min_conf": min_conf}


def _self_check() -> None:
    n = 1000
    fwd = np.zeros(n)

    # A perfect predictor on moves that clear cost must be profitable. Signals are
    # spaced wider than the horizon so every one is takeable regardless of how
    # HORIZON is configured -- otherwise this test's expected count silently
    # depends on the config it is meant to be independent of.
    spacing = HORIZON + 10
    fwd[::spacing] = 0.01
    pred = np.where(fwd > 0, UP, NEUTRAL)
    conf = np.ones(n)
    t = simulate(pred, conf, fwd)
    assert len(t) == len(range(0, n, spacing)), f"got {len(t)} trades"
    assert np.allclose(t["pnl"], 0.01 - COST), "pnl must be net of cost"

    # Non-overlap: signals on consecutive bars must collapse to one trade per horizon.
    dense = np.full(n, UP)
    assert len(simulate(dense, conf, np.full(n, 0.01))) == n // (HORIZON + 1) + 1

    # A winning move smaller than the round trip must still lose money. Derived
    # from COST, not hardcoded: a fixed 0.2% was "below cost" on spot fees and
    # silently became "above cost" the moment the execution model changed.
    small = simulate(np.array([UP]), np.array([1.0]), np.array([COST * 0.5]))
    assert small["pnl"].iloc[0] < 0, "sub-cost win must be a net loss"
    big = simulate(np.array([UP]), np.array([1.0]), np.array([COST * 2]))
    assert big["pnl"].iloc[0] > 0, "move of twice the cost must be a net win"

    # Shorts profit from down moves.
    assert simulate(np.array([DOWN]), np.array([1.0]), np.array([-0.01]))["pnl"].iloc[0] > 0

    # Max drawdown is negative and bounded.
    m = trade_metrics(pd.DataFrame({"bar": [0, 7], "side": [1, 1], "pnl": [-0.1, 0.05]}), 1000)
    assert np.isclose(m["max_dd"], -0.1) and m["win_rate"] == 0.5, m
    # A first-trade loss must register as drawdown, not be hidden by the peak.
    assert np.isclose(trade_metrics(pd.DataFrame({"bar": [0], "side": [1], "pnl": [-0.2]}), 100)["max_dd"], -0.2)

    # Precision counts only the bars where we acted.
    p = class_precision(np.array([UP, UP, DOWN]), np.array([UP, NEUTRAL, DOWN]))
    assert p["prec_up"] == 0.5 and p["prec_down"] == 1.0 and p["n_up"] == 2

    # No signal must be flat, never a crash.
    assert trade_metrics(simulate(np.full(n, NEUTRAL), conf, fwd), n)["n_trades"] == 0

    print("metrics self-check passed")


if __name__ == "__main__":
    _self_check()
