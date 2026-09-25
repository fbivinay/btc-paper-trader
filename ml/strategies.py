"""Low-turnover long/flat signals, each a pure function of past closes.

Every signal at day t uses only data up to and including the close of t; the
simulator executes it at the open of t+1. Nothing here looks forward.

Why these families: under a tax that hits every winning sale and never offsets
a losing one, each round trip is a tax event and each losing round trip is pure
waste. The only sensible targets are strategies that trade a few times a year
and hold for the large moves -- trend and momentum filters, with hysteresis so
that price chopping around a line does not generate a stream of small losses.
"""

import numpy as np
import pandas as pd


def _hysteresis(above_enter: np.ndarray, below_exit: np.ndarray) -> np.ndarray:
    """Go long when the entry condition fires, flat when the exit fires, else hold."""
    out = np.zeros(len(above_enter)); pos = 0
    for i in range(len(out)):
        if pos == 0 and above_enter[i]:
            pos = 1
        elif pos == 1 and below_exit[i]:
            pos = 0
        out[i] = pos
    return out


def sma_band(close: pd.Series, n: int, band: float) -> np.ndarray:
    """Long above SMA(n)*(1+band), flat below SMA(n)*(1-band)."""
    sma = close.rolling(n).mean()
    enter = (close > sma * (1 + band)).fillna(False).to_numpy()
    exit_ = (close < sma * (1 - band)).fillna(False).to_numpy()
    return _hysteresis(enter, exit_)


def momentum(close: pd.Series, lookback: int, band: float) -> np.ndarray:
    """Long when the lookback return is above +band, flat when below -band."""
    r = close.pct_change(lookback)
    return _hysteresis((r > band).fillna(False).to_numpy(),
                       (r < -band).fillna(False).to_numpy())


def dual_sma(close: pd.Series, fast: int, slow: int) -> np.ndarray:
    """Long while SMA(fast) is above SMA(slow)."""
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    return (f > s).fillna(False).to_numpy().astype(float)


def donchian(close: pd.Series, entry: int, exit_: int) -> np.ndarray:
    """Turtle-style: enter on a new entry-day high, exit on a new exit-day low."""
    hi = close.rolling(entry).max().shift(1)
    lo = close.rolling(exit_).min().shift(1)
    return _hysteresis((close > hi).fillna(False).to_numpy(),
                       (close < lo).fillna(False).to_numpy())


def grid() -> list[tuple[str, dict, callable]]:
    """The candidate set. Deliberately modest: every extra candidate is another
    chance to find something that only looks good by luck."""
    g = []
    for n in (50, 100, 150, 200):
        for b in (0.0, 0.03, 0.06, 0.10):
            g.append((f"sma{n}_b{b:g}", dict(n=n, band=b), sma_band))
    for lb in (30, 60, 90, 180):
        for b in (0.0, 0.05, 0.10):
            g.append((f"mom{lb}_b{b:g}", dict(lookback=lb, band=b), momentum))
    for f, s in ((20, 100), (50, 150), (50, 200), (100, 200)):
        g.append((f"dual{f}_{s}", dict(fast=f, slow=s), dual_sma))
    for e, x in ((55, 20), (100, 50), (150, 75)):
        g.append((f"donch{e}_{x}", dict(entry=e, exit_=x), donchian))
    return g


def _self_check() -> None:
    # A signal must not change when FUTURE prices change.
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 600))))
    for name, kw, fn in grid():
        a = fn(close, **kw)
        future = close.copy(); future.iloc[400:] *= 3.0
        b = fn(future, **kw)
        assert (a[:400] == b[:400]).all(), f"{name} reads the future"
    # A steady uptrend must end long; a steady downtrend must end flat.
    up = pd.Series(np.linspace(100, 400, 500))
    dn = pd.Series(np.linspace(400, 100, 500))
    for name, kw, fn in grid():
        assert fn(up, **kw)[-1] == 1, f"{name} not long in an uptrend"
        assert fn(dn, **kw)[-1] == 0, f"{name} not flat in a downtrend"
    print(f"strategies self-check passed ({len(grid())} candidates, no lookahead)")


if __name__ == "__main__":
    _self_check()
