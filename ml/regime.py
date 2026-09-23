"""Market regime from the same features the model sees.

Deterministic and cheap. Its job is not to be clever -- it is to give Jev (and
the fallback table) a compact description of conditions, and to give the
dashboard something a human can read next to the prediction.

Thresholds are conventional TA values (ADX 25 for trend strength) rather than
fitted, because fitting them on the same data the model trains on would just be
another way to overfit the same five years.
"""

from dataclasses import dataclass

ADX_TREND = 0.25          # adx feature is scaled to 0-1, so this is ADX 25
ADX_STRONG = 0.40
VOL_HIGH = 0.80           # percentile of 7-day realised vol
VOL_LOW = 0.20


@dataclass(frozen=True)
class Regime:
    name: str
    trending: bool
    direction: int        # +1 up, -1 down, 0 none
    volatility: str       # 'low' | 'normal' | 'high'

    def describe(self) -> str:
        return f"{self.name} ({self.volatility} volatility)"


def classify(adx: float, di_diff: float, ema_spread: float,
             vol_pctile: float, bb_width: float) -> Regime:
    """Label current conditions from already-computed features.

    adx, di_diff, ema_spread, vol_pctile and bb_width come straight from
    features.py, so the regime can never disagree with what the model was fed.
    """
    vol = "high" if vol_pctile >= VOL_HIGH else "low" if vol_pctile <= VOL_LOW else "normal"

    trending = adx >= ADX_TREND
    if trending:
        # di_diff carries the trend's sign; ema_spread breaks ties when DI is flat.
        direction = 1 if di_diff > 0 else -1 if di_diff < 0 else (1 if ema_spread > 0 else -1)
        strong = adx >= ADX_STRONG
        name = ("strong_" if strong else "") + ("uptrend" if direction > 0 else "downtrend")
        return Regime(name, True, direction, vol)

    # Not trending. A wide Bollinger band without trend is chop, not range.
    name = "volatile_range" if vol == "high" else "quiet_range" if vol == "low" else "range"
    return Regime(name, False, 0, vol)


def from_row(row) -> Regime:
    """Classify from a features row (pandas Series or dict)."""
    get = row.get if hasattr(row, "get") else (lambda k, d=0.0: getattr(row, k, d))
    return classify(
        adx=float(get("adx", 0.0)),
        di_diff=float(get("di_diff", 0.0)),
        ema_spread=float(get("ema_spread", 0.0)),
        vol_pctile=float(get("vol_pctile", 0.5)),
        bb_width=float(get("bb_width", 0.0)),
    )


def _self_check() -> None:
    up = classify(adx=0.31, di_diff=0.15, ema_spread=0.01, vol_pctile=0.45, bb_width=0.02)
    assert up.name == "uptrend" and up.trending and up.direction == 1 and up.volatility == "normal"

    strong = classify(0.55, 0.20, 0.02, 0.50, 0.03)
    assert strong.name == "strong_uptrend"

    down = classify(0.31, -0.15, -0.01, 0.50, 0.02)
    assert down.name == "downtrend" and down.direction == -1

    # No trend strength means range regardless of DI.
    flat = classify(0.10, 0.30, 0.05, 0.50, 0.02)
    assert not flat.trending and flat.direction == 0 and flat.name == "range"

    assert classify(0.10, 0.0, 0.0, 0.95, 0.05).name == "volatile_range"
    assert classify(0.10, 0.0, 0.0, 0.05, 0.01).name == "quiet_range"

    # Flat DI falls back to the EMA spread for direction, never to 0 while trending.
    tie = classify(0.31, 0.0, -0.01, 0.50, 0.02)
    assert tie.trending and tie.direction == -1

    # A trending regime always carries a non-zero direction.
    for di in (-0.5, -0.01, 0.0, 0.01, 0.5):
        r = classify(0.30, di, 0.01, 0.5, 0.02)
        assert r.direction != 0, (di, r)

    assert from_row({"adx": 0.31, "di_diff": 0.15, "ema_spread": 0.01,
                     "vol_pctile": 0.45, "bb_width": 0.02}).name == "uptrend"

    print("regime self-check passed")


if __name__ == "__main__":
    _self_check()
