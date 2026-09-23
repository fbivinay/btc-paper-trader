"""Record positioning data that Binance does not keep history for.

Open interest and the long/short ratio are retained for roughly 30 days, and
only about two days at 5-minute granularity. They cannot be back-filled, so they
cannot be trained on today -- but every candle not recorded now is gone for
good. This captures them so that in a few months there is a history worth
training on.

Funding rate is captured alongside them purely so the dashboard reads one table
instead of calling three endpoints.

Called by live_loop each run. Failures here never stop trading: positioning data
is a nice-to-have and the futures API is a separate service from the one that
provides candles.
"""

import json
import urllib.request

HOSTS = ["https://fapi.binance.com", "https://fapi-gcp.binance.com"]
SYMBOL = "BTCUSDT"


def _get(path: str, timeout: float = 10.0):
    last = None
    for host in HOSTS:
        try:
            req = urllib.request.Request(host + path,
                                         headers={"User-Agent": "btc-paper-trader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
    raise last if last else RuntimeError("no host")


def snapshot() -> dict:
    """Current positioning. Missing fields come back as None rather than raising."""
    out = {"funding_rate": None, "open_interest": None,
           "long_short_ratio": None, "top_trader_ratio": None}

    try:
        prem = _get(f"/fapi/v1/premiumIndex?symbol={SYMBOL}")
        out["funding_rate"] = float(prem["lastFundingRate"])
    except Exception:
        pass

    try:
        oi = _get(f"/fapi/v1/openInterest?symbol={SYMBOL}")
        out["open_interest"] = float(oi["openInterest"])
    except Exception:
        pass

    try:
        ls = _get(f"/futures/data/globalLongShortAccountRatio?symbol={SYMBOL}&period=5m&limit=1")
        if ls:
            out["long_short_ratio"] = float(ls[-1]["longShortRatio"])
    except Exception:
        pass

    try:
        tt = _get(f"/futures/data/topLongShortPositionRatio?symbol={SYMBOL}&period=5m&limit=1")
        if tt:
            out["top_trader_ratio"] = float(tt[-1]["longShortRatio"])
    except Exception:
        pass

    return out


def record(db, candle_time) -> dict | None:
    """Store one snapshot. Returns it, or None if nothing could be fetched."""
    try:
        snap = snapshot()
    except Exception:
        return None
    if all(v is None for v in snap.values()):
        return None
    try:
        db.insert("market_context",
                  {"candle_time": candle_time.isoformat(), **snap},
                  upsert_on="candle_time")
    except Exception:
        return None
    return snap


if __name__ == "__main__":
    s = snapshot()
    for k, v in s.items():
        print(f"{k:<18} {v}")
