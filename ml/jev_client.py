"""Jev decision layer: picks a strategy from the model's prediction and the regime.

Jev is a decision model, not a text LLM -- it returns typed choices with
calibrated probabilities. That makes it a good fit here: we want a strategy
label and a number, not prose.

The fallback table is not optional. Three reasons it exists:

  * Jev rate limits aggressively. Measured: two requests back to back returned
    429, and recovery took ~60s. A trading loop cannot stall on that.
  * The community free tier has no published quota, so it can change without notice.
  * An unguarded network call in the decision path means a timeout becomes a
    stuck position.

Every decision records which path produced it, so a week of silent fallback is
queryable rather than invisible.
"""

import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://www.jevai.org/api/v1/decisions"
MODEL = "jev-1"
TIMEOUT = 8.0

STRATEGIES = {
    "trend_following": "A confirmed directional trend is underway; enter with it and hold.",
    "momentum": "Price and volume are accelerating sharply in one direction.",
    "breakout": "Price is leaving a consolidation range on expanding volume.",
    "mean_reversion": "Price is stretched far from its mean in a rangebound market.",
    "vwap": "Price is dislocated from VWAP and likely to revert toward it.",
    "no_trade": "The signal is weak, the regime is unclear, or conditions are unfavourable.",
}

# Strategies that express a directional view. 'no_trade' obviously does not.
DIRECTIONAL = set(STRATEGIES) - {"no_trade"}


def build_state(prediction: str, confidence: float, regime, price: float,
                atr_pct: float, vwap_dist: float, bb_pctb: float,
                open_positions: int) -> str:
    """One compact paragraph of market state. Deliberately carries no user data.

    Per-user capital and limits are the risk engine's business, not Jev's, and
    sending them would put account details into a third-party request for no gain.
    """
    return (
        f"BTC/USDT 5-minute chart, 4-hour forward view. "
        f"Model prediction: {prediction} with confidence {confidence:.2f}. "
        f"Market regime: {regime.name}, {regime.volatility} volatility. "
        f"ATR is {atr_pct:.2%} of price. "
        f"Price is {vwap_dist:+.2%} from the 24h VWAP and sits at "
        f"{bb_pctb:.2f} across its Bollinger band (0 = lower, 1 = upper). "
        f"There {'is' if open_positions == 1 else 'are'} {open_positions} "
        f"position(s) already open."
    )


def fallback_strategy(prediction: str, confidence: float, regime) -> tuple[str, float]:
    """Deterministic strategy choice. Runs when Jev is unavailable.

    Same inputs, same decision, every time. Confidence returned here is the
    model's own, not an independent estimate -- we do not invent a number that
    looks like Jev's calibrated one.
    """
    if prediction == "NEUTRAL":
        return "no_trade", confidence

    agrees = (prediction == "UP" and regime.direction >= 0) or \
             (prediction == "DOWN" and regime.direction <= 0)

    # Fighting a confirmed trend is the one case worth refusing outright.
    if regime.trending and not agrees:
        return "no_trade", confidence

    if regime.trending:
        return "trend_following", confidence
    if regime.volatility == "high":
        return "breakout", confidence
    if regime.volatility == "low":
        return "mean_reversion", confidence
    return "vwap", confidence


def decide(prediction: str, confidence: float, regime, price: float,
           atr_pct: float, vwap_dist: float, bb_pctb: float,
           open_positions: int = 0, api_key: str | None = None) -> dict:
    """Choose a strategy. Never raises, never blocks longer than TIMEOUT.

    Returns {strategy, action, jev_confidence, decided_by, raw}.
    """
    key = api_key or os.environ.get("JEV_API_KEY")
    result = {"strategy": None, "action": "HOLD", "jev_confidence": None,
              "decided_by": "fallback", "raw": None, "error": None}

    if key:
        state = build_state(prediction, confidence, regime, price, atr_pct,
                            vwap_dist, bb_pctb, open_positions)
        body = json.dumps({
            "model": MODEL,
            "state": state,
            "questions": {
                "strategy": {
                    "type": "choice",
                    "instructions": "Which trading strategy best fits this market state?",
                    "criteria": STRATEGIES,
                },
                "should_trade": {
                    "type": "noul",
                    "instructions": "Should a new position be opened right now?",
                },
            },
        }).encode()

        # An explicit User-Agent is required: the default "Python-urllib/3.x" is
        # rejected with 403, where the identical request from curl is accepted.
        req = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "User-Agent": "btc-paper-trader/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                payload = json.loads(r.read())
            if payload.get("code") == 0:
                answers = payload.get("data", {}).get("answers", {})
                choice = answers.get("strategy", {})
                strategy = choice.get("choice")
                if strategy in STRATEGIES:
                    should = answers.get("should_trade", {})
                    result.update(strategy=strategy, decided_by="jev", raw=payload,
                                  jev_confidence=choice.get("confidence"))
                    # Jev's own yes/no gate, when it gives one, can veto but never
                    # force: the risk engine still has the final say downstream.
                    veto = isinstance(should, dict) and should.get("probability", 1.0) < 0.5
                    if veto:
                        result["strategy"] = "no_trade"
                else:
                    result["error"] = f"unknown strategy {strategy!r}"
            else:
                result["error"] = payload.get("message", "non-zero code")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                json.JSONDecodeError, OSError) as e:
            # 429 lands here and is expected, not exceptional.
            result["error"] = f"{type(e).__name__}: {e}"
    else:
        result["error"] = "no JEV_API_KEY set"

    if result["strategy"] is None:
        strategy, _ = fallback_strategy(prediction, confidence, regime)
        result["strategy"] = strategy

    result["action"] = ("HOLD" if result["strategy"] == "no_trade"
                        else "BUY" if prediction == "UP"
                        else "SELL" if prediction == "DOWN" else "HOLD")
    return result


def _self_check() -> None:
    from regime import classify
    up_trend = classify(0.31, 0.15, 0.01, 0.50, 0.02)
    down_trend = classify(0.31, -0.15, -0.01, 0.50, 0.02)
    quiet = classify(0.10, 0.0, 0.0, 0.05, 0.01)
    wild = classify(0.10, 0.0, 0.0, 0.95, 0.05)
    normal_range = classify(0.10, 0.0, 0.0, 0.50, 0.02)

    # Fallback is total: every prediction x regime pair yields a known strategy.
    for pred in ("UP", "DOWN", "NEUTRAL"):
        for reg in (up_trend, down_trend, quiet, wild, normal_range):
            s, _ = fallback_strategy(pred, 0.7, reg)
            assert s in STRATEGIES, (pred, reg.name, s)

    assert fallback_strategy("UP", 0.7, up_trend)[0] == "trend_following"
    assert fallback_strategy("NEUTRAL", 0.9, up_trend)[0] == "no_trade"
    # Refuses to fight a confirmed trend in either direction.
    assert fallback_strategy("DOWN", 0.9, up_trend)[0] == "no_trade"
    assert fallback_strategy("UP", 0.9, down_trend)[0] == "no_trade"
    assert fallback_strategy("UP", 0.7, wild)[0] == "breakout"
    assert fallback_strategy("UP", 0.7, quiet)[0] == "mean_reversion"
    assert fallback_strategy("UP", 0.7, normal_range)[0] == "vwap"

    # With no key it must fall back cleanly rather than raise or hang.
    saved = os.environ.pop("JEV_API_KEY", None)
    try:
        r = decide("UP", 0.7, up_trend, 100000, 0.004, 0.001, 0.6)
        assert r["decided_by"] == "fallback" and r["strategy"] == "trend_following"
        assert r["action"] == "BUY" and r["error"]
        assert decide("DOWN", 0.7, down_trend, 100000, 0.004, 0.0, 0.4)["action"] == "SELL"
        assert decide("NEUTRAL", 0.7, up_trend, 100000, 0.004, 0.0, 0.5)["action"] == "HOLD"
        # A bad key must also degrade rather than raise.
        bad = decide("UP", 0.7, up_trend, 100000, 0.004, 0.0, 0.5, api_key="jev_invalid")
        assert bad["decided_by"] == "fallback" and bad["strategy"] in STRATEGIES
    finally:
        if saved:
            os.environ["JEV_API_KEY"] = saved

    # State string carries market context and no account details.
    s = build_state("UP", 0.78, up_trend, 100000, 0.004, 0.001, 0.6, 0)
    assert "uptrend" in s and "0.78" in s and len(s) < 2000
    for leaked in ("capital", "balance", "user", "email", "$"):
        assert leaked not in s.lower(), f"state leaks {leaked}"

    print("jev client self-check passed")


if __name__ == "__main__":
    _self_check()
