"""Deterministic risk engine. AI proposes, this approves.

Pure functions, no database and no network. Everything it needs arrives as
arguments and the verdict is fully determined by them, which is what makes it
auditable: the same inputs always produce the same decision, and every rejection
names the rule that fired.

Nothing upstream -- not the model, not Jev, not autonomous mode -- can bypass
these checks, because the live loop cannot open a position except through
`approve()`.
"""

from dataclasses import dataclass, asdict, field

FEE = 0.001
SLIPPAGE = 0.0005
MIN_NOTIONAL = 10.0          # Binance's real minimum order; kept so paper sizes stay realistic


@dataclass(frozen=True)
class RiskProfile:
    risk_per_trade: float     # fraction of trading capital risked between entry and stop
    max_positions: int
    max_exposure: float       # fraction of trading capital allowed in open positions
    confidence_pct: float     # trade only this top fraction of signals by confidence
    stop_atr_mult: float
    take_profit_r: float      # take profit as a multiple of the stop distance (R)
    fallback_min_conf: float  # absolute floor used only if a model ships no gates


# risk_per_trade is small because the horizon is short. A 30-minute trade stops
# out at 0.3-0.6%, so risking 1% of capital would imply a position 150-300% of
# it -- leverage this system does not have. These values are sized so that the
# ATR-derived stop, not the no-leverage ceiling, is what determines position size
# at BTC's typical 5m ATR of 0.1-0.4%. Set them higher and every profile collapses
# to "maximum allowed size, always", which is not risk management.
#
# confidence_pct is a PERCENTILE, not an absolute threshold, because the absolute
# one is not stable. Across the 4h walk-forward the top-9% cutoff ranged from
# 0.43 to 0.95 depending on the fold. A hardcoded 0.5 would have traded a third
# of all candles in one quarter and none at all in the next. Each promoted model
# ships its own confidence distribution and the percentile is resolved against
# that; see resolve_min_confidence.
PROFILES = {
    "conservative": RiskProfile(0.0005, 1, 0.35, 0.05, 2.0, 1.5, 0.60),
    "balanced":     RiskProfile(0.0015, 2, 0.60, 0.10, 1.5, 1.5, 0.50),
    "aggressive":   RiskProfile(0.0030, 3, 1.00, 0.25, 1.2, 2.0, 0.40),
}


def resolve_min_confidence(profile_name: str, model_gates: dict | None) -> float:
    """Turn a profile's percentile into this model's absolute confidence floor.

    `model_gates` maps a percentile to the confidence value at that percentile,
    measured on the model's own validation set at promotion time, e.g.
    {0.05: 0.81, 0.10: 0.73, 0.25: 0.58}. Without it we fall back to a fixed
    floor, which is safe but blunt.
    """
    p = PROFILES[profile_name]
    if not model_gates:
        return p.fallback_min_conf
    # Normalise keys first: these arrive from JSON, where 0.10 comes back as the
    # string "0.10" and str(0.10) is "0.1", so round-tripping the key is not safe.
    gates = {float(k): float(v) for k, v in model_gates.items()}
    # Nearest available percentile at or tighter than the one requested, so a
    # missing key errs toward trading less rather than more.
    eligible = [k for k in gates if k <= p.confidence_pct]
    return gates[max(eligible)] if eligible else gates[min(gates)]


@dataclass
class PortfolioState:
    cash: float
    equity: float
    open_positions: int
    open_notional: float
    daily_pnl: float           # realised, since the start of the UTC day
    day_start_equity: float


@dataclass
class Proposal:
    side: str                  # 'LONG' or 'SHORT'
    confidence: float
    price: float
    atr_pct: float             # ATR as a fraction of price


@dataclass
class Verdict:
    approved: bool
    rule: str                  # which rule decided; 'ok' when approved
    detail: str = ""
    qty: float = 0.0
    notional: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def approve(proposal: Proposal, state: PortfolioState, profile_name: str,
            virtual_capital: float, trading_allocation: float,
            daily_profit_target: float, max_daily_loss: float,
            model_gates: dict | None = None) -> Verdict:
    """Return the sized, bounded trade, or the reason there isn't one.

    Checks run cheapest-and-hardest first so that the reported rule is the most
    fundamental reason for rejection, not whichever test happened to run last.
    """
    if profile_name not in PROFILES:
        return Verdict(False, "bad_profile", f"unknown risk profile {profile_name!r}")
    p = PROFILES[profile_name]

    if proposal.side not in ("LONG", "SHORT"):
        return Verdict(False, "bad_side", f"side {proposal.side!r}")
    if not (proposal.price > 0):
        return Verdict(False, "bad_price", f"price {proposal.price}")

    trading_capital = virtual_capital * trading_allocation

    # --- hard daily stops, checked before anything that could size a trade ---
    loss_limit = -abs(max_daily_loss) * state.day_start_equity
    if state.daily_pnl <= loss_limit:
        return Verdict(False, "daily_loss_limit",
                       f"daily pnl {state.daily_pnl:.2f} <= limit {loss_limit:.2f}")

    target = daily_profit_target * state.day_start_equity
    if state.daily_pnl >= target:
        return Verdict(False, "daily_target_reached",
                       f"daily pnl {state.daily_pnl:.2f} >= target {target:.2f}")

    # --- signal quality ---
    min_conf = resolve_min_confidence(profile_name, model_gates)
    if proposal.confidence < min_conf:
        return Verdict(False, "low_confidence",
                       f"confidence {proposal.confidence:.3f} < {min_conf:.3f} "
                       f"(top {p.confidence_pct:.0%} of signals)")

    # --- concurrency and exposure ---
    if state.open_positions >= p.max_positions:
        return Verdict(False, "max_positions",
                       f"{state.open_positions} open, limit {p.max_positions}")

    exposure_room = p.max_exposure * trading_capital - state.open_notional
    if exposure_room <= 0:
        return Verdict(False, "max_exposure",
                       f"open notional {state.open_notional:.2f} at limit "
                       f"{p.max_exposure * trading_capital:.2f}")

    # --- sizing: risk a fixed fraction between entry and stop ---
    # An ATR-derived stop means position size shrinks automatically when the
    # market gets violent, which is the whole point of sizing off volatility
    # rather than off a fixed percentage of capital.
    stop_frac = max(proposal.atr_pct * p.stop_atr_mult, 0.002)
    risk_amount = p.risk_per_trade * trading_capital
    notional = risk_amount / stop_frac

    # Whichever ceiling is lowest wins. At tight stops the exposure cap binds
    # first: risking 1% with a 0.6% stop implies a position ~167% of trading
    # capital, which is leverage. Capping here means risk_per_trade is a ceiling
    # on risk, not a promise to reach it.
    notional = min(notional, exposure_room, state.cash)
    if notional < MIN_NOTIONAL:
        return Verdict(False, "below_min_notional",
                       f"sized {notional:.2f} < minimum {MIN_NOTIONAL:.2f}")

    qty = notional / proposal.price
    direction = 1 if proposal.side == "LONG" else -1
    stop_loss = proposal.price * (1 - direction * stop_frac)
    take_profit = proposal.price * (1 + direction * stop_frac * p.take_profit_r)

    return Verdict(True, "ok", f"{profile_name} risk {p.risk_per_trade:.2%} "
                               f"stop {stop_frac:.2%}",
                   qty=qty, notional=notional, stop_loss=stop_loss,
                   take_profit=take_profit, risk_amount=notional * stop_frac)


def _self_check() -> None:
    base_state = PortfolioState(cash=5000, equity=10000, open_positions=0,
                                open_notional=0, daily_pnl=0, day_start_equity=10000)
    good = Proposal(side="LONG", confidence=0.60, price=100000, atr_pct=0.004)
    kw = dict(virtual_capital=10000, trading_allocation=0.5,
              daily_profit_target=0.02, max_daily_loss=0.02)

    v = approve(good, base_state, "balanced", **kw)
    assert v.approved and v.rule == "ok", v
    assert v.stop_loss < good.price < v.take_profit, "long stop/target inverted"
    # risk_per_trade is a ceiling, not a target: capping may land below it, but
    # never above. This is the invariant that actually protects the account.
    assert v.risk_amount <= PROFILES['balanced'].risk_per_trade * 5000 + 1e-9, v.risk_amount
    # Never levered -- position size cannot exceed the exposure ceiling.
    assert v.notional <= PROFILES['balanced'].max_exposure * 5000 + 1e-9, v.notional

    # With a stop wide enough that no ceiling binds, risk hits the profile target.
    wide = approve(Proposal("LONG", 0.60, 100000, 0.05), base_state, "balanced", **kw)
    assert abs(wide.risk_amount - PROFILES['balanced'].risk_per_trade * 5000) < 1e-6, wide.risk_amount

    # Shorts mirror.
    s = approve(Proposal("SHORT", 0.60, 100000, 0.004), base_state, "balanced", **kw)
    assert s.approved and s.take_profit < 100000 < s.stop_loss, s

    # Daily loss limit is absolute: a perfect signal must still be refused.
    blown = PortfolioState(5000, 9800, 0, 0, daily_pnl=-200, day_start_equity=10000)
    assert approve(Proposal("LONG", 0.99, 100000, 0.004), blown, "balanced",
                   **kw).rule == "daily_loss_limit"

    # So is the profit target.
    done = PortfolioState(5000, 10200, 0, 0, daily_pnl=200, day_start_equity=10000)
    assert approve(good, done, "balanced", **kw).rule == "daily_target_reached"

    # Confidence floor, and it differs by profile. Without model gates the
    # fallback floors apply (0.60 / 0.50 / 0.40).
    weak = Proposal("LONG", 0.45, 100000, 0.004)
    assert approve(weak, base_state, "conservative", **kw).rule == "low_confidence"
    assert approve(weak, base_state, "aggressive", **kw).approved

    # With a model's own gates, the same percentile resolves to that model's
    # scale. A model whose top-10% cutoff is 0.73 must reject 0.70 for balanced
    # while a model whose cutoff is 0.45 accepts it -- this is the whole reason
    # the threshold travels with the model.
    tight = {0.05: 0.81, 0.10: 0.73, 0.25: 0.58}
    loose = {0.05: 0.52, 0.10: 0.45, 0.25: 0.38}
    mid = Proposal("LONG", 0.70, 100000, 0.004)
    assert approve(mid, base_state, "balanced", model_gates=tight, **kw).rule == "low_confidence"
    assert approve(mid, base_state, "balanced", model_gates=loose, **kw).approved

    # Conservative is stricter than aggressive on the same model.
    assert (resolve_min_confidence("conservative", tight)
            > resolve_min_confidence("aggressive", tight))
    # A percentile with no exact match must round toward trading less.
    assert resolve_min_confidence("balanced", {0.05: 0.81, 0.25: 0.58}) == 0.81
    # String keys survive a JSON round trip.
    assert resolve_min_confidence("balanced", {"0.10": 0.73}) == 0.73

    # Position count and exposure ceilings.
    assert approve(good, PortfolioState(5000, 10000, 2, 1000, 0, 10000),
                   "balanced", **kw).rule == "max_positions"
    assert approve(good, PortfolioState(5000, 10000, 1, 3000, 0, 10000),
                   "balanced", **kw).rule == "max_exposure"
    # Volatility sizing must actually bind at realistic ATR, not be masked by the
    # exposure ceiling -- otherwise every profile is just "maximum size, always".
    sizes = [approve(Proposal("LONG", 0.6, 100000, a), base_state, "balanced", **kw).notional
             for a in (0.001, 0.002, 0.004)]
    assert sizes[0] > sizes[1] > sizes[2], sizes

    # Higher volatility must shrink the position, never grow it.
    calm = approve(Proposal("LONG", 0.60, 100000, 0.002), base_state, "balanced", **kw)
    wild = approve(Proposal("LONG", 0.60, 100000, 0.020), base_state, "balanced", **kw)
    assert wild.notional < calm.notional, (calm.notional, wild.notional)

    # Sizing can never exceed available cash or the exposure ceiling.
    poor = PortfolioState(cash=50, equity=10000, open_positions=0, open_notional=0,
                          daily_pnl=0, day_start_equity=10000)
    v = approve(good, poor, "balanced", **kw)
    assert not v.approved or v.notional <= 50.0 + 1e-9, v

    # A stop can never be placed on the wrong side of entry, at any ATR.
    for atr in (0.0, 0.001, 0.05, 0.5):
        r = approve(Proposal("LONG", 0.6, 100000, atr), base_state, "balanced", **kw)
        if r.approved:
            assert r.stop_loss < 100000 < r.take_profit, (atr, r)

    # Garbage in, refusal out -- never an exception, never an approval.
    assert not approve(Proposal("SIDEWAYS", 0.9, 100, 0.01), base_state, "balanced", **kw).approved
    assert not approve(Proposal("LONG", 0.9, 0, 0.01), base_state, "balanced", **kw).approved
    assert not approve(good, base_state, "reckless", **kw).approved

    print("risk engine self-check passed")


if __name__ == "__main__":
    _self_check()
