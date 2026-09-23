"""Simulated execution. Opens, marks and closes positions; no real orders.

Pure functions over explicit state, same as the risk engine, so the whole
decision path can be replayed from stored rows.

Three modelling choices decide whether this lies to you, and all three are made
pessimistically on purpose:

1. Entry is the NEXT candle's open, never the close that triggered the decision.
   At the close of t you only know the close of t; filling there is hindsight.
2. When a candle's range covers both the stop and the target, the stop wins. A
   5m candle does not record the order its extremes happened in, and assuming
   the good one came first is the single most common way a backtest invents
   profit it would never have earned.
3. A candle that opens past the stop fills at the open, not at the stop price.
   Stops are not guaranteed fills, and gaps are exactly when that matters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import COST, FEE, HORIZON, SLIPPAGE

BAR = timedelta(minutes=5)

STOP_LOSS, TAKE_PROFIT, HORIZON_EXIT, MANUAL = "STOP_LOSS", "TAKE_PROFIT", "HORIZON", "MANUAL"


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class Position:
    side: str                 # 'LONG' or 'SHORT'
    qty: float
    entry_price: float        # after slippage
    entry_time: datetime
    stop_loss: float
    take_profit: float
    expires_at: datetime
    fees: float = 0.0
    strategy: str | None = None
    decision_id: int | None = None

    status: str = "OPEN"
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None
    pnl: float | None = None
    pnl_pct: float | None = None

    @property
    def direction(self) -> int:
        return 1 if self.side == "LONG" else -1

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def unrealised(self, price: float) -> float:
        """Mark to market, net of the exit fee not yet paid."""
        gross = (price - self.entry_price) * self.qty * self.direction
        return gross - FEE * self.qty * price


def fill_price(price: float, side: str, entering: bool) -> float:
    """Slippage always works against us, on both legs."""
    direction = 1 if side == "LONG" else -1
    adverse = direction if entering else -direction
    return price * (1 + adverse * SLIPPAGE)


def open_position(side: str, qty: float, entry_candle: Candle,
                  stop_loss: float, take_profit: float,
                  strategy: str | None = None, decision_id: int | None = None) -> Position:
    """Open at the given candle's OPEN. Caller passes the candle AFTER the signal."""
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"side {side!r}")
    if qty <= 0:
        raise ValueError(f"qty {qty}")

    entry = fill_price(entry_candle.open, side, entering=True)
    return Position(
        side=side, qty=qty, entry_price=entry, entry_time=entry_candle.time,
        stop_loss=stop_loss, take_profit=take_profit,
        expires_at=entry_candle.time + HORIZON * BAR,
        fees=FEE * qty * entry, strategy=strategy, decision_id=decision_id,
    )


def check_exit(pos: Position, candle: Candle) -> tuple[float, str] | None:
    """Would this candle close the position, and at what price?

    Returns (raw_price, reason) before slippage, or None to stay open.
    """
    if pos.status != "OPEN":
        return None

    if pos.side == "LONG":
        # Gap through the stop fills at the open, which is worse than the stop.
        if candle.open <= pos.stop_loss:
            return candle.open, STOP_LOSS
        if candle.low <= pos.stop_loss:
            return pos.stop_loss, STOP_LOSS
        if candle.open >= pos.take_profit:
            return candle.open, TAKE_PROFIT
        if candle.high >= pos.take_profit:
            return pos.take_profit, TAKE_PROFIT
    else:
        if candle.open >= pos.stop_loss:
            return candle.open, STOP_LOSS
        if candle.high >= pos.stop_loss:
            return pos.stop_loss, STOP_LOSS
        if candle.open <= pos.take_profit:
            return candle.open, TAKE_PROFIT
        if candle.low <= pos.take_profit:
            return pos.take_profit, TAKE_PROFIT

    if candle.time >= pos.expires_at:
        return candle.close, HORIZON_EXIT
    return None


def close_position(pos: Position, raw_price: float, when: datetime, reason: str) -> Position:
    """Close and book the PnL. Mutates and returns the position."""
    if pos.status != "OPEN":
        raise ValueError("position already closed")

    # A stop or target fills AT its price -- slippage is already reflected by the
    # gap rule in check_exit. Applying it again here would double-charge.
    exit_price = raw_price if reason in (STOP_LOSS, TAKE_PROFIT) else \
        fill_price(raw_price, pos.side, entering=False)

    exit_fee = FEE * pos.qty * exit_price
    gross = (exit_price - pos.entry_price) * pos.qty * pos.direction

    pos.exit_price = exit_price
    pos.exit_time = when
    pos.exit_reason = reason
    pos.fees += exit_fee
    pos.pnl = gross - exit_fee - (pos.fees - exit_fee)   # both legs' fees
    pos.pnl_pct = pos.pnl / pos.notional if pos.notional else 0.0
    pos.status = "CLOSED"
    return pos


def step(positions: list[Position], candle: Candle) -> list[Position]:
    """Advance every open position by one candle. Returns those closed."""
    closed = []
    for p in positions:
        hit = check_exit(p, candle)
        if hit:
            closed.append(close_position(p, hit[0], candle.time, hit[1]))
    return closed


def equity(cash: float, positions: list[Position], price: float) -> float:
    """Cash plus the marked value of everything still open."""
    open_pos = [p for p in positions if p.status == "OPEN"]
    return cash + sum(p.notional + p.unrealised(price) for p in open_pos)


def _self_check() -> None:
    t0 = datetime(2026, 1, 1, 12, 0)
    c = lambda o, h, l, cl, m=0: Candle(t0 + m * BAR, o, h, l, cl)

    # --- a winner is a winner, net of both fees ---
    p = open_position("LONG", 0.1, c(100000, 100000, 100000, 100000), 99000, 102000)
    assert p.entry_price > 100000, "entry slippage must be adverse"
    hit = check_exit(p, c(100500, 102500, 100400, 102400, 1))
    assert hit and hit[1] == TAKE_PROFIT
    close_position(p, hit[0], t0 + BAR, hit[1])
    assert p.pnl > 0 and p.pnl < (102000 - 100000) * 0.1, "fees must reduce a win"

    # --- a loser is a loser ---
    p = open_position("LONG", 0.1, c(100000, 100000, 100000, 100000), 99000, 102000)
    hit = check_exit(p, c(99800, 99900, 98900, 99000, 1))
    assert hit and hit[1] == STOP_LOSS
    close_position(p, hit[0], t0 + BAR, hit[1])
    assert p.pnl < 0

    # --- the rule that matters: stop wins a candle that touches both ---
    p = open_position("LONG", 0.1, c(100000, 100000, 100000, 100000), 99000, 102000)
    both = c(100000, 103000, 98000, 101000, 1)
    assert check_exit(p, both)[1] == STOP_LOSS, "ambiguous candle must resolve pessimistically"

    # --- a gap through the stop fills worse than the stop ---
    p = open_position("LONG", 0.1, c(100000, 100000, 100000, 100000), 99000, 102000)
    price, reason = check_exit(p, c(97000, 97500, 96000, 96500, 1))
    assert reason == STOP_LOSS and price == 97000, "gap must fill at the open"
    close_position(p, price, t0 + BAR, reason)
    stop_pnl = (99000 - 100000) * 0.1
    assert p.pnl < stop_pnl, "gap loss must exceed the nominal stop loss"

    # --- shorts mirror exactly ---
    s = open_position("SHORT", 0.1, c(100000, 100000, 100000, 100000), 101000, 98000)
    assert s.entry_price < 100000, "short entry slippage must be adverse"
    hit = check_exit(s, c(99500, 99600, 97900, 98000, 1))
    assert hit and hit[1] == TAKE_PROFIT
    close_position(s, hit[0], t0 + BAR, hit[1])
    assert s.pnl > 0, "short must profit from a fall"

    s = open_position("SHORT", 0.1, c(100000, 100000, 100000, 100000), 101000, 98000)
    assert check_exit(s, c(100000, 102000, 97000, 99000, 1))[1] == STOP_LOSS

    # --- horizon exit only once expired, and not before ---
    p = open_position("LONG", 0.1, c(100000, 100000, 100000, 100000), 99000, 102000)
    quiet = c(100000, 100100, 99900, 100000, 1)
    assert check_exit(p, quiet) is None, "quiet candle must not close a position"
    late = Candle(p.expires_at, 100000, 100100, 99900, 100050)
    hit = check_exit(p, late)
    assert hit and hit[1] == HORIZON_EXIT

    # --- a flat round trip loses exactly the round-trip cost ---
    p = open_position("LONG", 1.0, c(100000, 100000, 100000, 100000), 90000, 110000)
    close_position(p, 100000, t0 + BAR, HORIZON_EXIT)
    implied = -p.pnl / 100000
    assert abs(implied - COST) < 5e-5, f"flat round trip cost {implied:.4%}, expected {COST:.4%}"

    # --- equity accounting: closing returns cash consistent with PnL ---
    cash = 10000.0
    p = open_position("LONG", 0.05, c(100000, 100000, 100000, 100000), 99000, 102000)
    cash -= p.notional
    assert abs(equity(cash, [p], p.entry_price) - (10000 - p.qty * p.entry_price * FEE)) < 1e-6
    close_position(p, 101000, t0 + BAR, HORIZON_EXIT)
    cash += p.notional + p.pnl
    assert abs(cash - (10000 + p.pnl)) < 1e-9, "cash must reconcile to starting cash plus PnL"

    # --- refuses nonsense rather than booking it ---
    for bad in (("SIDEWAYS", 0.1), ("LONG", 0.0), ("LONG", -1.0)):
        try:
            open_position(bad[0], bad[1], c(100000, 100000, 100000, 100000), 99000, 102000)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")

    print("paper engine self-check passed")


if __name__ == "__main__":
    _self_check()
