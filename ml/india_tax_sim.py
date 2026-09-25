"""Long/flat simulator with every Indian charge applied, no leverage.

Position is either 100% BTC or 100% cash. Signals are computed from the close of
day t and executed at the OPEN of day t+1 -- never at the close that produced
the signal.

Charges, per the Indian VDA regime (Section 115BBH / 194S):

  exchange fee    per side, on trade value
  GST             18% on the exchange fee
  slippage        per side, adverse
  TDS             1% of every SALE's proceeds, withheld at source. It is an
                  advance tax credit: refunded via ITR once real liability is
                  settled. Modelled as withheld at the sale and refunded on the
                  1 October after the financial year closes (31 March), which is
                  deliberately slow -- ITR refunds are not instant.
  tax             31.2% (30% + 4% cess) on the gain of each profitable sale.
                  NO loss offset: a losing sale reduces no one's tax, ever, not
                  even against a later gain on the same asset. Fees are not
                  deductible. Paid immediately at the sale, which is
                  conservative on timing (real advance tax is quarterly).

Because tax is paid in full at each sale, all TDS is excess prepayment and is
refunded in full later. The totals are exact; only the timing is pessimistic.

Reported equity is LIQUIDATION value: what you would walk away with if you sold
everything today and paid what is owed. Mark-to-market equity would flatter a
strategy sitting on an untaxed gain.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TAX = 0.312
TDS = 0.01
GST = 0.18


@dataclass
class Costs:
    fee: float = 0.002        # exchange fee per side (Indian retail spot is 0.1-0.5%)
    slip: float = 0.0005      # slippage per side
    gst: float = GST
    tax: float = TAX
    tds: float = TDS

    @property
    def fee_all_in(self) -> float:
        return self.fee * (1 + self.gst)


@dataclass
class Result:
    equity: pd.Series                 # daily liquidation value
    trades: pd.DataFrame
    tax_paid: float
    fees_paid: float
    tds_withheld: float
    wasted_losses: float              # losses that no-offset made unusable

    def cagr(self) -> float:
        e = self.equity
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        return (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0

    def max_dd(self) -> float:
        e = self.equity
        return float((e / e.cummax() - 1).min())

    def yearly(self) -> pd.Series:
        e = self.equity
        ye = e.groupby(e.index.year).last()
        start = pd.Series([e.iloc[0]], index=[ye.index[0] - 1])
        return pd.concat([start, ye]).pct_change().dropna()


def simulate(df: pd.DataFrame, signal: np.ndarray, capital: float = 100.0,
             costs: Costs = Costs()) -> Result:
    """df needs open_time, open, close. signal[t] in {0,1}, decided at close of t."""
    t = pd.to_datetime(df["open_time"]).dt.tz_localize(None).to_numpy()
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    sig = np.asarray(signal, dtype=float)
    n = len(df)

    cash, qty, entry_px = capital, 0.0, 0.0
    tds_pending: dict[pd.Timestamp, float] = {}      # refund date -> amount
    tax_paid = fees_paid = tds_total = wasted = 0.0
    trades, equity = [], np.empty(n)

    def refund_date(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        fy_end_year = ts.year if ts.month <= 3 else ts.year + 1
        return pd.Timestamp(year=fy_end_year, month=10, day=1)

    for i in range(n):
        now = pd.Timestamp(t[i])

        # TDS refunds that have come due.
        for d in [d for d in tds_pending if d <= now]:
            cash += tds_pending.pop(d)

        # Execute yesterday's decision at today's open.
        want = sig[i - 1] if i > 0 else 0.0
        if want >= 0.5 and qty == 0 and cash > 0:
            px = o[i] * (1 + costs.slip)
            fee = cash * costs.fee_all_in
            qty = (cash - fee) / px
            entry_px = px
            fees_paid += fee
            trades.append((now, "BUY", px, qty, 0.0, 0.0))
            cash = 0.0
        elif want < 0.5 and qty > 0:
            px = o[i] * (1 - costs.slip)
            proceeds = qty * px
            fee = proceeds * costs.fee_all_in
            tds = proceeds * costs.tds
            gain = qty * (px - entry_px)
            tax = costs.tax * gain if gain > 0 else 0.0
            if gain < 0:
                wasted += -gain
            cash += proceeds - fee - tds - tax
            tds_pending[refund_date(now)] = tds_pending.get(refund_date(now), 0.0) + tds
            fees_paid += fee; tax_paid += tax; tds_total += tds
            trades.append((now, "SELL", px, qty, gain, tax))
            qty = 0.0

        # Liquidation value at today's close.
        pending = sum(tds_pending.values())
        if qty > 0:
            px = c[i] * (1 - costs.slip)
            proceeds = qty * px
            gain = qty * (px - entry_px)
            liq = proceeds - proceeds * costs.fee_all_in - (costs.tax * gain if gain > 0 else 0.0)
            equity[i] = cash + pending + liq
        else:
            equity[i] = cash + pending

    tr = pd.DataFrame(trades, columns=["time", "side", "price", "qty", "gain", "tax"])
    return Result(pd.Series(equity, index=pd.DatetimeIndex(t)), tr,
                  tax_paid, fees_paid, tds_total, wasted)


def simulate_ls(df: pd.DataFrame, signal: np.ndarray, funding_daily: np.ndarray | None = None,
                capital: float = 100.0, costs: Costs = Costs(),
                fut_fee: float = 0.0005) -> Result:
    """Long / short / flat, never leveraged: |position| is always <= equity.

    +1  long spot BTC        (spot fee, GST, slippage, TDS, 31.2% no offset)
    -1  short BTC perpetual at 1x notional
         - futures fee + GST per side, slippage
         - funding: a short RECEIVES funding when the rate is positive and PAYS
           when it is negative; longs held as spot pay no funding
         - the same 31.2% tax on each profitable close, no loss offset, and the
           same 1% TDS on close notional (conservative: the treatment of
           crypto derivatives in India is not settled, so they are charged as
           though they were spot VDA transfers)
     0  cash

    Switching straight from long to short closes one and opens the other on the
    same open, paying both sets of costs.
    """
    t = pd.to_datetime(df["open_time"]).dt.tz_localize(None).to_numpy()
    o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
    sig = np.asarray(signal, dtype=float)
    fund = np.zeros(len(df)) if funding_daily is None else np.asarray(funding_daily, float)
    n = len(df)
    ffee = fut_fee * (1 + costs.gst)

    cash = capital
    pos, qty, entry = 0, 0.0, 0.0
    tds_pending: dict = {}
    tax_paid = fees_paid = tds_total = wasted = 0.0
    trades, equity = [], np.empty(n)

    def refund_date(ts):
        ts = pd.Timestamp(ts)
        return pd.Timestamp(year=ts.year if ts.month <= 3 else ts.year + 1, month=10, day=1)

    def close(i, now):
        nonlocal cash, pos, qty, entry, tax_paid, fees_paid, tds_total, wasted
        if pos == 1:
            px = o[i] * (1 - costs.slip); fee_rate = costs.fee_all_in
            gain = qty * (px - entry)
        else:
            px = o[i] * (1 + costs.slip); fee_rate = ffee
            gain = qty * (entry - px)
        notional = qty * px
        fee = notional * fee_rate; tds = notional * costs.tds
        tax = costs.tax * gain if gain > 0 else 0.0
        if gain < 0: wasted += -gain
        if pos == 1:
            cash += notional - fee - tds - tax
        else:
            # the short's margin was the entry notional; settle the difference
            cash += qty * entry + gain - fee - tds - tax
        tds_pending[refund_date(now)] = tds_pending.get(refund_date(now), 0.0) + tds
        fees_paid += fee; tax_paid += tax; tds_total += tds
        trades.append((now, "CLOSE_LONG" if pos == 1 else "CLOSE_SHORT", px, qty, gain, tax))
        pos, qty = 0, 0.0

    def open_(i, now, side):
        nonlocal cash, pos, qty, entry, fees_paid
        if cash <= 0: return
        if side == 1:
            px = o[i] * (1 + costs.slip); fee = cash * costs.fee_all_in
            qty = (cash - fee) / px; cash = 0.0
        else:
            px = o[i] * (1 - costs.slip); fee = cash * ffee
            qty = (cash - fee) / px; cash = 0.0    # 1x: notional equals equity
        entry = px; fees_paid += fee; pos = side
        trades.append((now, "OPEN_LONG" if side == 1 else "OPEN_SHORT", px, qty, 0.0, 0.0))

    for i in range(n):
        now = pd.Timestamp(t[i])
        for d_ in [d_ for d_ in tds_pending if d_ <= now]:
            cash += tds_pending.pop(d_)

        want = int(np.sign(sig[i - 1])) if i > 0 else 0
        if want != pos:
            if pos != 0: close(i, now)
            if want != 0: open_(i, now, want)

        # funding accrues on an open short (received when positive)
        if pos == -1 and fund[i] != 0:
            cash += qty * c[i] * fund[i]

        pending = sum(tds_pending.values())
        if pos == 1:
            px = c[i] * (1 - costs.slip); g = qty * (px - entry)
            liq = qty * px * (1 - costs.fee_all_in) - (costs.tax * g if g > 0 else 0.0)
        elif pos == -1:
            px = c[i] * (1 + costs.slip); g = qty * (entry - px)
            liq = qty * entry + g - qty * px * ffee - (costs.tax * g if g > 0 else 0.0)
        else:
            liq = 0.0
        equity[i] = cash + pending + liq

    tr = pd.DataFrame(trades, columns=["time", "side", "price", "qty", "gain", "tax"])
    return Result(pd.Series(equity, index=pd.DatetimeIndex(t)), tr,
                  tax_paid, fees_paid, tds_total, wasted)


def _self_check_ls() -> None:
    days = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    no = Costs(fee=0.0, slip=0.0, gst=0.0)
    fall = [100, 100, 100, 50, 50, 50, 50, 50, 50, 50]
    df = pd.DataFrame({"open_time": days, "open": fall, "close": fall})

    # A short into a halving must make money, and that gain must be taxed.
    r = simulate_ls(df, np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]), costs=no, fut_fee=0.0)
    assert r.tax_paid > 0, "a profitable short must be taxed"
    assert abs(r.equity.iloc[-1] - (100 + 50 - 0.312 * 50)) < 1e-6, r.equity.iloc[-1]

    # A short into a rally loses, and the loss is wasted (no offset).
    rise = [100, 100, 100, 150, 150, 150, 150, 150, 150, 150]
    df2 = pd.DataFrame({"open_time": days, "open": rise, "close": rise})
    r = simulate_ls(df2, np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]), costs=no, fut_fee=0.0)
    assert r.wasted_losses > 0 and r.tax_paid == 0
    assert r.equity.iloc[-1] < 100

    # Positive funding pays a short.
    flat = pd.DataFrame({"open_time": days, "open": 100.0, "close": 100.0})
    r = simulate_ls(flat, np.array([-1] * 9 + [0]), funding_daily=np.full(10, 0.001),
                    costs=no, fut_fee=0.0)
    assert r.equity.iloc[-1] > 100, "positive funding must be received by a short"

    # Long leg must agree with the long-only simulator.
    up = pd.DataFrame({"open_time": days, "open": [100]*4 + [200]*6, "close": [100]*4 + [200]*6})
    a = simulate(up, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), costs=no)
    b = simulate_ls(up, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), costs=no, fut_fee=0.0)
    assert abs(a.equity.iloc[-1] - b.equity.iloc[-1]) < 1e-9
    print("long/short simulator self-check passed")


def _self_check() -> None:
    days = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    flat = pd.DataFrame({"open_time": days, "open": 100.0, "close": 100.0})
    no = Costs(fee=0.0, slip=0.0, gst=0.0)

    # Never trading must leave capital untouched.
    r = simulate(flat, np.zeros(10), costs=no)
    assert abs(r.equity.iloc[-1] - 100) < 1e-9

    # A round trip at an unchanged price loses exactly the frictions, no tax.
    costs = Costs(fee=0.002, slip=0.0)
    sig = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    r = simulate(flat, sig, costs=costs)
    assert r.tax_paid == 0
    expect = 100 * (1 - costs.fee_all_in) * (1 - costs.fee_all_in)
    assert abs(r.equity.iloc[-1] - expect) < 1e-6, (r.equity.iloc[-1], expect)

    # A winning round trip is taxed at 31.2% of the gain, fees NOT deductible.
    up = flat.copy(); up.loc[4:, ["open", "close"]] = 200.0
    r = simulate(up, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), costs=no)
    gain = 100 / 100 * (200 - 100)
    assert abs(r.tax_paid - 0.312 * gain) < 1e-9
    assert abs(r.equity.iloc[-1] - (200 - 0.312 * 100)) < 1e-6

    # NO LOSS OFFSET: lose 50, then gain 50 back. A normal regime taxes nothing
    # (net gain zero). This regime taxes the second leg in full.
    path = [100, 100, 50, 50, 50, 100, 100, 100, 100, 100]
    wl = pd.DataFrame({"open_time": days, "open": path, "close": path})
    sig = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])   # buy@100 sell@50, buy@50 sell@100
    r = simulate(wl, sig, costs=no)
    assert r.wasted_losses > 0, "the first loss must be recorded as unusable"
    assert r.tax_paid > 0, "the second, winning trade must be taxed despite the prior loss"
    assert r.equity.iloc[-1] < 100, "net zero price move must leave you BELOW start"

    # TDS is withheld on the sale and refunded later, so after refund it costs nothing.
    long_days = pd.date_range("2020-02-01", periods=400, freq="D", tz="UTC")
    lf = pd.DataFrame({"open_time": long_days, "open": 100.0, "close": 100.0})
    s = np.zeros(400); s[:5] = 1
    r = simulate(lf, s, costs=no)
    assert r.tds_withheld > 0
    assert abs(r.equity.iloc[-1] - 100) < 1e-9, "refunded TDS must restore capital"

    print("india tax simulator self-check passed")


if __name__ == "__main__":
    _self_check()
    _self_check_ls()
