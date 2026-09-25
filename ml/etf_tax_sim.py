"""Portfolio simulator for US-listed ETFs held by an Indian resident.

Rules modelled (confirm with a Chartered Accountant before relying on them):

  instruments     US ETFs (spot Bitcoin, gold, ...), one column each
  execution       a target weight decided after the close of day i is traded at
                  the open of the next trading day; a change smaller than `band`
                  is skipped, except a move to or from zero
  per-trade cost  brokerage + spread, per side
  forex           markup on the INR->USD remittance at the start and on the
                  USD->INR repatriation at the end -- once each, because the
                  money stays in the US account between trades
  idle cash       parked in a T-bill ETF, earning the T-bill rate minus its fee;
                  that income is taxed at the slab rate and cannot absorb losses
  short selling   not permitted (RBI LRS prohibits it) -- weights are 0..1
  tax             Indian capital gains on foreign securities, NOT VDA:
                    held < 24 months  -> short-term, taxed at the slab rate
                    held >= 24 months -> long-term, 12.5% + 4% cess = 13%
                  lots are sold oldest first; gains are netted per financial
                  year (April-March):
                    short-term losses offset short-term AND long-term gains
                    long-term losses offset only long-term gains
                    unabsorbed losses carry forward 8 years
                  tax is reserved at each profitable sale and settled at the
                  financial year end, when netting refunds any excess reserve
  TCS             20% on remittances above 10 lakh is refundable; not modelled
  rupee           results are in USD terms; INR/USD moves hit every strategy alike

Equity is LIQUIDATION value: what you would receive if you sold everything
today, paid this year's tax after netting, and repatriated.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LT_DAYS = 730


@dataclass
class EtfCosts:
    slab: float = 0.312       # short-term rate incl. cess -- top slab, the conservative case
    ltcg: float = 0.13        # 12.5% + 4% cess
    cost: float = 0.0005      # per side
    fx: float = 0.015         # markup each way, once
    cash_fee: float = 0.001   # T-bill ETF expense, taken off the T-bill rate
    band: float = 0.2         # smallest weight change worth a trade


@dataclass
class EtfResult:
    equity: pd.Series
    trades: pd.DataFrame
    tax_paid: float
    capital: float = 100.0    # returns are measured from the money put in, so forex counts

    def cagr(self) -> float:
        e = self.equity; yrs = (e.index[-1] - e.index[0]).days / 365.25
        return (e.iloc[-1] / self.capital) ** (1 / yrs) - 1 if yrs > 0 and e.iloc[-1] > 0 else -1.0

    def max_dd(self) -> float:
        e = self.equity; return float((e / e.cummax() - 1).min())

    def yearly(self) -> pd.Series:
        e = self.equity
        ye = e.groupby(e.index.year).last()
        return pd.concat([pd.Series([self.capital], index=[ye.index[0] - 1]), ye]).pct_change().dropna()


def fy_tax(st: float, lt: float, carry: list, fy: int, c: EtfCosts):
    """Tax for one financial year after set-off and brought-forward losses.

    Returns (tax, remaining_carry, loss_used). `carry` holds (fy, amount, kind)
    with amount > 0, oldest first; entries older than 8 years have lapsed.
    """
    used = 0.0
    # Intra-year: a short-term loss may absorb a long-term gain (not the reverse).
    if st < 0 and lt > 0:
        x = min(-st, lt); st += x; lt -= x; used += x
    st_pos, lt_pos = max(st, 0.0), max(lt, 0.0)
    left = []
    for cfy, amt, kind in carry:
        if fy - cfy > 8:
            continue
        if kind == "st":
            x = min(amt, st_pos); st_pos -= x; amt -= x; used += x
            x = min(amt, lt_pos); lt_pos -= x; amt -= x; used += x
        else:
            x = min(amt, lt_pos); lt_pos -= x; amt -= x; used += x
        if amt > 1e-12:
            left.append((cfy, amt, kind))
    if st < 0: left.append((fy, -st, "st"))
    if lt < 0: left.append((fy, -lt, "lt"))
    return c.slab * st_pos + c.ltcg * lt_pos, left, used


def _sell_fifo(lots: list, units: float, px: float, today: int) -> list:
    """Take `units` from the oldest lots first -> [(proceeds, gain, long_term), ...]."""
    out = []
    while units > 1e-12 and lots:
        u, basis, bought = lots[0]
        take = min(u, units)
        part = basis * take / u
        out.append((take * px, take * px - part, today - bought >= LT_DAYS))
        if take >= u - 1e-12:
            lots.pop(0)
        else:
            lots[0] = [u - take, basis - part, bought]
        units -= take
    return out


def simulate(opens: pd.DataFrame, closes: pd.DataFrame, weights: pd.DataFrame,
             rate: pd.Series | None = None, expense: dict | None = None,
             capital: float = 100.0, costs: EtfCosts = EtfCosts()) -> EtfResult:
    """opens/closes/weights share one trading-day index and one column per ETF.

    rate: annual T-bill yield (0.05 = 5%) for idle cash.
    expense: {etf: annual fee} charged on SIMULATED holdings; real ETF prices
    already have the fee taken out of them.
    """
    idx = pd.DatetimeIndex(opens.index)
    names = list(opens.columns)
    O, C = opens.to_numpy(float), closes[names].to_numpy(float)
    W = np.clip(weights.reindex(index=idx, columns=names).fillna(0.0).to_numpy(float), 0, 1)
    n, k = O.shape
    fy = (idx.year + (idx.month >= 4)).to_numpy()     # Indian FY, named by the year it ends in
    day = (idx - idx[0]).days.to_numpy()
    gap = np.r_[0, np.diff(day)]
    yld = np.zeros(n) if rate is None else rate.reindex(idx, method="ffill").fillna(0.0).to_numpy()
    yld = yld - costs.cash_fee
    decay = [(1 - (expense or {}).get(a, 0.0)) ** (1 / 365) for a in names]

    cash = capital * (1 - costs.fx)                   # INR -> USD once, at the start
    lots = [[] for _ in names]                        # per ETF: [units, basis, day bought]
    st = lt = reserve = tax_paid = 0.0
    carry, cur_fy = [], fy[0]
    trades, equity = [], np.empty(n)

    for i in range(n):
        if fy[i] != cur_fy:                           # settle last year's tax after netting
            tax, carry, _ = fy_tax(st, lt, carry, cur_fy, costs)
            cash += reserve - tax; tax_paid += tax
            st = lt = reserve = 0.0; cur_fy = fy[i]
        if yld[i] > 0 and cash > 0:                   # T-bill income since last session, slab-taxed
            income = cash * yld[i] / 365 * gap[i]
            cash += income * (1 - costs.slab); tax_paid += income * costs.slab
        for a in range(k):
            if decay[a] < 1:
                for lot in lots[a]:
                    lot[0] *= decay[a] ** gap[i]

        if i > 0:
            held = np.array([sum(l[0] for l in lots[a]) for a in range(k)])
            total = cash + held @ O[i]
            cur = held * O[i] / total
            want = W[i - 1]
            go = ((want == 0) != (cur == 0)) | (np.abs(want - cur) >= costs.band - 1e-9)
            for a in np.flatnonzero(go & (want < cur)):       # sell first, to fund the buys
                units = held[a] if want[a] == 0 else (cur[a] - want[a]) * total / O[i, a]
                px = O[i, a] * (1 - costs.cost)
                for proceeds, gain, long_term in _sell_fifo(lots[a], units, px, day[i]):
                    if long_term: lt += gain
                    else:         st += gain
                    hold = (costs.ltcg if long_term else costs.slab) * max(gain, 0.0)
                    cash += proceeds - hold; reserve += hold
                trades.append((idx[i], names[a], "SELL", units, px))
            for a in np.flatnonzero(go & (want > cur)):
                spend = min(cash, (want[a] - cur[a]) * total)
                if spend > 0:
                    px = O[i, a] * (1 + costs.cost)
                    lots[a].append([spend / px, spend, day[i]])
                    cash -= spend
                    trades.append((idx[i], names[a], "BUY", spend / px, px))

        hs, hl, value = st, lt, 0.0                   # liquidation value at the close
        for a in range(k):
            px = C[i, a] * (1 - costs.cost)
            for u, basis, bought in lots[a]:
                value += u * px
                if day[i] - bought >= LT_DAYS: hl += u * px - basis
                else:                          hs += u * px - basis
        tax_now, _, _ = fy_tax(hs, hl, carry, cur_fy, costs)
        equity[i] = (cash + reserve + value - tax_now) * (1 - costs.fx)

    return EtfResult(pd.Series(equity, index=idx),
                     pd.DataFrame(trades, columns=["date", "etf", "side", "units", "price"]),
                     tax_paid, capital)


def _self_check() -> None:
    zero = EtfCosts(cost=0.0, fx=0.0, cash_fee=0.0, band=0.0)

    def run(path, w, start="2021-05-03", opens=None, costs=zero, rate=None):
        idx = pd.bdate_range(start, periods=len(path))
        c = pd.DataFrame({"X": np.asarray(path, float)}, index=idx)
        o = c if opens is None else pd.DataFrame({"X": np.asarray(opens, float)}, index=idx)
        w = pd.DataFrame({"X": np.asarray(w, float)}, index=idx)
        return simulate(o, c, w, rate=None if rate is None else pd.Series(rate, idx), costs=costs)

    # Never trading only costs the two forex conversions.
    r = run([100.0] * 20, [0] * 20, costs=EtfCosts(cost=0, fx=0.015, cash_fee=0, band=0))
    assert abs(r.equity.iloc[-1] - 100 * 0.985 * 0.985) < 1e-9

    # Short-term win: taxed at slab.
    r = run([100] * 3 + [200] * 7, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert abs(r.equity.iloc[-1] - (200 - 0.312 * 100)) < 1e-6, r.equity.iloc[-1]

    # THE DIFFERENCE FROM VDA: lose 50, gain 50 back, same year -> no tax, back to 100.
    r = run([100, 100, 50, 50, 50, 100, 100, 100], [1, 0, 0, 1, 0, 0, 0, 0])
    assert abs(r.equity.iloc[-1] - 100) < 1e-6, f"loss must offset gain, got {r.equity.iloc[-1]}"

    # Carry-forward: lose 40 in FY22, gain 40 in FY23 -> the carried loss absorbs it.
    idx = pd.bdate_range("2021-05-03", "2022-06-30")
    path = np.where(idx < "2021-05-10", 100.0, np.where(idx < "2022-05-02", 60.0, 100.0))
    w = ((idx < "2021-05-14") | ((idx >= "2022-04-11") & (idx < "2022-05-09"))).astype(float)
    r = run(path, w)
    assert abs(r.equity.iloc[-1] - 100) < 1e-6, f"carried loss must absorb gain, got {r.equity.iloc[-1]}"

    # Long-term: held over 24 months, taxed at 13% not slab.
    r = run([100] * 5 + [200] * 595, [1] * 580 + [0] * 20)
    assert abs(r.equity.iloc[-1] - (200 - 0.13 * 100)) < 1e-6, r.equity.iloc[-1]

    # A decision at a close trades at the NEXT open, never the close it saw.
    r = run([15, 25, 35, 45], [1, 1, 1, 1], opens=[10, 20, 30, 40])
    assert r.trades.iloc[0]["price"] == 20

    # Halving a position sells half, oldest lot first; the tax outcome is unchanged.
    r = run([100] * 3 + [200] * 5, [1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert abs(r.trades.iloc[1]["units"] - 0.5) < 1e-9
    assert abs(r.equity.iloc[-1] - (200 - 0.312 * 100)) < 1e-6, r.equity.iloc[-1]

    # A weight change inside the band is not worth a trade.
    r = run([100] * 5, [1, 1, 0.9, 0.9, 0.9], costs=EtfCosts(cost=0, fx=0, cash_fee=0, band=0.2))
    assert (r.trades["side"] == "SELL").sum() == 0

    # Idle cash earns the T-bill rate, taxed at slab: 5% for a year -> about 3.4%.
    idx = pd.bdate_range("2021-04-01", "2022-03-31")
    r = run([100.0] * len(idx), [0] * len(idx), start="2021-04-01", rate=[0.05] * len(idx))
    assert abs(r.equity.iloc[-1] / 100 - 1 - 0.05 * 0.688) < 0.002, r.equity.iloc[-1]

    print("ETF tax simulator self-check passed")


if __name__ == "__main__":
    _self_check()
