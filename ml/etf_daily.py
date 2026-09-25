"""The daily automation: decide after the US close, execute after the next open.

    python ml/etf_daily.py decide              # ~22:30 UTC Mon-Fri
    python ml/etf_daily.py execute             # ~14:45 UTC Mon-Fri, market open in any season
    python ml/etf_daily.py decide --dry-run    # compute and print, write nothing

decide
  1. fetch real daily prices; refuse stale, missing or absurd data
  2. run the model on the latest finished session and record the decision ONCE --
     never overwritten, so the live track record cannot be edited afterwards
  3. rebuild the model portfolio from IBIT's launch, after and before tax and
     charges, next to buy & hold. Days with a recorded decision use it.
execute
  4. only when a broker is configured AND the SEND_ORDERS switch is 'on': move the
     account to the last decision's weights (broker.py enforces the safety rules)

Every run leaves a row in etf_runs, which the dashboard shows as the heartbeat.
A failure also exits non-zero, so GitHub emails the repository owner.
"""

import argparse
import os
import sys
import urllib.request

import pandas as pd

from db import DB
from etf_data import IBIT_START, check, load
from etf_model import signals
from etf_tax_sim import GROSS, EtfCosts, run

CAPITAL = 10_000.0      # the model portfolio's virtual starting dollars
SLABS = (0.26, 0.208, 0.156, 0.104, 0.052)   # the other Indian slab rates incl. cess; 31.2% is the default
DASHBOARD = "https://btc-paper-trader-fbivinays-projects.vercel.app"


def is_live(session: pd.Timestamp) -> bool:
    """True while the session's decision can still be acted on: before the next open."""
    ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    return ny < (session + pd.offsets.BDay(1)).replace(hour=9, minute=30)


def portfolio(btc, gold, tbill, w: pd.DataFrame, costs: EtfCosts = EtfCosts()) -> tuple:
    """(model, buy & hold IBIT) from IBIT's launch under the given tax and charges."""
    px = {"BTC": btc.loc[IBIT_START:], "GLD": gold.loc[IBIT_START:]}
    model = {"BTC": w["w_btc"], "GLD": w["w_gold"]}
    hold = {"BTC": pd.Series(1.0, w.index)}
    return (run(px, model, tbill, costs=costs, capital=CAPITAL),
            run(px, hold, tbill, costs=costs, capital=CAPITAL))


def alert(db: DB, day: pd.Timestamp, s: pd.Series) -> str | None:
    """Phone alert for people who place the trades by hand, in any broker app.

    Compares the new split with the last one alerted -- what such a person now
    holds -- using the ledger's own band, so it fires exactly when a trade is due.
    Delivered through ntfy.sh (free, no account) when NTFY_TOPIC is set.
    """
    new = {k: round(float(s[k]), 4) for k in ("w_btc", "w_gold", "w_cash")}
    last = db.select("etf_alerts", "select=*&order=date.desc", limit=1)
    old = last[0] if last else None
    band = EtfCosts().band
    if old and not any((new[k] == 0) != (old[k] == 0) or abs(new[k] - old[k]) >= band
                       for k in ("w_btc", "w_gold")):
        return None
    was = (lambda k: f"{old[k]:.0%} -> ") if old else (lambda k: "")
    msg = (f"Move to: IBIT {was('w_btc')}{new['w_btc']:.0%}, gold (GLD) {was('w_gold')}{new['w_gold']:.0%}, "
           f"T-bills {was('w_cash')}{new['w_cash']:.0%}. From the {day:%d %b} close; trade at the next US open.")
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode(), method="POST",
                                     headers={"Title": "ETF model: rebalance", "Click": DASHBOARD,
                                              "Tags": "chart_with_upwards_trend"})
        urllib.request.urlopen(req, timeout=20).close()
    db.insert("etf_alerts", {"date": f"{day:%Y-%m-%d}", **new, "delivered": bool(topic)},
              upsert_on="date", keep_existing=True)
    return msg


def decide(db: DB | None) -> str:
    btc, gold, tbill, usdinr = load()
    check(btc, gold)
    sig = signals(btc["close"], gold["close"])
    day = btc.index[-1]

    rows = []
    known = {} if db is None else {
        pd.Timestamp(r["date"]): r for r in db.select("etf_decisions", "select=date,w_btc,w_gold")}
    for d in sig.loc[IBIT_START:].index:
        if d in known:
            continue
        s = sig.loc[d]
        rows.append({"date": f"{d:%Y-%m-%d}", "live": bool(d == day and is_live(d)),
                     "btc_votes": int(s.btc_votes), "btc_vol": round(float(s.btc_vol), 4),
                     "gold_votes": int(s.gold_votes), "gold_vol": round(float(s.gold_vol), 4),
                     "w_btc": round(float(s.w_btc), 4), "w_gold": round(float(s.w_gold), 4),
                     "w_cash": round(float(s.w_cash), 4),
                     "btc_close": round(float(btc.loc[d, "close"]), 4),
                     "gold_close": round(float(gold.loc[d, "close"]), 4),
                     "tbill": round(float(tbill.asof(d)), 5), "usdinr": round(float(usdinr.asof(d)), 3)})
        known[d] = rows[-1]

    # The portfolio follows the decisions as recorded, not as recomputed today.
    w = pd.DataFrame({"w_btc": {d: r["w_btc"] for d, r in known.items()},
                      "w_gold": {d: r["w_gold"] for d, r in known.items()}}).sort_index()
    (m_net, h_net), (m_gross, h_gross) = portfolio(btc, gold, tbill, w), portfolio(btc, gold, tbill, w, GROSS)
    res = {"model_net": m_net, "model_gross": m_gross, "hold_net": h_net, "hold_gross": h_gross}
    equity = pd.DataFrame({k: r.equity for k, r in res.items()})
    equity["btc_close"], equity["gold_close"] = btc["close"], gold["close"]
    by_slab = []
    for slab in SLABS:
        m, h = portfolio(btc, gold, tbill, w, EtfCosts(slab=slab))
        by_slab += [{"date": f"{d:%Y-%m-%d}", "slab": slab, "model_net": round(float(a), 4),
                     "hold_net": round(float(b), 4)} for d, a, b in zip(m.equity.index, m.equity, h.equity)]
    t = res["model_net"].trades
    trades = [{"date": f"{r.date:%Y-%m-%d}", "etf": "IBIT" if r.etf == "BTC" else r.etf, "side": r.side,
               "units": round(r.units, 6), "price": round(r.price, 4),
               "value": round(r.units * r.price, 2), "gain": round(r.gain, 2)} for r in t.itertuples()]

    s = sig.loc[day]
    summary = (f"{day:%Y-%m-%d} close -> IBIT {s.w_btc:.0%} (votes {int(s.btc_votes)}/8), "
               f"gold {s.w_gold:.0%} (votes {int(s.gold_votes)}/8), T-bills {s.w_cash:.0%}; "
               f"{'live' if is_live(day) else 'after the next open, recorded as history'}")
    if db is None:
        print(summary)
        print(f"model after tax {res['model_net'].equity.iloc[-1]:,.0f}  before {res['model_gross'].equity.iloc[-1]:,.0f}  "
              f"hold after {res['hold_net'].equity.iloc[-1]:,.0f}  before {res['hold_gross'].equity.iloc[-1]:,.0f}  "
              f"(from {CAPITAL:,.0f} on {IBIT_START:%Y-%m-%d}); {len(rows)} decisions would be written")
        return summary

    if rows:
        db.insert("etf_decisions", rows, upsert_on="date", keep_existing=True)
    db.insert("etf_equity", [{"date": f"{d:%Y-%m-%d}", **{k: round(float(v), 4) for k, v in r.items()}}
                             for d, r in equity.iterrows()], upsert_on="date")
    db.insert("etf_equity_slab", by_slab, upsert_on="slab,date")
    if trades:
        db.insert("etf_trades", trades, upsert_on="date,etf,side")
    if is_live(day) and (sent := alert(db, day, s)):
        summary += f" | alert: {sent}"
    return summary


def execute(db: DB | None) -> str:
    if os.environ.get("SEND_ORDERS", "off").lower() != "on":
        return "skipped: the SEND_ORDERS switch is off"
    import broker
    b = broker.from_env()
    if b is None:
        return "skipped: no broker keys configured"
    if db is None:
        return f"dry run: would rebalance the {b.mode} account"
    last = db.select("etf_decisions", "select=*&order=date.desc", limit=1)[0]
    age = (pd.Timestamp.now() - pd.Timestamp(last["date"])).days
    if age > 5:
        raise RuntimeError(f"newest decision is {age} days old; refusing to trade on it")
    if db.select("etf_orders", f"select=id&decision_date=eq.{last['date']}&mode=eq.{b.mode}", limit=1):
        return f"skipped: decision {last['date']} was already executed"
    if not b.clock()["is_open"]:
        return "skipped: market closed today"

    def record(o):
        db.insert("etf_orders", {"decision_date": last["date"], "mode": b.mode, "symbol": o["symbol"],
                                 "side": o["side"], "qty": o.get("qty"), "notional": o.get("notional"),
                                 "broker_order_id": o.get("broker_order_id"), "status": o.get("status", "sending"),
                                 "filled_qty": o.get("filled_qty"), "filled_avg_price": o.get("filled_avg_price")},
                  upsert_on="decision_date,mode,symbol,side")

    orders = broker.rebalance(b, {"BTC": last["w_btc"], "GLD": last["w_gold"], "CASH": last["w_cash"]},
                              EtfCosts().band, last["date"], record)
    bad = [o for o in orders if o["status"] != "filled"]
    if bad:
        raise RuntimeError(f"{len(bad)} order(s) not filled: " + ", ".join(f"{o['symbol']} {o['status']}" for o in bad))
    return f"{b.mode}: {len(orders)} order(s) for decision {last['date']} filled"


def _self_check() -> None:
    """Alerts fire when a hand-trader must act, and stay quiet otherwise. Offline."""
    class FakeDB:
        def __init__(self): self.rows = []
        def select(self, table, query="", limit=None): return self.rows[-1:]
        def insert(self, table, row, upsert_on=None, keep_existing=False): self.rows.append(row)
    os.environ.pop("NTFY_TOPIC", None)                       # never post from a test
    db, day = FakeDB(), pd.Timestamp("2026-09-25")
    split = lambda b, g: pd.Series({"w_btc": b, "w_gold": g, "w_cash": 1 - b - g})
    assert alert(db, day, split(0.77, 0.06)), "the first decision must set a starting split"
    assert alert(db, day, split(0.70, 0.10)) is None, "a move inside the band is not worth a trade"
    assert alert(db, day, split(0.50, 0.10)), "a move past the band must alert"
    assert alert(db, day, split(0.50, 0.0)), "leaving an ETF completely must always alert"
    print("daily job self-check passed: alerts fire exactly when a trade is due")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["decide", "execute", "check"])
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    a = ap.parse_args()
    if a.job == "check":
        return _self_check()
    db = None if a.dry_run else DB()
    try:
        detail = (decide if a.job == "decide" else execute)(db)
        status = "skipped" if detail.startswith("skipped") else "ok"
    except Exception as e:
        if db is not None:
            db.insert("etf_runs", {"job": a.job, "status": "error", "detail": f"{type(e).__name__}: {e}"[:500]})
        raise
    if db is not None:
        db.insert("etf_runs", {"job": a.job, "status": status, "detail": detail[:500]})
    print(f"{a.job}: {detail}")


if __name__ == "__main__":
    sys.exit(main())
