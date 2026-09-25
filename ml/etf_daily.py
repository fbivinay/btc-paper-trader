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

import pandas as pd

from db import DB
from etf_data import IBIT_START, check, load
from etf_model import signals
from etf_tax_sim import GROSS, EtfCosts, run

CAPITAL = 10_000.0      # the model portfolio's virtual starting dollars


def is_live(session: pd.Timestamp) -> bool:
    """True while the session's decision can still be acted on: before the next open."""
    ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    return ny < (session + pd.offsets.BDay(1)).replace(hour=9, minute=30)


def portfolio(btc, gold, tbill, w: pd.DataFrame) -> dict:
    """The model and buy & hold, after and before tax and charges, from IBIT's launch."""
    px = {"BTC": btc.loc[IBIT_START:], "GLD": gold.loc[IBIT_START:]}
    model = {"BTC": w["w_btc"], "GLD": w["w_gold"]}
    hold = {"BTC": pd.Series(1.0, w.index)}
    return {"model_net": run(px, model, tbill, capital=CAPITAL),
            "model_gross": run(px, model, tbill, costs=GROSS, capital=CAPITAL),
            "hold_net": run(px, hold, tbill, capital=CAPITAL),
            "hold_gross": run(px, hold, tbill, costs=GROSS, capital=CAPITAL)}


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
    res = portfolio(btc, gold, tbill, w)
    equity = pd.DataFrame({k: r.equity for k, r in res.items()})
    equity["btc_close"], equity["gold_close"] = btc["close"], gold["close"]
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
    if trades:
        db.insert("etf_trades", trades, upsert_on="date,etf,side")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["decide", "execute"])
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    a = ap.parse_args()
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
