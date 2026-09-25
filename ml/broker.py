"""Alpaca brokerage client, and the rebalance that turns model weights into orders.

Paper trading by default. Real money needs BOTH live keys and ALPACA_LIVE=yes,
and the daily job sends nothing at all unless its SEND_ORDERS switch is 'on'.

Rules enforced here whatever the model says:
  no leverage   buys are paid from cash only, never from margin buying power
  no shorting   a sell never exceeds the shares held
  no churn      weight changes smaller than the model's band are skipped
  size cap      one order moves at most MAX_ORDER_USD (default $50,000)
  no repeats    every order carries a client_order_id built from the decision
                date, so the broker itself rejects a second copy

Nothing about the account (balances, sizes) is printed: in a public repository
the Actions log is public.
"""

import json
import os
import time
import urllib.error
import urllib.request

SYMBOL = {"BTC": "IBIT", "GLD": "GLD", "CASH": "SGOV"}   # model asset -> ETF traded
CASH_BUFFER = 0.02            # share of equity kept as real cash for fees and rounding
MIN_ORDER_USD = 1.0           # Alpaca's smallest fractional order
DONE = {"filled", "canceled", "expired", "rejected", "done_for_day"}


class BrokerError(RuntimeError):
    pass


class Alpaca:
    def __init__(self, key: str, secret: str, live: bool = False, timeout: float = 20.0):
        self.base = "https://api.alpaca.markets" if live else "https://paper-api.alpaca.markets"
        self.mode = "live" if live else "paper"
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                        "Content-Type": "application/json"}
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raise BrokerError(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:300]}") from None

    def account(self) -> dict:
        return self._req("GET", "/v2/account")

    def clock(self) -> dict:
        return self._req("GET", "/v2/clock")

    def positions(self) -> dict:
        """{symbol: (qty, price)}"""
        return {p["symbol"]: (float(p["qty"]), float(p["current_price"]))
                for p in self._req("GET", "/v2/positions")}

    def forbid_margin_and_shorts(self) -> None:
        """Ask the broker to block margin and shorting too -- a second lock behind our own."""
        self._req("PATCH", "/v2/account/configurations",
                  {"no_shorting": True, "max_margin_multiplier": "1"})

    def submit(self, symbol: str, side: str, client_id: str, qty: float | None = None,
               notional: float | None = None) -> dict:
        body = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day",
                "client_order_id": client_id}
        if qty is not None:
            body["qty"] = f"{qty:.6f}"
        else:
            body["notional"] = f"{notional:.2f}"
        return self._req("POST", "/v2/orders", body)

    def wait(self, order_id: str, timeout: float = 90.0) -> dict:
        end = time.time() + timeout
        while True:
            o = self._req("GET", f"/v2/orders/{order_id}")
            if o["status"] in DONE or time.time() > end:
                return o
            time.sleep(2)


def from_env() -> Alpaca | None:
    key, secret = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    return Alpaca(key, secret, live=os.environ.get("ALPACA_LIVE", "").lower() == "yes")


def plan(equity: float, held: dict, targets: dict, band: float) -> list:
    """Orders that move `held` {symbol: (qty, price)} toward `targets` {symbol: weight}.

    Pure arithmetic, no network. Sells come first so their cash can fund the buys.
    """
    sells, buys = [], []
    for sym, want in targets.items():
        qty, price = held.get(sym, (0.0, 0.0))
        cur = qty * price / equity if equity > 0 else 0.0
        if not ((want == 0) != (cur == 0) or abs(want - cur) >= band - 1e-9):
            continue
        if want < cur:
            q = qty if want == 0 else min(qty, (cur - want) * equity / price)
            if q * price >= MIN_ORDER_USD:
                sells.append({"symbol": sym, "side": "sell", "qty": q})
        elif (want - cur) * equity >= MIN_ORDER_USD:
            buys.append({"symbol": sym, "side": "buy", "notional": (want - cur) * equity})
    return sells + buys


def rebalance(b: Alpaca, weights: dict, band: float, decision_date: str, record) -> list:
    """Move the account to `weights` {model asset: weight}. `record(order_dict)` is called
    before each order is sent and again with the broker's answer."""
    acct = b.account()
    if acct.get("status") != "ACTIVE" or acct.get("trading_blocked") or acct.get("account_blocked"):
        raise BrokerError("account is not active or is blocked; nothing sent")
    try:
        b.forbid_margin_and_shorts()
    except BrokerError as e:
        print(f"note: could not set broker-side margin lock ({str(e)[:80]}); our own cash-only rule still applies")

    targets = {SYMBOL["BTC"]: weights["BTC"], SYMBOL["GLD"]: weights["GLD"],
               SYMBOL["CASH"]: max(0.0, weights["CASH"] - CASH_BUFFER)}
    cap = float(os.environ.get("MAX_ORDER_USD") or 50_000)
    equity = float(acct["equity"])
    done = []
    for o in plan(equity, b.positions(), targets, band):
        if o["side"] == "buy":
            cash = float(b.account()["cash"])          # re-read: sells have just settled into it
            o["notional"] = min(o["notional"], cash - CASH_BUFFER * equity, cap)
            if o["notional"] < MIN_ORDER_USD:
                continue                                 # not enough cash: never borrow
        elif o["qty"] * b.positions().get(o["symbol"], (0, 0))[1] > cap:
            o["qty"] = cap / b.positions()[o["symbol"]][1]
        o["client_id"] = f"etf-{decision_date}-{o['symbol']}-{o['side']}"
        record(o)
        sent = b.submit(o["symbol"], o["side"], o["client_id"], o.get("qty"), o.get("notional"))
        final = b.wait(sent["id"])
        o.update(broker_order_id=sent["id"], status=final["status"],
                 filled_qty=final.get("filled_qty"), filled_avg_price=final.get("filled_avg_price"))
        record(o)
        done.append(o)
        print(f"{o['side']} {o['symbol']}: {o['status']}")
    return done


def _self_check() -> None:
    held = {"IBIT": (100.0, 50.0), "GLD": (0.0, 0.0), "SGOV": (20.0, 100.0)}   # $5,000 + $2,000
    eq = 10_000.0                                                              # + $3,000 cash
    orders = plan(eq, held, {"IBIT": 0.2, "GLD": 0.3, "SGOV": 0.48}, band=0.2)
    sides = [(o["symbol"], o["side"]) for o in orders]
    # IBIT 50% -> 20%: sell $3,000 = 60 shares. GLD 0 -> 30%: buy. SGOV 20% -> 48%: buy.
    assert sides == [("IBIT", "sell"), ("GLD", "buy"), ("SGOV", "buy")], sides
    assert abs(orders[0]["qty"] - 60) < 1e-9 and abs(orders[1]["notional"] - 3000) < 1e-6

    # Inside the band: no trade. To zero: sell every share, never more (no shorting).
    assert plan(eq, held, {"IBIT": 0.45, "GLD": 0.0, "SGOV": 0.2}, band=0.2) == []
    o = plan(eq, held, {"IBIT": 0.0, "GLD": 0.0, "SGOV": 0.2}, band=0.2)
    assert o == [{"symbol": "IBIT", "side": "sell", "qty": 100.0}], o

    # Buys are paid from cash only: with $500 of cash, a $3,000 buy shrinks to fit.
    class Fake(Alpaca):
        def __init__(self):
            self.pos = {"IBIT": (100.0, 50.0)}; self.sent = []
        def account(self): return {"status": "ACTIVE", "equity": "5500", "cash": "500"}
        def positions(self): return self.pos
        def forbid_margin_and_shorts(self): pass
        def submit(self, symbol, side, client_id, qty=None, notional=None):
            self.sent.append((symbol, side, qty, notional, client_id)); return {"id": "x"}
        def wait(self, order_id, timeout=90.0): return {"status": "filled"}
    f = Fake()
    rebalance(f, {"BTC": 0.9, "GLD": 0.5, "CASH": 0.0}, 0.2, "2026-09-25", lambda o: None)
    buy = [s for s in f.sent if s[1] == "buy"][0]
    assert buy[0] == "GLD" and abs(buy[3] - (500 - 0.02 * 5500)) < 1e-6, buy
    assert buy[4] == "etf-2026-09-25-GLD-buy"
    print("broker self-check passed: no leverage, no shorting, no repeats")


if __name__ == "__main__":
    _self_check()
