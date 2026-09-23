"""One pass of the live pipeline: candle -> prediction -> decision -> paper trades.

    python ml/live_loop.py            # process every candle since the last one seen
    python ml/live_loop.py --dry-run  # compute everything, write nothing
    python ml/live_loop.py --once     # only the most recent closed candle

Run every 5 minutes. It is written to be safe to run late, twice, or after an
outage:

  * Everything keys off candle_time, never wall clock. A run that fires 20
    minutes late processes the candles it missed rather than skipping them,
    which matters because GitHub Actions cron routinely runs late.
  * The decisions insert upserts on candle_time, so a double run cannot create
    two decisions for one candle.
  * Open positions are advanced through EVERY missed candle before any new
    entry is considered, so a stop that would have been hit during an outage is
    still honoured at the right price.

Cost control: Jev is called at most once per candle, and only when at least one
active user's confidence gate would let the signal through. The free tier
rate-limits hard, and a strategy label changes nothing on a candle that is
going to be rejected anyway.
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch

import features as F
import jev_client
import paper_engine as PE
import regime as R
import risk_engine as RE
from config import COST, HORIZON, horizon_label
from db import DB, DBError, load_env
from train import SEQ_LEN, LSTMClassifier, NOT_FEATURES

BINANCE = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
BAR = timedelta(minutes=5)

# Feature warmup: vol_pctile needs 7 days, then SEQ_LEN bars of model context.
WARMUP_BARS = F.BARS_PER_DAY * 7 + SEQ_LEN + 50
MAX_BACKFILL = 288          # refuse to replay more than a day in one run

LABELS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


# --------------------------------------------------------------------- data ---

def fetch_recent(bars: int = WARMUP_BARS) -> pd.DataFrame:
    """Recent closed candles from Binance, oldest first.

    The live REST API, not the data.binance.vision dumps: those are published
    daily and are hours stale. The final kline is the one still forming and is
    always dropped -- acting on a partial candle is the live-trading equivalent
    of a lookahead bug.
    """
    rows, end_time = [], None
    while len(rows) < bars:
        url = f"{BINANCE}?symbol={SYMBOL}&interval={INTERVAL}&limit=1000"
        if end_time:
            url += f"&endTime={end_time}"
        req = urllib.request.Request(url, headers={"User-Agent": "btc-paper-trader/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "tb_base", "tb_quote", "ignore"])
    df = df[["open_time", "open", "high", "low", "close", "volume", "trades"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(float)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    # Drop the candle still in progress.
    now = datetime.now(timezone.utc)
    df = df[df["open_time"] + BAR <= now].reset_index(drop=True)
    return df.tail(bars).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_gap"] = False
    feats = F.add_indicators(df)
    out = pd.concat([df[["open_time", "open", "high", "low", "close"]], feats], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


# -------------------------------------------------------------------- model ---

def load_production_model(db: DB, dry_run: bool = False):
    """The model row marked `production`, plus its weights and confidence gates."""
    rows = db.select("model_versions", "select=*&status=eq.production", limit=1)
    if not rows:
        sys.exit("no production model. Run: python ml/promote_model.py")
    row = rows[0]

    path = row["metrics"].get("artifact")
    if not path or not os.path.exists(path):
        sys.exit(f"production model {row['version']} points at a missing file: {path!r}")

    # weights_only=True: checkpoints are tensors and plain types only, so
    # loading one cannot execute code even if the file were tampered with.
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMClassifier(len(bundle["feat_cols"]), bundle["hidden"], bundle["layers"])
    model.load_state_dict(bundle["state"])
    model.eval()
    return {
        "version": row["version"],
        "gates": row["metrics"].get("gates") or {},
        "gross_edge": row["metrics"].get("gross_edge"),
        "model": model, "mu": bundle["mu"].numpy(), "sd": bundle["sd"].numpy(),
        "feat_cols": bundle["feat_cols"],
    }


@torch.no_grad()
def predict(mb, feats: pd.DataFrame, upto: int):
    """Predict for the candle at index `upto`, using the SEQ_LEN bars ending there."""
    window = feats[mb["feat_cols"]].to_numpy(np.float32)[upto - SEQ_LEN + 1: upto + 1]
    x = np.clip((window - mb["mu"]) / mb["sd"], -10, 10)
    logits = mb["model"](torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
    probs = torch.softmax(logits, dim=1).numpy()[0]
    return int(probs.argmax()), float(probs.max()), probs


# ------------------------------------------------------------------ account ---

def portfolio_state(db: DB, user: dict, price: float, now: datetime):
    """Derive cash, equity and daily PnL from the position ledger.

    Nothing is stored as a running balance. A mutable balance column drifts the
    moment any write is retried or replayed; recomputing from positions means
    the numbers cannot disagree with the trades that produced them.
    """
    uid = user["user_id"]
    closed = db.select("positions", f"select=pnl,exit_time&user_id=eq.{uid}&status=eq.CLOSED")
    open_rows = db.select("positions", f"select=*&user_id=eq.{uid}&status=eq.OPEN")

    realised = sum(float(p["pnl"] or 0) for p in closed)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    realised_today = sum(float(p["pnl"] or 0) for p in closed
                         if p["exit_time"] and pd.Timestamp(p["exit_time"]) >= day_start)

    positions = [_to_position(r) for r in open_rows]
    open_notional = sum(p.notional for p in positions)
    capital = float(user["virtual_capital"])
    cash = capital + realised - open_notional
    equity = PE.equity(cash, positions, price)

    return RE.PortfolioState(
        cash=max(cash, 0.0), equity=equity, open_positions=len(positions),
        open_notional=open_notional, daily_pnl=realised_today,
        day_start_equity=capital + realised - realised_today,
    ), positions, open_rows


def _to_position(row: dict) -> PE.Position:
    return PE.Position(
        side=row["side"], qty=float(row["qty"]), entry_price=float(row["entry_price"]),
        entry_time=pd.Timestamp(row["entry_time"]).to_pydatetime(),
        stop_loss=float(row["stop_loss"]), take_profit=float(row["take_profit"]),
        expires_at=pd.Timestamp(row["entry_time"]).to_pydatetime() + HORIZON * BAR,
        fees=float(row["fees"] or 0), strategy=row.get("strategy"),
        decision_id=row.get("decision_id"),
    )


# --------------------------------------------------------------------- loop ---

def process_candle(db, mb, feats, idx, users, dry_run=False, verbose=True):
    row = feats.iloc[idx]
    candle_time = row["open_time"].to_pydatetime()
    price = float(row["close"])
    candle = PE.Candle(candle_time, float(row["open"]), float(row["high"]),
                       float(row["low"]), price)

    # 1. Advance open positions FIRST. A stop hit by this candle must be booked
    #    before the same candle is allowed to open anything new.
    exits = 0
    for user in users:
        _, positions, open_rows = portfolio_state(db, user, price, candle_time)
        for pos, dbrow in zip(positions, open_rows):
            hit = PE.check_exit(pos, candle)
            if not hit:
                continue
            PE.close_position(pos, hit[0], candle_time, hit[1])
            exits += 1
            if not dry_run:
                db.update("positions", f"id=eq.{dbrow['id']}", {
                    "status": "CLOSED", "exit_price": round(pos.exit_price, 2),
                    "exit_time": candle_time.isoformat(), "exit_reason": pos.exit_reason,
                    "fees": round(pos.fees, 4), "pnl": round(pos.pnl, 4),
                    "pnl_pct": pos.pnl_pct,
                })

    # 2. Model prediction and regime.
    cls, conf, probs = predict(mb, feats, idx)
    prediction = LABELS[cls]
    reg = R.from_row(row)

    # 3. Jev, only if some active user's gate could let this through. Calling it
    #    on a candle every user will reject spends a rate-limited request to
    #    label a decision that is already HOLD.
    gates = mb["gates"]
    loosest = min((RE.resolve_min_confidence(u["risk_profile"], gates) for u in users),
                  default=1.0)
    tradeable = prediction != "NEUTRAL" and conf >= loosest

    if tradeable:
        jev = jev_client.decide(prediction, conf, reg, price,
                                atr_pct=float(row["atr_pct"]),
                                vwap_dist=float(row["vwap_dist"]),
                                bb_pctb=float(row["bb_pctb"]))
    else:
        strat, _ = jev_client.fallback_strategy(prediction, conf, reg)
        jev = {"strategy": strat, "decided_by": "fallback", "jev_confidence": None,
               "raw": None, "error": "below every user's confidence gate",
               "action": "HOLD" if strat == "no_trade" else
                         ("BUY" if prediction == "UP" else "SELL")}

    decision = {
        "candle_time": candle_time.isoformat(), "price": round(price, 2),
        "prediction": prediction, "confidence": round(conf, 6),
        "prob_down": float(probs[0]), "prob_neutral": float(probs[1]),
        "prob_up": float(probs[2]),
        "regime": reg.name, "atr_pct": float(row["atr_pct"]),
        "strategy": jev["strategy"], "action": jev["action"],
        "decided_by": jev["decided_by"], "jev_confidence": jev["jev_confidence"],
        "jev_raw": jev["raw"], "model_version": mb["version"],
    }

    decision_id = None
    if not dry_run:
        written = db.insert("decisions", decision, upsert_on="candle_time")
        decision_id = written[0]["id"] if written else None

    if verbose:
        gate_note = "" if tradeable else f"  [below gate {loosest:.3f}]"
        print(f"{candle_time:%Y-%m-%d %H:%M}  ${price:>10,.0f}  {prediction:<7} "
              f"{conf:.3f}  {reg.name:<15} {jev['strategy']:<16} {jev['action']:<5} "
              f"[{jev['decided_by']}]{gate_note}" + (f"  {exits} exit(s)" if exits else ""))

    # 4. Per-user risk and entry. Only the sizing is per user; the prediction,
    #    the regime and the Jev call above were computed once.
    opened = 0
    if jev["action"] in ("BUY", "SELL"):
        side = "LONG" if jev["action"] == "BUY" else "SHORT"
        # Entry is the NEXT candle's open, so a decision on the final candle has
        # nothing to fill against yet and waits for the next run.
        nxt = feats.iloc[idx + 1] if idx + 1 < len(feats) else None

        for user in users:
            state, _, _ = portfolio_state(db, user, price, candle_time)
            verdict = RE.approve(
                RE.Proposal(side, conf, price, float(row["atr_pct"])), state,
                user["risk_profile"], float(user["virtual_capital"]),
                float(user["trading_allocation"]), float(user["daily_profit_target"]),
                float(user["max_daily_loss"]), model_gates=gates)

            if not dry_run:
                db.insert("risk_events", {
                    "user_id": user["user_id"], "decision_id": decision_id,
                    "rule": verdict.rule, "detail": verdict.detail[:500],
                    "approved": verdict.approved})

            if not verdict.approved or nxt is None:
                continue

            entry_candle = PE.Candle(nxt["open_time"].to_pydatetime(), float(nxt["open"]),
                                     float(nxt["high"]), float(nxt["low"]), float(nxt["close"]))
            pos = PE.open_position(side, verdict.qty, entry_candle,
                                   verdict.stop_loss, verdict.take_profit,
                                   strategy=jev["strategy"], decision_id=decision_id)
            opened += 1
            if not dry_run:
                db.insert("positions", {
                    "user_id": user["user_id"], "decision_id": decision_id,
                    "side": pos.side, "status": "OPEN", "strategy": pos.strategy,
                    "qty": round(pos.qty, 8), "entry_price": round(pos.entry_price, 2),
                    "entry_time": pos.entry_time.isoformat(),
                    "stop_loss": round(pos.stop_loss, 2),
                    "take_profit": round(pos.take_profit, 2),
                    "fees": round(pos.fees, 4)})

    # 5. Equity snapshot per user, for the dashboard's curve.
    if not dry_run:
        snaps = []
        for user in users:
            state, _, _ = portfolio_state(db, user, price, candle_time)
            snaps.append({"user_id": user["user_id"], "candle_time": candle_time.isoformat(),
                          "equity": round(state.equity, 2), "cash": round(state.cash, 2),
                          "open_pnl": round(state.equity - state.cash - state.open_notional, 4)})
        if snaps:
            db.insert("equity_snapshots", snaps, upsert_on="user_id,candle_time")

    return {"exits": exits, "opened": opened, "decided_by": jev["decided_by"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute but write nothing")
    ap.add_argument("--once", action="store_true", help="only the latest closed candle")
    ap.add_argument("--max-backfill", type=int, default=MAX_BACKFILL)
    args = ap.parse_args()

    load_env()
    db = DB()

    users = db.select("profiles", "select=*&active=eq.true")
    if not users:
        print("no active users; writing decisions only")

    mb = load_production_model(db, args.dry_run)
    print(f"model {mb['version']}  horizon {horizon_label()}  "
          f"gates {mb['gates'] or 'none'}  users {len(users)}\n")

    raw = fetch_recent()
    feats = build_features(raw)
    print(f"{len(feats)} candles, latest {feats['open_time'].iloc[-1]:%Y-%m-%d %H:%M} UTC\n")

    last = db.select("decisions", "select=candle_time&order=candle_time.desc", limit=1)
    last_time = pd.Timestamp(last[0]["candle_time"]) if last else None

    # The final row is the entry candle for the second-to-last decision, so the
    # newest candle we can DECIDE on is len-2.
    end = len(feats) - 1
    if args.once or last_time is None:
        start = end - 1
    else:
        newer = feats.index[feats["open_time"] > last_time]
        start = int(newer[0]) if len(newer) else end
        if end - start > args.max_backfill:
            print(f"backfill capped at {args.max_backfill} candles "
                  f"({end - start} pending)")
            start = end - args.max_backfill

    if start >= end:
        print("up to date, nothing to process")
        return

    totals = {"exits": 0, "opened": 0, "jev": 0, "fallback": 0}
    for i in range(max(start, SEQ_LEN), end):
        r = process_candle(db, mb, feats, i, users, dry_run=args.dry_run)
        totals["exits"] += r["exits"]
        totals["opened"] += r["opened"]
        totals[r["decided_by"]] += 1

    print(f"\n{end - start} candle(s): {totals['opened']} opened, {totals['exits']} closed, "
          f"jev {totals['jev']} / fallback {totals['fallback']}"
          + ("  [DRY RUN, nothing written]" if args.dry_run else ""))


if __name__ == "__main__":
    main()
