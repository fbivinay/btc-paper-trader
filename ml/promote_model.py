"""Register a trained model and decide whether it may go to production.

    python ml/promote_model.py --register artifacts/artifacts/walkforward_h48.json
    python ml/promote_model.py --list
    python ml/promote_model.py --force-promote <version>

A model finishing training is not a reason to deploy it. This is the gate: a
candidate is promoted only if it clears every criterion below, and otherwise the
current production model keeps running.

The criteria are deliberately economic, not statistical:

  GROSS EDGE > COST   The one that matters. A model whose average gross edge per
                      trade does not exceed the round trip fee loses money no
                      matter how accurate it is. Accuracy is not checked at all;
                      68% of labels were NEUTRAL at the 30m horizon, so a model
                      that never trades could score 68% and earn nothing.
  CONSISTENCY         Profitable on a majority of walk-forward folds. A model
                      that wins one fold and loses three got lucky once.
  DIRECTIONAL BALANCE Predicts both directions. A model that only ever says UP
                      has learned the drift, not the signal.
  BEATS INCUMBENT     Better gross edge than the model already in production.

As of the 4h sweep NO candidate clears the first gate: best gross edge is
+0.035% against a 0.25% cost. That is the correct outcome and the gate says so
rather than promoting the least-bad option.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import COST, HORIZON, horizon_label
from db import DB, load_env

ROOT = Path(__file__).resolve().parent.parent
MIN_FOLD_MAJORITY = 0.5      # more than half the folds must be profitable
MIN_BALANCE = 0.25           # weaker direction at least 25% as frequent as the stronger
BASE = "top100pct"


def gates_from_folds(folds: list) -> dict:
    """Median confidence cutoff per percentile across folds.

    Median, not mean: the top-9% cutoff ranged 0.43-0.95 across folds, so a mean
    is dragged by whichever quarter happened to be strange.
    """
    out = {}
    for name, pct in (("top25pct", 0.25), ("top9pct", 0.10), ("top50pct", 0.50)):
        vals = [f["gates"][name] for f in folds if name in f.get("gates", {})]
        if vals:
            out[str(pct)] = float(np.median(vals))
    return out


def best_gate(folds: list) -> tuple[str, float]:
    """The gate with the highest mean gross edge -- the operating point."""
    names = [k for k in folds[0] if k.startswith("top")]
    scored = {n: float(np.mean([f[n]["avg_pnl"] for f in folds])) + COST for n in names}
    name = max(scored, key=scored.get)
    return name, scored[name]


def evaluate(entry: dict) -> dict:
    """Score one training window against the promotion criteria."""
    folds = entry["folds"]
    gate_name, gross_edge = best_gate(folds)
    profitable = sum(f[BASE]["total_return"] > 0 for f in folds)
    balance = entry["mean"].get("dir_balance", 0.0)

    checks = {
        "gross_edge_beats_cost": gross_edge > COST,
        "majority_of_folds_profitable": profitable > len(folds) * MIN_FOLD_MAJORITY,
        "predicts_both_directions": balance >= MIN_BALANCE,
    }
    return {
        "window_days": entry["window_days"], "n_folds": len(folds),
        "gate": gate_name, "gross_edge": gross_edge,
        "folds_profitable": profitable, "dir_balance": balance,
        "sharpe": entry["mean"]["sharpe"], "checks": checks,
        "passes": all(checks.values()),
        "gates": gates_from_folds(folds),
    }


def register(db: DB, results_path: Path, dry_run: bool = False) -> None:
    summary = json.loads(results_path.read_text())
    if not summary:
        sys.exit(f"{results_path} is empty")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    incumbent = db.select("model_versions", "select=*&status=eq.production", limit=1)
    incumbent_edge = (incumbent[0]["metrics"].get("gross_edge", -1)
                      if incumbent else -float("inf"))

    print(f"horizon {horizon_label()}   cost to beat {COST:.3%}")
    if incumbent:
        print(f"incumbent {incumbent[0]['version']} gross edge {incumbent_edge:+.4%}\n")
    else:
        print("no incumbent model\n")

    scored = [evaluate(e) for e in summary]
    best = max(scored, key=lambda s: s["gross_edge"])

    print(f"{'window':>8} {'gate':>10} {'gross edge':>12} {'vs cost':>9} "
          f"{'folds':>7} {'balance':>8}  verdict")
    for s in scored:
        beats = "PASS" if s["gross_edge"] > COST else "FAIL"
        print(f"{s['window_days']:>7}d {s['gate']:>10} {s['gross_edge']:>+12.4%} "
              f"{beats:>9} {s['folds_profitable']:>3}/{s['n_folds']:<3} "
              f"{s['dir_balance']:>8.2f}  "
              f"{'PROMOTE' if s['passes'] else 'reject'}")

    print(f"\nbest candidate: {best['window_days']}d, gross edge {best['gross_edge']:+.4%}")
    for name, ok in best["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    beats_incumbent = best["gross_edge"] > incumbent_edge
    print(f"  {'PASS' if beats_incumbent else 'FAIL'}  beats_incumbent")

    promote = best["passes"] and beats_incumbent
    version = f"lstm-h{HORIZON}-w{best['window_days']}-{stamp}"
    artifact = str(ROOT / "artifacts" / "artifacts" /
                   f"model_h{HORIZON}_w{best['window_days']}_"
                   f"{summary[0]['folds'][-1]['test_to']}.pt")

    row = {
        "version": version,
        "window_days": best["window_days"],
        "trained_until": summary[0]["folds"][-1]["test_to"],
        "status": "candidate" if not promote else "production",
        "metrics": {
            "gross_edge": best["gross_edge"], "cost": COST, "gate": best["gate"],
            "gates": best["gates"], "sharpe": best["sharpe"],
            "folds_profitable": best["folds_profitable"], "n_folds": best["n_folds"],
            "dir_balance": best["dir_balance"], "checks": best["checks"],
            "artifact": artifact, "horizon": HORIZON,
        },
        "notes": ("promoted by gate" if promote else
                  "rejected: " + ", ".join(k for k, v in best["checks"].items() if not v)
                  + ("" if beats_incumbent else ", does not beat incumbent")),
    }

    print(f"\n{'PROMOTING' if promote else 'REGISTERING AS CANDIDATE'}: {version}")
    print(f"  {row['notes']}")
    if not promote and incumbent:
        print(f"  {incumbent[0]['version']} stays in production")
    elif not promote:
        print("  no production model -- the live loop will refuse to run")

    if dry_run:
        print("\n[dry run, nothing written]")
        return

    if promote and incumbent:
        # The partial unique index allows only one production row, so the old one
        # must step down before the new one is written.
        db.update("model_versions", f"version=eq.{incumbent[0]['version']}",
                  {"status": "rejected", "notes": f"superseded by {version}"})
    if promote:
        row["promoted_at"] = datetime.now(timezone.utc).isoformat()
    db.insert("model_versions", row, upsert_on="version")
    print("\nwritten to model_versions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", type=Path, help="walkforward json to evaluate")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force-promote", metavar="VERSION",
                    help="promote a candidate that failed the gate (recorded in notes)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    db = DB()

    if args.list:
        rows = db.select("model_versions", "select=*&order=created_at.desc")
        if not rows:
            print("no models registered")
            return
        print(f"{'version':<34} {'status':<11} {'gross edge':>11} {'vs cost':>9}")
        for r in rows:
            e = r["metrics"].get("gross_edge")
            print(f"{r['version']:<34} {r['status']:<11} "
                  f"{e:>+11.4%} {'beats' if e and e > COST else 'below':>9}"
                  if e is not None else f"{r['version']:<34} {r['status']:<11}")
        return

    if args.force_promote:
        rows = db.select("model_versions", f"select=*&version=eq.{args.force_promote}")
        if not rows:
            sys.exit(f"no such version: {args.force_promote}")
        current = db.select("model_versions", "select=version&status=eq.production", limit=1)
        if current:
            db.update("model_versions", f"version=eq.{current[0]['version']}",
                      {"status": "rejected", "notes": "manually superseded"})
        db.update("model_versions", f"version=eq.{args.force_promote}", {
            "status": "production",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "notes": "MANUALLY FORCED past the validation gate"})
        print(f"forced {args.force_promote} to production (recorded in notes)")
        return

    if args.register:
        register(db, args.register, args.dry_run)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
