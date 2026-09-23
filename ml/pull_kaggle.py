"""Pull kernel output back down: logs, walkforward.json, trained checkpoints.

    python ml/pull_kaggle.py           # artifacts + summary table
    python ml/pull_kaggle.py --log     # kernel log only (for a failed run)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
KERNEL_ID = json.loads((ROOT / "kaggle" / "kernel-metadata.json").read_text())["id"]

BASE = "top100pct"


def sh(cmd: list[str], check: bool = True) -> str:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        print(out)
        sys.exit(f"failed: {' '.join(cmd)}")
    return out


def summarise() -> None:
    # Kaggle nests its output under the working-directory structure, so the real
    # file lands at artifacts/artifacts/walkforward.json while a stale local one
    # may still sit at artifacts/walkforward.json. Take the newest and say which,
    # rather than silently reporting last week's numbers as this run's.
    found = sorted(ART.rglob("walkforward.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        print("no walkforward.json -- run did not reach the end of training")
        return
    path = found[-1]
    print(f"\nreading {path.relative_to(ART.parent)}")
    if len(found) > 1:
        print(f"  ({len(found) - 1} older copy/copies ignored)")
    summary = json.loads(path.read_text())
    if not summary:
        print("walkforward.json is empty")
        return

    print(f"\n{'window':>8} {'folds':>6} {'mean ret':>10} {'sd':>8} {'sharpe':>8} "
          f"{'prec_up':>8} {'prec_dn':>8} {'bal':>6} {'profitable':>11}")
    for s in summary:
        m = s["mean"]
        print(f"{s['window_days']:>7}d {s['n_folds']:>6} {m['total_return']:>+10.2%} "
              f"{m['return_std']:>8.2%} {m['sharpe']:>+8.2f} {m['prec_up']:>8.1%} "
              f"{m['prec_down']:>8.1%} {m.get('dir_balance', 0):>6.2f} "
              f"{m['folds_profitable']:>7}/{s['n_folds']}")

    print("\nper-fold detail (ungated):")
    for s in summary:
        print(f"  window {s['window_days']}d")
        for f in s["folds"]:
            b = f[BASE]
            print(f"    {f['test_from']} .. {f['test_to']}  trades {b['n_trades']:>5}  "
                  f"ret {b['total_return']:>+8.2%}  win {b['win_rate']:>6.1%}  "
                  f"sharpe {b['sharpe']:>+6.2f}  dd {b['max_dd']:>7.2%}")

    # Selection is on consistency, not the best single number. A window that wins
    # one fold and loses three is a window that got lucky once.
    ranked = sorted(summary, key=lambda s: (s["mean"]["folds_profitable"],
                                            s["mean"]["sharpe"]), reverse=True)
    best = ranked[0]
    print(f"\nmost consistent window: {best['window_days']}d "
          f"({best['mean']['folds_profitable']}/{best['n_folds']} folds profitable, "
          f"sharpe {best['mean']['sharpe']:+.2f})")
    if best["mean"]["folds_profitable"] < best["n_folds"]:
        print("NOTE: no window is profitable on every fold. Do not deploy on this alone.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="print the kernel log and stop")
    args = ap.parse_args()

    if args.log:
        ART.mkdir(exist_ok=True)
        sh(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(ART)])
        log = next(iter(ART.glob("*.log")), None)
        if log:
            entries = json.loads(log.read_text())
            for e in entries:
                print(e.get("data", ""), end="")
        return

    ART.mkdir(exist_ok=True)
    print(sh(["kaggle", "kernels", "status", KERNEL_ID]))
    sh(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(ART)])

    files = sorted(ART.glob("*"))
    print(f"\n{len(files)} files in {ART}:")
    for f in files:
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
    summarise()


if __name__ == "__main__":
    main()
