"""Ship ml/*.py to Kaggle as a dataset, then push and optionally run the kernel.

Two steps because a Kaggle kernel cannot carry arbitrary local modules: the code
rides along as a (tiny, private) dataset that the kernel mounts read-only.

    python ml/push_kaggle.py            # push code + kernel
    python ml/push_kaggle.py --run      # ...and trigger a run
    python ml/push_kaggle.py --status   # check the latest run
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_SRC = ROOT / "ml"
CODE_OUT = ROOT / "kaggle" / "code"
KERNEL = ROOT / "kaggle"
MODULES = ["fetch_history.py", "features.py", "metrics.py", "train.py"]


def sh(cmd: list[str], check: bool = True) -> str:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out, flush=True)
    if check and r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}")
    return out


def kernel_id() -> str:
    return json.loads((KERNEL / "kernel-metadata.json").read_text())["id"]


def push_code(message: str) -> None:
    CODE_OUT.mkdir(parents=True, exist_ok=True)
    for m in MODULES:
        shutil.copy2(CODE_SRC / m, CODE_OUT / m)
    print(f"staged {len(MODULES)} modules -> {CODE_OUT}")

    # `datasets version` fails until the dataset exists, and the API reports that
    # as 403 rather than 404. Fall back to create on any failure instead of
    # pattern-matching the error string.
    r = subprocess.run(["kaggle", "datasets", "version", "-p", str(CODE_OUT),
                        "-m", message, "--dir-mode", "zip"],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip(), flush=True)
    if r.returncode != 0:
        print("version failed, creating dataset", flush=True)
        sh(["kaggle", "datasets", "create", "-p", str(CODE_OUT), "--dir-mode", "zip"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="trigger a kernel run after pushing")
    ap.add_argument("--status", action="store_true", help="only check the latest run")
    ap.add_argument("-m", "--message", default="update", help="dataset version message")
    args = ap.parse_args()

    if args.status:
        sh(["kaggle", "kernels", "status", kernel_id()])
        return

    push_code(args.message)
    # Kaggle indexes a new dataset version asynchronously. A kernel pushed in the
    # same second can mount the previous version.
    print("\nnote: if the kernel logs show stale code, re-run -- dataset "
          "indexing lags the push by a few seconds.\n")
    sh(["kaggle", "kernels", "push", "-p", str(KERNEL)])

    if args.run:
        sh(["kaggle", "kernels", "status", kernel_id()])
        print(f"\nwatch: https://www.kaggle.com/code/{kernel_id()}")


if __name__ == "__main__":
    main()
