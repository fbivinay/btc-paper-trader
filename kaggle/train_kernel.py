"""Kaggle GPU kernel: rebuild data from source, run the walk-forward sweep.

Deliberately re-downloads and re-derives features on every run rather than
reading an uploaded parquet. The weekly retrain needs fresh candles anyway, so
having one code path for both the sweep and the retrain removes a whole class of
"the uploaded dataset was stale" bug.

Outputs land in /kaggle/working and are collected by ml/pull_kaggle.py.
"""

import os
import subprocess
import sys

CODE = "/kaggle/input/btc-trading-code"
WORK = "/kaggle/working"

os.environ["BTC_DATA_DIR"] = f"{WORK}/data"
os.environ["BTC_ARTIFACT_DIR"] = f"{WORK}/artifacts"
os.makedirs(os.environ["BTC_DATA_DIR"], exist_ok=True)
os.makedirs(os.environ["BTC_ARTIFACT_DIR"], exist_ok=True)
sys.path.insert(0, CODE)

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("WARNING: no GPU. Enable it in kernel settings or this runs ~40x slower.", flush=True)

for step in ("fetch_history", "features"):
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, f"{CODE}/{step}.py"], check=True)

print("\n===== train =====", flush=True)
subprocess.run([sys.executable, f"{CODE}/train.py",
                "--windows", "180", "365", "730",
                "--folds", "4", "--epochs", "15", "--stride", "3"], check=True)

print("\nartifacts:", flush=True)
for f in sorted(os.listdir(os.environ["BTC_ARTIFACT_DIR"])):
    size = os.path.getsize(f"{os.environ['BTC_ARTIFACT_DIR']}/{f}") / 1e6
    print(f"  {f}  {size:.1f} MB", flush=True)
