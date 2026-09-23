"""Kaggle GPU kernel: rebuild data from source, sweep thresholds, run walk-forward.

Re-downloads and re-derives on every run rather than reading an uploaded
parquet. The weekly retrain needs fresh candles anyway, so one code path serves
both the sweep and the retrain, removing a class of "the uploaded dataset was
stale" bug.

Sweeps the label threshold as a multiple of the round-trip cost. At futures
maker pricing a 1x threshold labels 88% of candles tradeable and a 3x threshold
labels 66%; which trains a better model is empirical, not obvious.
"""

import os
import subprocess
import sys

CODE = "/kaggle/input/btc-trading-code"
WORK = "/kaggle/working"

# Rebuilt data goes to /tmp, not /kaggle/working: it is 165MB, reproducible, and
# anything left in the working directory has to be downloaded again on each pull.
os.environ["BTC_DATA_DIR"] = "/tmp/btcdata"
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

for step in ("fetch_history", "fetch_funding", "fetch_sentiment"):
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, f"{CODE}/{step}.py"], check=True)

for mult in ("1", "3"):
    env = dict(os.environ, BTC_THRESHOLD_MULT=mult)
    print(f"\n===== features, threshold {mult}x cost =====", flush=True)
    subprocess.run([sys.executable, f"{CODE}/features.py"], check=True, env=env)
    print(f"\n===== train, threshold {mult}x cost =====", flush=True)
    subprocess.run([sys.executable, f"{CODE}/train.py",
                    "--windows", "180", "365", "730",
                    "--folds", "4", "--epochs", "15", "--stride", "3"],
                   check=True, env=env)

print("\nartifacts:", flush=True)
for f in sorted(os.listdir(os.environ["BTC_ARTIFACT_DIR"])):
    size = os.path.getsize(f"{os.environ['BTC_ARTIFACT_DIR']}/{f}") / 1e6
    print(f"  {f}  {size:.1f} MB", flush=True)
