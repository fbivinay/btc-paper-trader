"""Kaggle GPU kernel: rebuild data from source, train on triple-barrier labels.

Labels are what a real trade would have done -- did +3R arrive before -R --
rather than where price happened to be at a fixed horizon. That is the question
the trading rule asks, so it is the question the model is trained on.

R = 0.5% with a 72h limit is chosen from data: it is the setting where 99.7% of
trades reach a barrier, making the payoff a true 3:1. Break-even win rate is
29.2% including Binance futures fees; a coin flip gets 25%.
"""

import os
import subprocess
import sys

CODE = "/kaggle/input/btc-trading-code"
WORK = "/kaggle/working"

os.environ["BTC_DATA_DIR"] = "/tmp/btcdata"
os.environ["BTC_ARTIFACT_DIR"] = f"{WORK}/artifacts"
os.environ["BTC_LABEL_MODE"] = "barrier"
os.makedirs(os.environ["BTC_DATA_DIR"], exist_ok=True)
os.makedirs(os.environ["BTC_ARTIFACT_DIR"], exist_ok=True)
sys.path.insert(0, CODE)

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("WARNING: no GPU. Enable it in kernel settings.", flush=True)

for step in ("fetch_history", "fetch_funding", "fetch_sentiment"):
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, f"{CODE}/{step}.py"], check=True)

print("\n===== features (triple barrier) =====", flush=True)
subprocess.run([sys.executable, f"{CODE}/features.py"], check=True)

print("\n===== train =====", flush=True)
subprocess.run([sys.executable, f"{CODE}/train.py",
                "--windows", "180", "365", "730",
                "--folds", "4", "--epochs", "15", "--stride", "3"], check=True)

print("\nartifacts:", flush=True)
for f in sorted(os.listdir(os.environ["BTC_ARTIFACT_DIR"])):
    size = os.path.getsize(f"{os.environ['BTC_ARTIFACT_DIR']}/{f}") / 1e6
    print(f"  {f}  {size:.1f} MB", flush=True)
