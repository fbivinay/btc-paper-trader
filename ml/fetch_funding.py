"""Download BTC/USDT perpetual funding-rate history from Binance futures.

Funding is paid every 8 hours. It is positive when longs pay shorts, which
happens when the perpetual trades above spot -- so it is a direct read on
leveraged positioning, information that price and volume do not contain.

Unlike open interest and the long/short ratio, Binance keeps the full funding
history (back to 2019-09), which is what makes this the one positioning feature
that can actually be trained on.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
OUT = DATA / "BTCUSDT_funding.parquet"
SYMBOL = "BTCUSDT"

# Futures endpoints have no data.binance.vision mirror, so try the regional
# hosts in turn. fapi.binance.com is geo-blocked from US IPs, the same problem
# that broke the live loop on GitHub runners.
HOSTS = ["https://fapi.binance.com", "https://fapi-gcp.binance.com"]
START = 1568073600000          # 2019-09-10, the first funding event


def get(path: str, host: str):
    req = urllib.request.Request(host + path, headers={"User-Agent": "btc-paper-trader/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


def pick_host() -> str:
    for h in HOSTS:
        try:
            get(f"/fapi/v1/fundingRate?symbol={SYMBOL}&limit=1", h)
            return h
        except Exception as e:
            print(f"  {h}: {type(e).__name__} {e}")
    sys.exit("no reachable Binance futures endpoint")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    host = pick_host()
    print(f"using {host}")

    rows, start = [], START
    while True:
        batch = get(f"/fapi/v1/fundingRate?symbol={SYMBOL}&startTime={start}&limit=1000", host)
        if not batch:
            break
        rows += batch
        last = batch[-1]["fundingTime"]
        if last <= start or len(batch) < 1000:
            break
        start = last + 1
        time.sleep(0.15)          # stay well inside the weight limit

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = (df[["funding_time", "funding_rate"]]
          .drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True))

    df.to_parquet(OUT, index=False)
    print(f"\n{len(df):,} funding events -> {OUT}")
    print(f"range {df['funding_time'].iloc[0]:%Y-%m-%d} .. {df['funding_time'].iloc[-1]:%Y-%m-%d}")
    print(f"rate  mean {df['funding_rate'].mean():+.6f}  "
          f"min {df['funding_rate'].min():+.6f}  max {df['funding_rate'].max():+.6f}")
    print(f"positive {(df['funding_rate'] > 0).mean():.1%} of the time "
          "(longs paying shorts)")


if __name__ == "__main__":
    main()
