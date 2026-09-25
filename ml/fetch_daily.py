"""Download BTC/USDT daily candles from the start of Binance trading (Aug 2017).

Low-turnover strategies trade a handful of times a year, so they can only be
judged across several full bull and bear cycles. The 5-minute archive starts in
2021 and covers barely one; this gives 2017 onward: the 2017 bubble, the 2018
crash, the 2020-21 bull, the 2022 crash, the 2023-25 recovery and the 2025-26
decline.

Uses data-api.binance.vision, the public market-data mirror, which is not
geo-restricted.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
HOST = "https://data-api.binance.vision"
START_MS = 1502928000000        # 2017-08-17, first BTCUSDT candle on Binance


def fetch(interval: str) -> pd.DataFrame:
    rows, start = [], START_MS
    while True:
        url = (f"{HOST}/api/v3/klines?symbol=BTCUSDT&interval={interval}"
               f"&startTime={start}&limit=1000")
        req = urllib.request.Request(url, headers={"User-Agent": "btc-paper-trader/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        start = batch[-1][0] + 1
        time.sleep(0.2)

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df[["open_time", "open", "high", "low", "close", "volume",
             "quote_volume", "trades", "taker_buy_base"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    # Drop the candle still forming: acting on it would be lookahead.
    now = pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(interval.replace("d", "D"))
    df = df[df["open_time"] + step <= now]
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for interval in ("1d", "4h"):
        df = fetch(interval)
        out = DATA / f"BTCUSDT_{interval}.parquet"
        df.to_parquet(out, index=False)
        print(f"{interval}: {len(df):,} candles  "
              f"{df['open_time'].iloc[0]:%Y-%m-%d} .. {df['open_time'].iloc[-1]:%Y-%m-%d}  -> {out.name}")


if __name__ == "__main__":
    main()
