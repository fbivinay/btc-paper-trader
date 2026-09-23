"""Download BTC/USDT 5m klines from data.binance.vision into one parquet file.

Monthly ZIP dumps, not the REST API: ~60 requests instead of ~1000, no API key,
and not geo-blocked. Run again any time -- already-downloaded months are skipped.
"""

import io
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
YEARS = 5
BASE = "https://data.binance.vision/data/spot/monthly/klines"
RAW = Path(os.environ.get("BTC_DATA_DIR",
                          Path(__file__).resolve().parent.parent / "data")) / "raw"
OUT = RAW.parent / f"{SYMBOL}_{INTERVAL}.parquet"

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def months_back(n_years: int) -> list[str]:
    """Month keys YYYY-MM, oldest first, ending at last complete month."""
    today = date.today()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    keys = []
    for _ in range(n_years * 12):
        keys.append(f"{y:04d}-{m:02d}")
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return sorted(keys)


def fetch_month(key: str) -> pd.DataFrame | None:
    cache = RAW / f"{SYMBOL}-{INTERVAL}-{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{BASE}/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{key}.zip"
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        print(f"  {key}: not published, skipping", flush=True)
        return None
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            # Binance added a header row to newer dumps; sniff the first byte.
            head = f.read(1)
    has_header = not head.isdigit()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            header=0 if has_header else None,
            names=COLS,
            skiprows=1 if has_header else 0,
        )

    df = df[["open_time", "open", "high", "low", "close", "volume", "trades"]]
    # open_time is ms before ~2025-01 and microseconds after. Normalise by magnitude.
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume"]})
    df.to_parquet(cache, index=False)
    print(f"  {key}: {len(df):>6} candles", flush=True)
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    keys = months_back(YEARS)
    print(f"Fetching {SYMBOL} {INTERVAL}: {keys[0]} .. {keys[-1]} ({len(keys)} months)")

    frames = [df for k in keys if (df := fetch_month(k)) is not None]
    if not frames:
        sys.exit("no data downloaded")

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    # Gaps matter: a missing candle silently shifts every windowed feature.
    expected = pd.date_range(full["open_time"].iloc[0], full["open_time"].iloc[-1], freq="5min")
    missing = len(expected) - len(full)

    full.to_parquet(OUT, index=False)
    print(f"\n{len(full):,} candles -> {OUT}")
    print(f"range  {full['open_time'].iloc[0]}  ..  {full['open_time'].iloc[-1]}")
    print(f"gaps   {missing:,} missing candles ({missing / len(expected):.4%})")


if __name__ == "__main__":
    main()
