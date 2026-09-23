"""Download the Crypto Fear & Greed Index.

Free, no key, daily since 2018-02 -- which is what makes it usable where the
other sentiment sources are not. Marketaux's free tier returns three articles
per request with no historical access; news and social sentiment archives are
otherwise paid. This is the one crowd-sentiment series with enough history to
train on.

The index blends volatility, momentum, social volume, dominance and trends into
0-100, where low is fear and high is greed. It is a contrarian indicator by
construction: extremes tend to mark turning points rather than continuations.
"""

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
OUT = DATA / "fear_greed.parquet"
URL = "https://api.alternative.me/fng/?limit=0&format=json"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "btc-paper-trader/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.loads(r.read())["data"]

    df = pd.DataFrame(rows)
    df["fng_time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fng"] = df["value"].astype(float)
    df = (df[["fng_time", "fng"]]
          .drop_duplicates("fng_time").sort_values("fng_time").reset_index(drop=True))

    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} daily values -> {OUT}")
    print(f"range {df['fng_time'].iloc[0]:%Y-%m-%d} .. {df['fng_time'].iloc[-1]:%Y-%m-%d}")
    print(f"value mean {df['fng'].mean():.1f}  min {df['fng'].min():.0f}  max {df['fng'].max():.0f}")


if __name__ == "__main__":
    main()
