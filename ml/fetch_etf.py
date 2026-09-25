"""Download daily ETF prices and the 3-month T-bill rate from the Yahoo chart API.

  IBIT  iShares Bitcoin Trust, the largest spot Bitcoin ETF (trading since 2024-01-11)
  GLD   SPDR Gold Shares; its dates are also the US trading calendar for 2017 onward
  ^IRX  13-week T-bill yield: what idle cash earns parked in a T-bill ETF

Free, no key.

    python ml/fetch_etf.py
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
START = 1498867200             # 2017-07-01
TICKERS = ("IBIT", "GLD")


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def etf(ticker: str) -> pd.DataFrame:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={START}&period2={int(time.time())}&interval=1d")
    j = json.loads(get(url))["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    date = (pd.to_datetime(j["timestamp"], unit="s", utc=True)
            .tz_convert("America/New_York").normalize().tz_localize(None))
    return pd.DataFrame({"date": date, "open": q["open"], "close": q["close"]}).dropna()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for tk in TICKERS:
        d = etf(tk)
        d.to_parquet(DATA / f"etf_{tk}.parquet", index=False)
        print(f"{tk}: {len(d)} days, {d.date.iloc[0]:%Y-%m-%d} .. {d.date.iloc[-1]:%Y-%m-%d}")
    t = etf("^IRX")
    t = pd.DataFrame({"date": t["date"], "rate": t["close"] / 100})
    t.to_parquet(DATA / "tbill.parquet", index=False)
    print(f"T-bill: {len(t)} days, last {t.rate.iloc[-1]:.2%} on {t.date.iloc[-1]:%Y-%m-%d}")


if __name__ == "__main__":
    main()
