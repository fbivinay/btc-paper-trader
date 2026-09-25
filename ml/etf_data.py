"""Real daily market data for the ETF model, fetched fresh from the Yahoo chart API.

  IBIT   iShares Bitcoin Trust, the spot Bitcoin ETF traded (since 2024-01-11)
  GLD    SPDR Gold Shares, the gold ETF traded; its dates are the US trading calendar
  ^IRX   13-week T-bill yield: what idle cash earns in a T-bill ETF
  INR=X  rupees per dollar, for showing amounts in rupees

The slowest signal needs a year of history, and IBIT has none before its launch.
Before 2024-01-11 Bitcoin's own real price stands in, taken at US market hours
(12:00 UTC ~ the open, 20:00 UTC ~ the close) and scaled to IBIT's price at the
switch-over. That history never changes, so it lives in btc_before_ibit.csv.

    python ml/etf_data.py      # print the latest prices
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HISTORY = Path(__file__).with_name("btc_before_ibit.csv")
IBIT_START = pd.Timestamp("2024-01-11")
START = 1498867200            # 2017-07-01


def yahoo(ticker: str) -> pd.DataFrame:
    """Daily open/close indexed by US trading date, completed sessions only."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={START}&period2={int(pd.Timestamp.now().timestamp())}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):                 # a free API hiccups; three tries, then fail loudly
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                j = json.loads(r.read())["chart"]["result"][0]
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    q = j["indicators"]["quote"][0]
    date = (pd.to_datetime(j["timestamp"], unit="s", utc=True)
            .tz_convert("America/New_York").normalize().tz_localize(None))
    d = pd.DataFrame({"open": q["open"], "close": q["close"]}, index=date).dropna()
    d = d[~d.index.duplicated(keep="last")]
    # Today's bar keeps changing until the close; a decision must only see finished days.
    ny = pd.Timestamp.now(tz="America/New_York")
    if ny.hour * 60 + ny.minute < 16 * 60 + 30:
        d = d[d.index < ny.normalize().tz_localize(None)]
    return d


def load():
    """-> (btc, gold, tbill, usdinr).

    btc and gold: open/close per US trading day. btc["real"] is True on actual IBIT
    prices and False on the pre-launch Bitcoin stand-in.
    """
    ibit, gold = yahoo("IBIT"), yahoo("GLD")
    tbill = yahoo("^IRX")["close"] / 100
    usdinr = yahoo("INR=X")["close"]
    hist = pd.read_csv(HISTORY, index_col="date", parse_dates=True)
    scale = ibit.loc[IBIT_START, "close"] / hist.loc[IBIT_START, "close"]
    pre = hist[hist.index < IBIT_START] * scale
    btc = pd.concat([pre.assign(real=False), ibit.assign(real=True)])
    days = gold.index.intersection(btc.index)
    return btc.loc[days], gold.loc[days], tbill, usdinr


def check(btc: pd.DataFrame, gold: pd.DataFrame) -> None:
    """Refuse to trade on data that looks broken: stale, missing or absurd."""
    last = btc.index[-1]
    assert gold.index[-1] == last, "IBIT and GLD disagree on the last trading day"
    assert (pd.Timestamp.now() - last).days <= 5, f"data is stale: last bar {last:%Y-%m-%d}"
    for name, d in (("IBIT", btc), ("GLD", gold)):
        assert d[["open", "close"]].iloc[-300:].notna().all().all(), f"{name} has gaps"
        move = d["close"].pct_change().iloc[-1]
        assert abs(move) < 0.35, f"{name} moved {move:+.0%} in a day; halting until checked"


if __name__ == "__main__":
    btc, gold, tbill, usdinr = load()
    check(btc, gold)
    print(f"last session {btc.index[-1]:%Y-%m-%d}: IBIT ${btc['close'].iloc[-1]:.2f}  "
          f"GLD ${gold['close'].iloc[-1]:.2f}  T-bill {tbill.iloc[-1]:.2%}  USD/INR {usdinr.iloc[-1]:.2f}")
    print(f"{len(btc)} days from {btc.index[0]:%Y-%m-%d}; real IBIT from {btc.index[btc['real']][0]:%Y-%m-%d}")
