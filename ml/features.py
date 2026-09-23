"""Feature engineering and label generation for BTC/USDT 5m candles.

Two rules drive every choice in this file:

1. Every feature is scale-free. Raw price or raw EMA as an input means a model
   trained at $40k BTC sees garbage at $120k. Everything is a ratio, a z-score,
   or an already-bounded oscillator.
2. Nothing looks forward. Features at candle t use data up to and including the
   close of t. The label uses t+1..t+horizon and nothing else touches it.

Indicators are hand-rolled rather than pandas-ta: pandas-ta 0.3.x does
`from numpy import NaN`, which does not exist in numpy 2.x.

Scaling is deliberately NOT done here. A scaler fitted over the whole file would
leak test-set statistics into training. train.py fits it per walk-forward window.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZON, BARS_PER_DAY, COST as ROUND_TRIP_COST, horizon_label

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
SRC = DATA / "BTCUSDT_5m.parquet"
OUT = DATA / f"BTCUSDT_5m_features_h{HORIZON}.parquet"


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing -- what RSI/ATR/ADX actually use, not a plain EMA."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    # --- trend: expressed as distance from price, not the level itself ---
    ema20, ema50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    f["ema20_dist"] = c / ema20 - 1
    f["ema50_dist"] = c / ema50 - 1
    f["ema_spread"] = ema20 / ema50 - 1

    # --- MACD, normalised by price so it is comparable across regimes ---
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / c
    f["macd_signal"] = signal / c
    f["macd_hist"] = (macd - signal) / c

    # --- RSI (already bounded 0-100, rescaled to -1..1) ---
    delta = c.diff()
    rs = wilder(delta.clip(lower=0), 14) / wilder(-delta.clip(upper=0), 14).replace(0, np.nan)
    f["rsi"] = (100 - 100 / (1 + rs)) / 50 - 1

    # --- Bollinger: %B and bandwidth, both scale-free ---
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    f["bb_pctb"] = (c - mid) / (2 * sd).replace(0, np.nan)
    f["bb_width"] = (4 * sd) / mid

    # --- ATR as a fraction of price ---
    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    f["atr_pct"] = atr / c

    # --- VWAP over a rolling 24h window (crypto has no session boundary) ---
    tp = (h + l + c) / 3
    vwap = (tp * v).rolling(BARS_PER_DAY).sum() / v.rolling(BARS_PER_DAY).sum()
    f["vwap_dist"] = c / vwap - 1

    # --- ADX: the regime classifier's main input ---
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=df.index), 14) / atr
    minus_di = 100 * wilder(pd.Series(minus_dm, index=df.index), 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    f["adx"] = wilder(dx, 14) / 100
    f["di_diff"] = (plus_di - minus_di) / 100

    # --- returns over several lookbacks ---
    for n in (1, 3, 6, 12, 72):
        f[f"ret_{n}"] = c.pct_change(n)

    # --- realised volatility, and where it sits in its own recent history ---
    r1 = c.pct_change()
    f["vol_12"] = r1.rolling(12).std()
    f["vol_72"] = r1.rolling(72).std()
    f["vol_ratio"] = f["vol_12"] / f["vol_72"].replace(0, np.nan)
    f["vol_pctile"] = f["vol_72"].rolling(BARS_PER_DAY * 7).rank(pct=True)

    # --- volume, as a z-score against its own recent history ---
    vol_mean = v.rolling(BARS_PER_DAY).mean()
    f["vol_z"] = (v - vol_mean) / v.rolling(BARS_PER_DAY).std().replace(0, np.nan)
    f["trades_z"] = (df["trades"] - df["trades"].rolling(BARS_PER_DAY).mean()) / df[
        "trades"
    ].rolling(BARS_PER_DAY).std().replace(0, np.nan)

    # --- candle shape ---
    rng = (h - l).replace(0, np.nan)
    f["hl_range"] = rng / c
    f["body"] = (c - df["open"]) / rng
    f["upper_wick"] = (h - np.maximum(c, df["open"])) / rng
    f["lower_wick"] = (np.minimum(c, df["open"]) - l) / rng

    # --- time of day: crypto volume has a real diurnal cycle ---
    mins = df["open_time"].dt.hour * 60 + df["open_time"].dt.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)

    return f


def add_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Label the trade we could actually take.

    Decision is made at the close of t, so the earliest realistic fill is the
    open of t+1. Exit is the close of t+HORIZON. Using close[t] as the entry
    instead would hand the model half a candle of hindsight.
    """
    # ponytail: fixed-horizon label -- reads only the entry and exit candle, so it
    # cannot see a stop-loss or take-profit hit inside the window. The paper engine
    # can, which makes live results diverge from the label. Switch to a triple-barrier
    # label (first of SL / TP / horizon) if that gap shows up in the backtest.
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-HORIZON)
    fwd = exit_ / entry - 1

    label = pd.Series(1, index=df.index, dtype="int8")   # 1 = NEUTRAL
    label[fwd > threshold] = 2                           # 2 = UP
    label[fwd < -threshold] = 0                          # 0 = DOWN
    return pd.DataFrame({"fwd_return": fwd, "label": label})


def sweep_thresholds(df: pd.DataFrame) -> None:
    """Print the class balance each threshold produces, so the choice is evidence-based.

    The floor is ROUND_TRIP_COST: below it, a move labelled UP loses money after
    fees, so the model would be trained to chase unprofitable trades.
    """
    entry, exit_ = df["open"].shift(-1), df["close"].shift(-HORIZON)
    fwd = (exit_ / entry - 1).dropna()
    print(f"\n30m forward return: std {fwd.std():.4%}  median abs {fwd.abs().median():.4%}")
    print(f"round-trip cost floor: {ROUND_TRIP_COST:.4%}\n")
    print(f"{'threshold':>10} {'DOWN':>7} {'NEUTRAL':>8} {'UP':>7}   {'net edge if right':>18}")
    for t in (0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050):
        up = (fwd > t).mean()
        dn = (fwd < -t).mean()
        flag = "" if t >= ROUND_TRIP_COST else "  <- below cost floor"
        edge = fwd[fwd.abs() > t].abs().mean() - ROUND_TRIP_COST
        print(f"{t:>10.4%} {dn:>7.1%} {1 - up - dn:>8.1%} {up:>7.1%}   {edge:>17.4%}{flag}")


def main() -> None:
    df = pd.read_parquet(SRC)

    # Reindex onto a complete 5m grid so rolling windows span real time, not row
    # counts. 40 gaps in 525k rows, but one missing candle silently shifts every
    # window that straddles it.
    full = pd.date_range(df["open_time"].iloc[0], df["open_time"].iloc[-1], freq="5min")
    df = df.set_index("open_time").reindex(full)
    df["is_gap"] = df["close"].isna()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    df[["volume", "trades"]] = df[["volume", "trades"]].fillna(0)
    df = df.rename_axis("open_time").reset_index()

    print(f"{len(df):,} candles, {int(df['is_gap'].sum())} filled gaps")
    print(f"horizon: {HORIZON} bars = {horizon_label()}")

    feats = add_indicators(df)
    sweep_thresholds(df)

    threshold = ROUND_TRIP_COST
    out = pd.concat([df[["open_time", "open", "high", "low", "close", "volume", "is_gap"]],
                     feats, add_labels(df, threshold)], axis=1)

    # Warmup rows (longest window is 7d vol percentile) and the unlabelled tail.
    out = out.iloc[BARS_PER_DAY * 7:-HORIZON].reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    feat_cols = list(feats.columns)
    na = out[feat_cols].isna().sum()
    if na.any():
        print("\nremaining NaNs:\n", na[na > 0])
    out[feat_cols] = out[feat_cols].ffill().fillna(0)

    out.to_parquet(OUT, index=False)
    counts = out["label"].value_counts(normalize=True).sort_index()
    print(f"\n{len(out):,} labelled rows, {len(feat_cols)} features -> {OUT}")
    print(f"threshold {threshold:.4%}   DOWN {counts.get(0, 0):.1%}  "
          f"NEUTRAL {counts.get(1, 0):.1%}  UP {counts.get(2, 0):.1%}")


def _self_check() -> None:
    """Catch the two bugs that would silently poison everything downstream:
    a lookahead leak in the labels, and a non-stationary feature."""
    n = 2000
    rng = np.random.default_rng(0)
    price = 30000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    t = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open_time": t, "open": price, "close": price * 1.0001,
        "high": price * 1.001, "low": price * 0.999,
        "volume": rng.uniform(1, 10, n), "trades": rng.integers(1, 100, n),
    })

    lab = add_labels(df, 0.0025)
    # Row i's label must depend only on rows > i. Changing the past must not move it.
    df2 = df.copy()
    df2.loc[:100, "close"] *= 1.05
    lab2 = add_labels(df2, 0.0025)
    assert (lab["label"].iloc[200:] == lab2["label"].iloc[200:]).all(), "label reads the past"

    # And it must depend on the future: only open[t+1] and close[t+HORIZON] are
    # read. Drive the exit candle to both extremes rather than nudging it -- at a
    # long horizon the unperturbed label is often already directional, so a bump
    # in one direction can legitimately leave it unchanged.
    up_df, dn_df = df.copy(), df.copy()
    up_df.loc[500 + HORIZON, "close"] = df["open"].iloc[501] * 1.10
    dn_df.loc[500 + HORIZON, "close"] = df["open"].iloc[501] * 0.90
    assert add_labels(up_df, 0.0025)["label"].iloc[500] == 2, "exit up must label UP"
    assert add_labels(dn_df, 0.0025)["label"].iloc[500] == 0, "exit down must label DOWN"

    # Scale invariance: 10x the price, features must be unchanged.
    f1 = add_indicators(df)
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 10
    f2 = add_indicators(scaled)
    drift = (f1 - f2).abs().max().max()
    assert drift < 1e-9, f"non-stationary feature, max drift {drift}"

    print("self-check passed: no lookahead, labels use the future, features scale-free")


if __name__ == "__main__":
    import sys
    _self_check() if "--check" in sys.argv else main()
