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
from numpy.lib.stride_tricks import sliding_window_view

from config import (HORIZON, BARS_PER_DAY, COST as ROUND_TRIP_COST,
                    THRESHOLD, THRESHOLD_MULT, LABEL_MODE, BARRIER_R,
                    BARRIER_RR, BARRIER_MAX_HOLD, BREAKEVEN_WIN, horizon_label)

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
SRC = DATA / "BTCUSDT_5m.parquet"
OUT = (DATA / (f"BTCUSDT_5m_features_barrier_r{BARRIER_R*1000:g}.parquet"
               if LABEL_MODE == "barrier" else
               f"BTCUSDT_5m_features_h{HORIZON}_t{THRESHOLD_MULT:g}.parquet"))


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


# Every column that is derived from the future. train.py excludes these from the
# feature set by importing this name rather than repeating the list, because the
# lists drifted apart once already and the model was handed its own answer.
LABEL_COLUMNS = {"label", "fwd_return", "barrier_return", "long_pnl", "short_pnl",
                 "long_tp_bar", "long_sl_bar"}

FUNDING_PER_DAY = 3        # paid every 8 hours


def add_funding_features(df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Leveraged-positioning features from the perpetual funding rate.

    Funding is the one positioning signal with real history -- Binance keeps
    open interest and the long/short ratio for only ~30 days, so neither can be
    trained on. It is information price does not carry: a persistently positive
    rate means longs are paying to stay long, which is crowding.

    merge_asof with direction="backward" is what keeps this honest. At candle t
    it attaches the most recent funding event published at or BEFORE t. A plain
    join on nearest would let a rate published at 16:00 leak into candles from
    15:55, which is a four-hour lookahead dressed up as a merge.
    """
    f = pd.DataFrame(index=df.index)

    merged = pd.merge_asof(
        df[["open_time"]].sort_values("open_time"),
        funding.sort_values("funding_time"),
        left_on="open_time", right_on="funding_time", direction="backward",
    )
    rate = merged["funding_rate"].to_numpy()
    f["funding_rate"] = rate

    s = pd.Series(rate, index=df.index)
    # Windows are in candles: 8h of funding = 96 candles.
    per_event = 96
    f["funding_cum_1d"] = s.rolling(per_event * FUNDING_PER_DAY, min_periods=1).mean()
    f["funding_cum_7d"] = s.rolling(per_event * FUNDING_PER_DAY * 7, min_periods=1).mean()

    long_win = per_event * FUNDING_PER_DAY * 30
    mu = s.rolling(long_win, min_periods=per_event).mean()
    sd = s.rolling(long_win, min_periods=per_event).std()
    f["funding_z"] = (s - mu) / sd.replace(0, np.nan)
    f["funding_pctile"] = s.rolling(long_win, min_periods=per_event).rank(pct=True)

    # Funding settles at 00:00, 08:00 and 16:00 UTC. Position crowding unwinds
    # around those times, so where we sit in the cycle carries information.
    hours = df["open_time"].dt.hour + df["open_time"].dt.minute / 60
    phase = (hours % 8) / 8
    f["funding_phase_sin"] = np.sin(2 * np.pi * phase)
    f["funding_phase_cos"] = np.cos(2 * np.pi * phase)

    return f


def add_sentiment_features(df: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """Crowd-sentiment features from the Fear & Greed index.

    Published once a day, so merge_asof backward again: at candle t only the
    value already published is visible. A forward or nearest merge would let
    tomorrow's sentiment inform today's trade, which is a full day of lookahead.

    The index is contrarian by construction -- extremes mark turning points more
    often than continuations -- so the distance from neutral and the rate of
    change matter more than the level.
    """
    f = pd.DataFrame(index=df.index)

    merged = pd.merge_asof(
        df[["open_time"]].sort_values("open_time"),
        fng.sort_values("fng_time"),
        left_on="open_time", right_on="fng_time", direction="backward",
    )
    v = pd.Series(merged["fng"].to_numpy(), index=df.index)

    f["fng"] = v / 50 - 1                       # -1 extreme fear .. +1 extreme greed
    f["fng_extreme"] = (v / 50 - 1).abs()       # distance from neutral
    f["fng_chg_7d"] = (v - v.shift(BARS_PER_DAY * 7)) / 100
    f["fng_z_30d"] = ((v - v.rolling(BARS_PER_DAY * 30, min_periods=BARS_PER_DAY).mean())
                      / v.rolling(BARS_PER_DAY * 30, min_periods=BARS_PER_DAY).std()
                      .replace(0, np.nan))
    return f


def add_barrier_labels(df: pd.DataFrame, R: float = BARRIER_R,
                       rr: float = BARRIER_RR,
                       max_hold: int = BARRIER_MAX_HOLD) -> pd.DataFrame:
    """Label what a real trade would have done, not where price ended up.

    For every candle, two hypothetical positions are opened at the next bar's
    open and run forward until one of their barriers is touched:

        LONG   target +rr*R   stop -R
        SHORT  target -rr*R   stop +R

    label 2 (UP)      the long reached its target before its stop
    label 0 (DOWN)    the short reached its target before its stop
    label 1 (NEUTRAL) neither did

    These are mutually exclusive. If the long wins, price passed +R on the way
    to +rr*R, so the short was already stopped out.

    Why this and not a fixed-horizon label: a horizon label says "price was
    higher in four hours", but a position with a stop can be closed at a loss
    hours earlier and never see that price. Training on the horizon teaches the
    model to answer a question the trading rule never asks.

    Vectorised over sliding windows -- the windows are views, so only the
    boolean comparison is materialised, and that is chunked to bound memory.
    """
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    n = len(o)

    entry = np.roll(o, -1)                      # fill at the NEXT bar's open
    NEVER = np.iinfo(np.int32).max
    long_tp = np.full(n, NEVER, np.int64); long_sl = np.full(n, NEVER, np.int64)
    short_tp = np.full(n, NEVER, np.int64); short_sl = np.full(n, NEVER, np.int64)

    hw = sliding_window_view(h, max_hold)
    lw = sliding_window_view(l, max_hold)
    usable = len(hw) - 1
    CHUNK = 40000
    for s0 in range(0, usable, CHUNK):
        s1 = min(s0 + CHUNK, usable)
        e = entry[s0:s1][:, None]
        H, L = hw[s0 + 1:s1 + 1], lw[s0 + 1:s1 + 1]
        for arr, hit in (
            (long_tp,  H >= e * (1 + R * rr)),
            (long_sl,  L <= e * (1 - R)),
            (short_tp, L <= e * (1 - R * rr)),
            (short_sl, H >= e * (1 + R)),
        ):
            arr[s0:s1] = np.where(hit.any(1), hit.argmax(1), NEVER)

    label = np.ones(n, dtype=np.int8)
    label[long_tp < long_sl] = 2
    label[short_tp < short_sl] = 0
    # Beyond the last full window there is no future to resolve against.
    label[usable:] = -1
    # One return column cannot describe both directions. A long resolves on
    # (+rr*R vs -R); a short resolves on (-rr*R vs +R). Those are different
    # races down the same price path, and a candle can be a loss for one and a
    # win for the other. So each direction gets its own realised trade return,
    # positive meaning profit, and the caller picks the one it actually traded.
    close = df["close"].to_numpy(np.float64)
    timeout_i = np.minimum(np.arange(n) + max_hold, n - 1)
    drift = close[timeout_i] / np.where(entry == 0, np.nan, entry) - 1

    long_pnl = np.where(long_tp < long_sl, R * rr,
                        np.where(long_sl < long_tp, -R, drift))
    short_pnl = np.where(short_tp < short_sl, R * rr,
                         np.where(short_sl < short_tp, -R, -drift))

    return pd.DataFrame({
        "label": label,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }, index=df.index)


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

    funding_path = DATA / "BTCUSDT_funding.parquet"
    if funding_path.exists():
        funding = pd.read_parquet(funding_path)
        fund = add_funding_features(df, funding)
        feats = pd.concat([feats, fund], axis=1)
        print(f"funding: {len(funding):,} events, {fund.shape[1]} features")
    else:
        print("no funding data; run ml/fetch_funding.py to include it")

    fng_path = DATA / "fear_greed.parquet"
    if fng_path.exists():
        fng = pd.read_parquet(fng_path)
        sent = add_sentiment_features(df, fng)
        feats = pd.concat([feats, sent], axis=1)
        print(f"sentiment: {len(fng):,} daily values, {sent.shape[1]} features")
    else:
        print("no sentiment data; run ml/fetch_sentiment.py to include it")

    sweep_thresholds(df)

    if LABEL_MODE == "barrier":
        lab = add_barrier_labels(df)
        print(f"\nbarrier labels: R {BARRIER_R:.2%}  "
              f"target {BARRIER_R*BARRIER_RR:.2%}  "
              f"max hold {BARRIER_MAX_HOLD//12}h")
        v = lab["label"][lab["label"] >= 0]
        print(f"  DOWN {(v==0).mean():.1%}  NEUTRAL {(v==1).mean():.1%}  UP {(v==2).mean():.1%}")
        print(f"  break-even win rate {BREAKEVEN_WIN:.1%} at {BARRIER_RR:g}:1 "
              f"(a coin flip gets {1/(1+BARRIER_RR):.1%})")
    else:
        lab = add_labels(df, THRESHOLD)
    out = pd.concat([df[["open_time", "open", "high", "low", "close", "volume", "is_gap"]],
                     feats, lab], axis=1)

    # Warmup rows (longest window is 7d vol percentile) and the unlabelled tail.
    tail = BARRIER_MAX_HOLD + 1 if LABEL_MODE == "barrier" else HORIZON
    out = out.iloc[BARS_PER_DAY * 7:-tail].reset_index(drop=True)
    if LABEL_MODE == "barrier":
        out = out[out["label"] >= 0].reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    feat_cols = list(feats.columns)
    na = out[feat_cols].isna().sum()
    if na.any():
        print("\nremaining NaNs:\n", na[na > 0])
    out[feat_cols] = out[feat_cols].ffill().fillna(0)

    out.to_parquet(OUT, index=False)
    counts = out["label"].value_counts(normalize=True).sort_index()
    print(f"\n{len(out):,} labelled rows, {len(feat_cols)} features -> {OUT}")
    label_desc = (f"barrier {BARRIER_R:.2%}/{BARRIER_R*BARRIER_RR:.2%}"
                  if LABEL_MODE == "barrier" else f"threshold {THRESHOLD:.4%}")
    print(f"{label_desc}   DOWN {counts.get(0, 0):.1%}  "
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

    # Funding must never be visible before it is published. A rate settled at
    # 08:00 must not appear on any candle before 08:00 -- merge_asof with the
    # wrong direction is an eight-hour leak that nothing else would catch.
    ft = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00"], utc=True)
    funding = pd.DataFrame({"funding_time": ft, "funding_rate": [0.0001, 0.0009, 0.0002]})
    candles = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=200,
                                                       freq="5min", tz="UTC")})
    ff = add_funding_features(candles, funding)
    before = candles["open_time"] < ft[1]
    assert (ff.loc[before, "funding_rate"] == 0.0001).all(), "funding leaked before publication"
    at_or_after = (candles["open_time"] >= ft[1]) & (candles["open_time"] < ft[2])
    assert (ff.loc[at_or_after, "funding_rate"] == 0.0009).all(), "funding not applied once published"

    # Sentiment published daily must not be visible before publication either.
    ft2 = pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
    fng = pd.DataFrame({"fng_time": ft2, "fng": [20.0, 80.0]})
    c2 = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=400,
                                                  freq="5min", tz="UTC")})
    sf = add_sentiment_features(c2, fng)
    pre = c2["open_time"] < ft2[1]
    assert (sf.loc[pre, "fng"] == (20 / 50 - 1)).all(), "sentiment leaked before publication"

    # Barrier labels: a hand-built path with a known answer.
    n2 = 3000
    price = np.full(n2, 100.0)
    price[10:] = 100.0
    t2 = pd.date_range("2024-01-01", periods=n2, freq="5min", tz="UTC")
    bd = pd.DataFrame({"open_time": t2, "open": price, "close": price,
                       "high": price, "low": price,
                       "volume": np.ones(n2), "trades": np.ones(n2)})
    # From bar 0 (entry at bar 1) price rises 2% at bar 5: long target (+1.5%)
    # is reached without the stop (-0.5%) ever being touched.
    bd.loc[5, "high"] = 102.0
    bl = add_barrier_labels(bd, R=0.005, rr=3.0, max_hold=100)
    assert bl["label"].iloc[0] == 2, f"clean long win mislabelled: {bl['label'].iloc[0]}"

    # Same rise, but the stop is touched first at bar 3 -- the long loses.
    bd2 = bd.copy(); bd2.loc[3, "low"] = 99.0
    bl2 = add_barrier_labels(bd2, R=0.005, rr=3.0, max_hold=100)
    assert bl2["label"].iloc[0] != 2, "stop touched first must not label a win"

    # Mirror: a 2% fall is a short win. Built from a clean frame, not from bd --
    # bd already spikes high at bar 5, which would touch both targets at once.
    bd3 = pd.DataFrame({"open_time": t2, "open": price, "close": price,
                        "high": price, "low": price,
                        "volume": np.ones(n2), "trades": np.ones(n2)})
    bd3.loc[5, "low"] = 98.0
    assert add_barrier_labels(bd3, R=0.005, rr=3.0, max_hold=100)["label"].iloc[0] == 0

    # Each direction carries its own realised return, both positive when that
    # direction wins. A price path that is a win for the long is a loss for the
    # short, and one column could not say both.
    lab_up = add_barrier_labels(bd, R=0.005, rr=3.0, max_hold=100)
    lab_dn = add_barrier_labels(bd3, R=0.005, rr=3.0, max_hold=100)
    assert lab_up["long_pnl"].iloc[0] > 0 and lab_up["short_pnl"].iloc[0] < 0,         "a long win must be a short loss"
    assert lab_dn["short_pnl"].iloc[0] > 0 and lab_dn["long_pnl"].iloc[0] < 0,         "a short win must be a long loss"
    # The winner collects rr times what the loser pays, with opposite signs.
    assert abs(lab_up["long_pnl"].iloc[0] / lab_up["short_pnl"].iloc[0] + 3.0) < 1e-9

    # Flat price reaches no barrier at all.
    assert add_barrier_labels(bd.assign(high=price, low=price),
                              R=0.005, rr=3.0, max_hold=100)["label"].iloc[0] == 1

    # The last max_hold rows have no resolvable future and must be marked -1.
    assert (add_barrier_labels(bd, R=0.005, rr=3.0, max_hold=100)["label"].iloc[-50:] == -1).all()

    print("self-check passed: no lookahead, labels use the future, features scale-free")


if __name__ == "__main__":
    import sys
    _self_check() if "--check" in sys.argv else main()
