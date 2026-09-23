"""Shared constants for the label horizon and trading costs.

These live in one place because a mismatch between them is silent and fatal: if
features.py labels a 4-hour move while metrics.py simulates non-overlapping
30-minute holds, every backtest number is wrong and nothing raises an error.

Override per run without editing code:
    BTC_HORIZON=6 python ml/features.py      # reproduce the 30-minute result
"""

import os

# Bars ahead the label looks. 48 x 5m = 4 hours.
#
# Chosen from data, not preference. Median absolute move over a horizon, against
# a 0.25% round-trip cost:
#     30 min  0.152%  0.61x cost   <- structurally unprofitable, see
#      1 h    0.212%  0.85x cost      results/walkforward_h6_30min.json
#      2 h    0.298%  1.19x cost
#      4 h    0.424%  1.70x cost   <- production
#     24 h    1.257%  5.03x cost
# Below ~1.5x the fee consumes the signal no matter how good the model is.
HORIZON = int(os.environ.get("BTC_HORIZON", 48))

FEE = 0.001                  # Binance spot taker, each way
SLIPPAGE = 0.0005
COST = 2 * FEE + SLIPPAGE    # 0.25% round trip

BARS_PER_DAY = 288
BARS_PER_YEAR = BARS_PER_DAY * 365


def horizon_label() -> str:
    m = HORIZON * 5
    return f"{m}min" if m < 60 else f"{m // 60}h"
