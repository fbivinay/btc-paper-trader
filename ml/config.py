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

# Execution model: BTC/USDT perpetual futures, working limit orders.
#
# This replaced spot taker pricing for one reason. Measured on the same model
# and the same 90-day out-of-sample window:
#
#   spot taker   0.250% round trip   -13.54%   sharpe -4.39
#   futures mkr  0.060% round trip    +6.01%   sharpe +0.84
#
# The binding constraint was never the model. It was paying 25bp to capture 4bp.
# A 4-hour holding period leaves ample time to work a limit order, which is what
# makes maker pricing realistic here rather than wishful.
FEE_MAKER = 0.0002           # per leg, posting liquidity
FEE_TAKER = 0.0005           # per leg, crossing the spread

# Not every limit order fills. When it does not, the choice is to chase (paying
# taker) or miss the trade. This blends the two, and the rate is deliberately
# conservative: break-even sits at roughly 65%, so assuming 100% would be
# assuming the answer.
MAKER_FILL_RATE = float(os.environ.get("BTC_MAKER_FILL_RATE", 0.60))

# Maker fills are adversely selected -- a resting bid fills precisely when
# sellers are hitting it. Charged per leg on top of the fee.
ADVERSE_SELECTION = 0.0001

SLIPPAGE = ADVERSE_SELECTION   # name kept: the paper engine charges it per leg
FEE = MAKER_FILL_RATE * FEE_MAKER + (1 - MAKER_FILL_RATE) * FEE_TAKER
COST = 2 * (FEE + SLIPPAGE)


# How far a move must exceed the round trip to be worth labelling tradeable.
#
# At spot fees the cost floor itself gave a sensible 33/33/34 split. At futures
# fees it gives 43/12/44 -- a model that trades almost every candle on very thin
# edges. Whether that beats trading selectively on fat ones is an empirical
# question, so it is swept rather than assumed.
THRESHOLD_MULT = float(os.environ.get("BTC_THRESHOLD_MULT", 3.0))
THRESHOLD = COST * THRESHOLD_MULT


def cost_at(fill_rate: float) -> float:
    """Round-trip cost at an arbitrary maker fill rate, for sensitivity analysis."""
    return 2 * (fill_rate * FEE_MAKER + (1 - fill_rate) * FEE_TAKER + ADVERSE_SELECTION)

BARS_PER_DAY = 288
BARS_PER_YEAR = BARS_PER_DAY * 365


def horizon_label() -> str:
    m = HORIZON * 5
    return f"{m}min" if m < 60 else f"{m // 60}h"
