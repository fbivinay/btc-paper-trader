# Meta-labelling experiment

Two-stage design from López de Prado: a primary decides direction, a secondary
decides whether to take that trade. Tested because the LSTM had shown real skill
at direction-in-4-hours and none at barrier order.

## Setup
- Label: does a long reach +1.5% before -0.5%, within 72h (binary)
- Model: HistGradientBoostingClassifier, 41 features, 2y train / 90d test, 4 folds
- Fees: 0.10% round trip (Binance futures VIP0 taker, both legs)
- Break-even win rate at 3:1 with fees: 29.2%. Random: 25.0%

## Results

Per-candle, ignoring overlap:

| selection | trades | win rate |
|---|---|---|
| everything, always long | 103,680 | 25.95% |
| top 10% by long probability | 10,368 | **29.52%** |
| top 10% by short probability | 10,368 | 24.84% |
| top 10% by max(long, short) | 10,368 | 27.33% |

The edge is long-only. The short model has no skill — BTC's upward drift
(funding positive 85.8% of the time) makes "when does a long work" learnable and
"when does a short work" a fight against the tide. Mixing them halved the signal.

Non-overlapping, which is what can actually be traded:

| selection | gate | concurrent | trades/month | win rate | 12mo return |
|---|---|---|---|---|---|
| chronological | top 2% | 1 | 3.7 | 31.82% | +1.4% |
| chronological | top 10% | 3 | 20.2 | 24.69% | -24.1% |
| greedy by probability | top 10% | 3 | 16.4 | 29.95% | +8.4% |

## Conclusion: negative

1. **The apparent edge is mostly overlap.** 29.52% per-candle becomes 24.37%
   once trades cannot overlap. Top-decile signals cluster, so counting each
   candle independently counts the same price move many times.

2. **The best surviving result is not significant.** 44 trades at 31.82% is 14
   wins against 11 expected; sd 2.87, z = 1.04, p ~ 0.15.

3. **It fails at the required trade frequency.** Reaching 20+ trades/month needs
   a looser gate, and win rate falls below random there.

The `greedy` row is reported only as an upper bound. It ranks every signal in the
period before choosing, which requires seeing the future — the exact mechanism
that makes backtests lie.
