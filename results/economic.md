# Economic evaluation — common OOS panel

Panel 2009-01-03 → 2026-06-12. Expanding walk-forward (min_train=1000, step=30, purge=h). Positions at close of day *t* earn `next_ret` (day *t+1*); costs on |Δposition| at 10 and 20 bps.

Position rules: naive = buy & hold; har_rv/garch = vol-target (40% ann. vol, max lev 1.0); xgboost / neural / TSFM = sign(forecast) long/short/flat.

Forecast parquets: `results/preds/` (MCS/DM on squared return loss uses the date intersection of loaded models). **Current tables use committed `baselines.json` (four baseline models).** After `python -m experiments.run_baselines`, run `python -m experiments.run_economic` to refresh MCS and add ModernTCN/PatchTST/Chronos/TimesFM parquets when available.

## Horizon h=1

| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |
|---|---|---|---|---|---|
| naive | 1.07 | 1.07 | -0.85 | 0.08 | n/a |
| har_rv | 0.95 | 0.90 | -0.54 | 14.65 | n/a |
| garch | 0.85 | 0.81 | -0.70 | 16.62 | n/a |
| xgboost | 0.88 | 0.74 | -0.94 | 102.25 | n/a |

*MCS not computed (no aligned prediction parquets). Run `python -m experiments.run_baselines` then re-run this script.*

## Horizon h=7

| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |
|---|---|---|---|---|---|
| naive | 1.04 | 1.04 | -0.85 | 0.08 | n/a |
| har_rv | 0.99 | 0.98 | -0.50 | 16.62 | n/a |
| garch | 0.86 | 0.82 | -0.66 | 16.62 | n/a |
| xgboost | 1.02 | 0.98 | -0.84 | 16.62 | n/a |

*MCS not computed (no aligned prediction parquets). Run `python -m experiments.run_baselines` then re-run this script.*

## Horizon h=30

| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |
|---|---|---|---|---|---|
| naive | 1.06 | 1.06 | -0.85 | 0.08 | n/a |
| har_rv | 1.03 | 1.02 | -0.47 | 16.62 | n/a |
| garch | 0.90 | 0.87 | -0.56 | 16.62 | n/a |
| xgboost | 0.93 | 0.93 | -0.94 | 16.62 | n/a |

*MCS not computed (no aligned prediction parquets). Run `python -m experiments.run_baselines` then re-run this script.*

## Implementation notes

- Backtest: `evaluation/backtest.py` (`run_backtest`, sign / vol-target positions, aligned squared losses).
- MCS + pairwise DM: `evaluation/mcs.py` on the common loss panel.
- Unit tests: `tests/test_evaluation_harness.py` (synthetic MCS ranking, no look-ahead, hit-rate).
