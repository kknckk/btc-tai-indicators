# Economic evaluation — common OOS panel

Panel 2009-01-03 → 2026-06-12. Expanding walk-forward (min_train=1000, step=30, purge=h). Positions at close of day *t* earn ``next_ret`` (day *t+1*); costs on |Δposition| at [10, 20] bps.

Position rules: naive = buy & hold; har_rv/garch = vol-target (40% ann. vol, max lev 1.0); others = sign(forecast) long/short/flat.

Forecast parquets: `/opt/data/grok_push/btc-tai-indicators/results/preds/` (MCS/DM on squared return loss uses the date intersection of loaded models).

## Horizon h=1

| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |
|---|---|---|---|---|---|
| naive | 1.07 | 1.07 | -0.85 | 0.08 | 0.527 |
| har_rv | 0.95 | 0.90 | -0.54 | 14.65 | 0.524 |
| garch | 0.85 | 0.81 | -0.70 | 16.62 | 0.526 |
| xgboost | 0.88 | 0.74 | -0.94 | 102.25 | 0.506 |

### Model Confidence Set (bootstrap, α=0.10, squared return loss)

Confidence set: **naive, har_rv, garch, xgboost**

Mean loss rank (lower is better):

| model | mean SE | elimination p |
|---|---|---|
| xgboost | 0.001454 | n/a |
| naive | 0.001464 | 0.281 |
| har_rv | 0.001464 | n/a |
| garch | 0.001464 | n/a |

Pairwise DM p-values (row vs column, squared error):

| | xgboost | naive | har_rv | garch |
|---|---|---|---|---|
| xgboost | — | 0.576 | 0.576 | 0.576 |
| naive | 0.576 | — | n/a | n/a |
| har_rv | 0.576 | n/a | — | n/a |
| garch | 0.576 | n/a | n/a | — |

## Horizon h=7

| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |
|---|---|---|---|---|---|
| naive | 1.04 | 1.04 | -0.85 | 0.08 | 0.527 |
| har_rv | 0.99 | 0.98 | -0.50 | 5.33 | 0.526 |
| garch | 0.86 | 0.82 | -0.66 | 14.95 | 0.526 |
| xgboost | 1.02 | 0.98 | -0.84 | 28.36 | 0.520 |

### Model Confidence Set (bootstrap, α=0.10, squared return loss)

Confidence set: **naive, har_rv, garch, xgboost**

Mean loss rank (lower is better):

| model | mean SE | elimination p |
|---|---|---|
| xgboost | 0.010187 | n/a |
| naive | 0.010362 | 0.298 |
| har_rv | 0.010362 | n/a |
| garch | 0.010362 | n/a |

Pairwise DM p-values (row vs column, squared error):

| | xgboost | naive | har_rv | garch |
|---|---|---|---|---|
| xgboost | — | 0.601 | 0.601 | 0.601 |
| naive | 0.601 | — | n/a | n/a |
| har_rv | 0.601 | n/a | — | n/a |
| garch | 0.601 | n/a | n/a | — |

## Horizon h=30

| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |
|---|---|---|---|---|---|
| naive | 1.06 | 1.06 | -0.85 | 0.08 | 0.528 |
| har_rv | 1.03 | 1.02 | -0.47 | 1.98 | 0.528 |
| garch | 0.90 | 0.87 | -0.56 | 9.33 | 0.526 |
| xgboost | 0.93 | 0.93 | -0.94 | 3.32 | 0.525 |

### Model Confidence Set (bootstrap, α=0.10, squared return loss)

Confidence set: **naive, har_rv, garch**

Mean loss rank (lower is better):

| model | mean SE | elimination p |
|---|---|---|
| naive | 0.057113 | n/a |
| har_rv | 0.057113 | 1.000 |
| garch | 0.057113 | n/a |
| xgboost | 0.072246 | 0.006 |

Pairwise DM p-values (row vs column, squared error):

| | naive | har_rv | garch | xgboost |
|---|---|---|---|---|
| naive | — | n/a | n/a | 0.115 |
| har_rv | n/a | — | n/a | 0.115 |
| garch | n/a | n/a | — | 0.115 |
| xgboost | 0.115 | 0.115 | 0.115 | — |

