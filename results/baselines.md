# Baselines — walk-forward results

Panel 2009-01-03 → 2026-06-12, seed 42, expanding window, min_train=1000, step=30, purge=h, extra embargo=0. Costs on |Δposition|.

Positions: naive = buy&hold (zero-return forecast carries no signal, so B&H is the economic RW benchmark); har_rv/garch = long-only vol-targeting (40% ann. vol, max lev 1.0); xgboost = sign(forecast) long/short. DM-p (ret) = Diebold-Mariano (HLN) on squared return error vs naive; har_rv/garch forecast zero return, so it is identical to naive (n/a) — their skill is in the variance table.

## Return forecasts (full OOS sample)

| model | h | RMSE | MAPE price % | R² OOS vs RW | dir-acc | Sharpe net 10bps | Sharpe net 20bps | MaxDD 20bps | DM-p vs naive (ret) |
|---|---|---|---|---|---|---|---|---|---|
| naive | 1 | 0.03826 | 2.44 | 0.0000 | n/a | 1.07 | 1.07 | -0.85 | n/a |
| har_rv | 1 | 0.03826 | 2.44 | 0.0000 | n/a | 0.95 | 0.90 | -0.54 | n/a |
| garch | 1 | 0.03826 | 2.44 | 0.0000 | n/a | 0.85 | 0.81 | -0.70 | n/a |
| xgboost | 1 | 0.03813 | 2.46 | 0.0071 | 0.518 | 0.88 | 0.74 | -0.94 | 0.576 |
| naive | 7 | 0.10180 | 7.01 | 0.0000 | n/a | 1.04 | 1.04 | -0.85 | n/a |
| har_rv | 7 | 0.10180 | 7.01 | 0.0000 | n/a | 0.99 | 0.98 | -0.50 | n/a |
| garch | 7 | 0.10180 | 7.01 | 0.0000 | n/a | 0.86 | 0.82 | -0.66 | n/a |
| xgboost | 7 | 0.10093 | 7.28 | 0.0170 | 0.528 | 1.02 | 0.98 | -0.84 | 0.601 |
| naive | 30 | 0.23898 | 15.98 | 0.0000 | n/a | 1.06 | 1.06 | -0.85 | n/a |
| har_rv | 30 | 0.23898 | 15.98 | 0.0000 | n/a | 1.03 | 1.02 | -0.47 | n/a |
| garch | 30 | 0.23898 | 15.98 | 0.0000 | n/a | 0.90 | 0.87 | -0.56 | n/a |
| xgboost | 30 | 0.26879 | 21.86 | -0.2650 | 0.545 | 0.93 | 0.93 | -0.94 | 0.115 |

## Same, sub-period since 2018-01-01

| model | h | RMSE | R² OOS vs RW | dir-acc | Sharpe net 20bps | MaxDD 20bps |
|---|---|---|---|---|---|---|
| naive | 1 | 0.03422 | 0.0000 | n/a | 0.65 | -0.81 |
| har_rv | 1 | 0.03422 | 0.0000 | n/a | 0.55 | -0.53 |
| garch | 1 | 0.03422 | 0.0000 | n/a | 0.57 | -0.61 |
| xgboost | 1 | 0.03414 | 0.0045 | 0.506 | 0.31 | -0.92 |
| naive | 7 | 0.09058 | 0.0000 | n/a | 0.65 | -0.81 |
| har_rv | 7 | 0.09058 | 0.0000 | n/a | 0.62 | -0.49 |
| garch | 7 | 0.09058 | 0.0000 | n/a | 0.58 | -0.60 |
| xgboost | 7 | 0.09260 | -0.0451 | 0.509 | 0.58 | -0.82 |
| naive | 30 | 0.19570 | 0.0000 | n/a | 0.65 | -0.81 |
| har_rv | 30 | 0.19570 | 0.0000 | n/a | 0.64 | -0.46 |
| garch | 30 | 0.19570 | 0.0000 | n/a | 0.59 | -0.53 |
| xgboost | 30 | 0.24144 | -0.5220 | 0.517 | 0.60 | -0.81 |

## Variance forecasts (target: mean daily r² over next h days)

naive = trailing-30d mean r². Lower QLIKE is better; DM on QLIKE loss vs naive.

| model | h | QLIKE | RMSE vol (daily) | DM stat vs naive | DM-p vs naive |
|---|---|---|---|---|---|
| naive | 1 | 2.1525 | 0.03154 | n/a | n/a |
| har_rv | 1 | 2.0714 | 0.03422 | -1.30 | 0.1946 |
| garch | 1 | 1.9868 | 0.03120 | -4.36 | 0.0000 |
| naive | 7 | 0.8222 | 0.02201 | n/a | n/a |
| har_rv | 7 | 0.7605 | 0.02568 | -0.74 | 0.4616 |
| garch | 7 | 0.6218 | 0.02143 | -3.51 | 0.0005 |
| naive | 30 | 0.6915 | 0.01801 | n/a | n/a |
| har_rv | 30 | 0.5495 | 0.02394 | -1.13 | 0.2568 |
| garch | 30 | 0.4195 | 0.02047 | -2.55 | 0.0109 |

## Average rank across horizons (1 = best)

| model | avg rank (return MSE) | avg rank (QLIKE) |
|---|---|---|
| naive | 2.67 | 3.00 |
| har_rv | 2.67 | 2.00 |
| garch | 2.67 | 1.00 |
| xgboost | 2.00 | n/a |

(ties in return MSE: naive/har_rv/garch share identical zero forecasts, hence equal rank.)

## XGBoost top-10 features by mean normalised gain — h=1

| feature | gain share |
|---|---|
| ret_cum_14 | 0.095 |
| ret_cum_1 | 0.094 |
| slog1p_RealizedPrice_d30 | 0.050 |
| slog1p_FeeMeanNtv_d30 | 0.042 |
| ret_cum_7 | 0.040 |
| slog1p_RealizedPrice_d7 | 0.035 |
| ret_cum_3 | 0.031 |
| ret_cum_5 | 0.026 |
| ret_cum_2 | 0.024 |
| slog1p_CapMrktCurUSD_d7 | 0.022 |

## XGBoost top-10 features by mean normalised gain — h=7

| feature | gain share |
|---|---|
| slog1p_CapRealUSD_d7 | 0.105 |
| NVTAdj90_d7 | 0.084 |
| slog1p_RealizedPrice_d30 | 0.079 |
| ret_cum_14 | 0.066 |
| NVTAdj90 | 0.061 |
| slog1p_RevAllTimeUSD_d30 | 0.052 |
| slog1p_CapRealUSD_d30 | 0.034 |
| slog1p_AdrBalCnt_d7 | 0.029 |
| NVT | 0.028 |
| ShareExUSD | 0.027 |

## XGBoost top-10 features by mean normalised gain — h=30

| feature | gain share |
|---|---|
| NVTAdj90 | 0.250 |
| slog1p_RevAllTimeUSD_d30 | 0.109 |
| slog1p_SplyCur_d30 | 0.105 |
| slog1p_CapRealUSD_d30 | 0.058 |
| slog1p_DiffMean_d30 | 0.049 |
| slog1p_SplyExpFut10yr_d30 | 0.045 |
| slog1p_AdrBalCnt_d30 | 0.042 |
| slog1p_FlowInExNtv_d30 | 0.034 |
| rv_30 | 0.028 |
| slog1p_UTXOCnt_d30 | 0.026 |

## Run info

- h=1: 160 folds, test 2013-04-14 → 2026-05-22, n_obs=4787, GARCH fit failures=0, XGB mean best_iter=79
- h=7: 160 folds, test 2013-04-20 → 2026-05-16, n_obs=4775, GARCH fit failures=0, XGB mean best_iter=50
- h=30: 158 folds, test 2013-05-13 → 2026-04-23, n_obs=4729, GARCH fit failures=0, XGB mean best_iter=23
