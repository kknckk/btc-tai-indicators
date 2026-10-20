# ModernTCN / PatchTST -- BTC log-return OOS results

- data source: `models._local_data.load_panel (fallback)`
- walk-forward source: `local expanding_window_splits (fallback)`
- rows used: 700, smoke=True

| Model | Horizon (d) | RMSE | MAE | R2_OOS_vs_RW | Dir. Acc. | n_params | train_s/fit | folds |
|---|---|---|---|---|---|---|---|---|
| moderntcn | 1 | 0.044081 | 0.035291 | -2.5317 | 0.5288 | 235379 | 4.77 | 2 |
| moderntcn | 7 | 0.065644 | 0.052193 | -0.2830 | 0.5789 | 235379 | 4.00 | 2 |
| moderntcn | 30 | 0.135526 | 0.109099 | -0.9462 | 0.3273 | 235379 | 3.80 | 2 |
| patchtst | 1 | 0.171642 | 0.131907 | -52.5477 | 0.4892 | 101459 | 1.24 | 2 |
| patchtst | 7 | 0.100430 | 0.079748 | -2.0029 | 0.5451 | 101459 | 1.31 | 2 |
| patchtst | 30 | 0.166708 | 0.136748 | -1.9448 | 0.4636 | 101459 | 1.14 | 2 |
