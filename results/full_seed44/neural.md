# ModernTCN / PatchTST -- BTC log-return OOS results

- data source: `experiments.data.load_panel`
- walk-forward source: `evaluation.walk_forward.make_folds`
- rows used: 5788, smoke=False

| Model | Horizon (d) | RMSE | MAE | R2_OOS_vs_RW | Dir. Acc. | n_params | train_s/fit | folds |
|---|---|---|---|---|---|---|---|---|
| moderntcn | 1 | 0.036637 | 0.026408 | -0.3071 | 0.4944 | 235379 | 378.25 | 5 |
| moderntcn | 7 | 0.113798 | 0.087725 | -0.8018 | 0.5124 | 235379 | 218.87 | 5 |
| moderntcn | 30 | 0.245559 | 0.195364 | -0.8544 | 0.5416 | 235379 | 347.73 | 5 |
| patchtst | 1 | 0.036311 | 0.026325 | -0.2839 | 0.5026 | 101459 | 212.04 | 5 |
| patchtst | 7 | 0.098536 | 0.074077 | -0.3509 | 0.5107 | 101459 | 98.47 | 5 |
| patchtst | 30 | 0.212209 | 0.164767 | -0.3849 | 0.5416 | 101459 | 83.50 | 5 |
