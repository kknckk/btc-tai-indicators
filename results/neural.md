# ModernTCN / PatchTST -- BTC log-return OOS results

- data source: `experiments.data.load_panel`
- walk-forward source: `evaluation.walk_forward.make_folds`
- rows used: 700, smoke=True

| Model | Horizon (d) | RMSE | MAE | R2_OOS_vs_RW | Dir. Acc. | n_params | train_s/fit | folds |
|---|---|---|---|---|---|---|---|---|
| moderntcn | 1 | 0.088501 | 0.080922 | -13.1732 | 0.5217 | 235379 | 4.26 | 2 |
| moderntcn | 7 | 0.123242 | 0.111048 | -3.4092 | 0.4729 | 235379 | 3.65 | 2 |
| moderntcn | 30 | 0.114622 | 0.095222 | 0.3771 | 0.6053 | 235379 | 3.55 | 2 |
| patchtst | 1 | 0.131462 | 0.108424 | -30.2731 | 0.4529 | 101459 | 1.20 | 2 |
| patchtst | 7 | 0.192840 | 0.156772 | -9.7953 | 0.5465 | 101459 | 1.16 | 2 |
| patchtst | 30 | 0.240950 | 0.194857 | -1.7525 | 0.3368 | 101459 | 1.08 | 2 |
