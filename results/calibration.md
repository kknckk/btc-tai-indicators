# Conformal calibration — BTC OOS intervals

Panel walk-forward and TSFM rolling-origin evaluation. Calibration scores use only OOS rows **before** each forecast origin (expanding window).

**Regenerate full tables (walk-forward + CQR):**

```bash
python -m experiments.run_baselines   # optional: writes results/preds/*.parquet
python -m experiments.run_calibration
```

Acceptance band for nominal 80%: empirical coverage **78–82%** per horizon.

## TSFM — nominal p10–p90 before conformal (committed runs)

| model | h | emp. coverage | median width (% of price) |
|---|---|---|---|
| TimesFM | 1 | 59.0% | 4.78% |
| TimesFM | 7 | 54.3% | 10.95% |
| TimesFM | 30 | 48.7% | 24.75% |
| Chronos-Bolt | 1 | 86.7% | 8.12% |
| Chronos-Bolt | 7 | 79.7% | 17.84% |
| Chronos-Bolt | 30 | 77.0% | 38.13% |

TimesFM bands are too narrow vs nominal 80%; Chronos can be too wide on some horizons. Rolling **CQR** on per-origin p10–p90 (see `evaluation/calibration.py`) retargets intervals; **split-conformal** on walk-forward log-return residuals does the same for point forecasts (XGBoost, naive, etc.).

## Methods (code)

| Method | Module | Use case |
|---|---|---|
| Split-conformal | `rolling_split_conformal`, `calibrate_frame_point` | Symmetric bands from past `\|y - ŷ\|` |
| CQR | `rolling_cqr`, `calibrate_frame_cqr` | Adjust TSFM p10–p90 with conformal score `max(q_lo - y, y - q_hi)` |
| Coverage vs width | `coverage_width_curve`, `coverage_width_curve_cqr` | Trade-off curves vs nominal level |
| RW width benchmark | `rolling_rw_widths` | Same nominal level; RW forecast 0 return (log) or last price (level) |

After `run_calibration`, this file is overwritten with per-horizon empirical coverage at 80%/95%, median width, and width relative to RW.
