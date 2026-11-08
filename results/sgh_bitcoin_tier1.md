# TRD-58 — Bitcoin (PriceUSD + on-chain), tier I

Seria: `bitcoin` · panel 2009-01-03 → 2026-06-12 (6370 dni)

Modele: BiLSTM, XGBoost. Harness: `evaluation/walk_forward.py` (sgh-2); dane: `experiments/data.py` (sgh-1).

| h (dni) | model | RMSE | skill vs RW | skill vs naive | DM-p vs naive | DM-p vs RW |
|---:|---|---:|---:|---:|---:|---:|
| 1 | xgboost | 18329.9459 | -295.1322 | -152.6699 | 0.0000 | 0.0000 |
| 7 | xgboost | 22176.4969 | -63.9058 | -55.5005 | 0.0000 | 0.0000 |
| 30 | xgboost | 27922.9684 | -20.9204 | -20.0853 | 0.0000 | 0.0000 |

Skill = 1 − MSE_model/MSE_benchmark (wyżej = lepiej). DM: Diebold–Mariano na kwadratach błędu poziomu.

Uruchomienie produkcyjne:
```bash
python -m experiments.data --check
python -m experiments.run_sgh_bitcoin_tier1
```
