# TRD-32 — stopa referencyjna NBP, tier I

Seria: `nbp_rate` · panel 2000-01-31 → 2025-12-31 (312 mies.)

Modele: reguła Taylora / reakcji, liniowy forward-looking (Ridge), ordered logit, ordered probit. Hiperparametry wybierane na końcówce train w każdym foldzie (bez dostrajania na teście). Harness: `evaluation/walk_forward.py` (sgh-2); dane: `experiments/macro_pl.py` (NBP + GUS CPI/PMI).

| h (mies.) | model | RMSE | skill vs RW | skill vs naive | DM-p vs naive | DM-p vs RW |
|---:|---|---:|---:|---:|---:|---:|
| 1 | taylor_rule | 0.2118 | -0.0938 | 0.6730 | 0.0000 | 0.0040 |
| 1 | forward_linear | 0.2187 | -0.1669 | 0.6512 | 0.0005 | 0.3382 |
| 1 | ordered_logit | 0.1923 | 0.0984 | 0.7305 | 0.0002 | 0.6005 |
| 1 | ordered_probit | 0.1874 | 0.1436 | 0.7440 | 0.0001 | 0.4212 |
| 3 | taylor_rule | 0.5664 | -0.1472 | 0.6404 | 0.0449 | 0.0058 |
| 3 | forward_linear | 0.5769 | -0.1899 | 0.6270 | 0.1106 | 0.5707 |
| 3 | ordered_logit | 0.4446 | 0.2932 | 0.7784 | 0.0387 | 0.1951 |
| 3 | ordered_probit | 0.4472 | 0.2850 | 0.7759 | 0.0394 | 0.2057 |
| 12 | taylor_rule | 2.3344 | -1.0802 | -0.0303 | 0.9229 | 0.2467 |
| 12 | forward_linear | 1.6527 | -0.0426 | 0.4836 | 0.3017 | 0.8824 |
| 12 | ordered_logit | 1.6078 | 0.0133 | 0.5113 | 0.2155 | 0.9075 |
| 12 | ordered_probit | 1.6101 | 0.0105 | 0.5099 | 0.2155 | 0.9246 |

Skill = 1 − MSE_model/MSE_benchmark (wyżej = lepiej). DM: Diebold–Mariano na kwadratach błędu poziomu.

Uruchomienie produkcyjne:
```bash
python -m experiments.macro_pl --check --series nbp_rate --series gus_cpi_yoy --series gus_pmi
python -m experiments.run_sgh_nbp_tier1 --cache-only
```
