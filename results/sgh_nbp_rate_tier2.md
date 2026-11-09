# TRD-33 — stopa referencyjna NBP, tier II

Seria: `nbp_rate` · panel 2000-01-31 → 2025-12-31 (312 mies.)

Modele: VAR, BVAR (ridge/Minnesota-lite), SVAR (Cholesky, fallback VAR), oczekiwania + ton RPP, random forest, gradient boosting, NLP (TF–IDF komunikatów). Opcjonalne komunikaty: `data/macro_pl/raw/rpp_communications.csv` (date, tone, text). Harness: `evaluation/walk_forward.py`; dane: NBP + GUS CPI/PMI — **do weryfikacji**.

| h (mies.) | model | RMSE | skill vs RW | skill vs naive | DM-p vs naive | DM-p vs RW |
|---:|---|---:|---:|---:|---:|---:|
| 1 | var | 0.2848 | -0.9778 | 0.4087 | 0.0475 | 0.0020 |
| 1 | bvar | 0.7223 | -11.7263 | -2.8045 | 0.0000 | 0.0000 |
| 1 | expectations_comm | 0.2325 | -0.3186 | 0.6058 | 0.0011 | 0.1126 |
| 1 | random_forest | 0.8468 | -16.4892 | -4.2283 | 0.0033 | 0.0013 |
| 1 | gradient_boosting | 0.5496 | -6.3667 | -1.2022 | 0.0001 | 0.0000 |
| 1 | svar | 0.2927 | -1.0901 | 0.3752 | 0.0664 | 0.0006 |
| 1 | rpp_nlp | 3.7031 | -333.4755 | -98.9891 | 0.0000 | 0.0000 |
| 3 | var | 1.0362 | -2.8387 | -0.2033 | 0.6780 | 0.0056 |
| 3 | bvar | 2.3766 | -19.1954 | -5.3307 | 0.0000 | 0.0000 |
| 3 | expectations_comm | 0.5966 | -0.2726 | 0.6011 | 0.1357 | 0.4696 |
| 3 | random_forest | 0.9134 | -1.9827 | 0.0650 | 0.8200 | 0.0111 |
| 3 | gradient_boosting | 0.8610 | -1.6503 | 0.1692 | 0.6957 | 0.0027 |
| 3 | svar | 0.9983 | -2.5631 | -0.1169 | 0.8105 | 0.0104 |
| 3 | rpp_nlp | 3.5819 | -44.8723 | -13.3797 | 0.0000 | 0.0000 |
| 12 | var | 5.1230 | -9.0180 | -3.9620 | 0.0327 | 0.0218 |
| 12 | bvar | 10.0961 | -37.9082 | -18.2713 | 0.0073 | 0.0065 |
| 12 | expectations_comm | 1.5400 | 0.0948 | 0.5516 | 0.2782 | 0.8320 |
| 12 | random_forest | 1.9039 | -0.3837 | 0.3147 | 0.3811 | 0.1566 |
| 12 | gradient_boosting | 1.7102 | -0.1164 | 0.4470 | 0.4263 | 0.8237 |
| 12 | svar | 4.5712 | -6.9762 | -2.9506 | 0.0646 | 0.0437 |
| 12 | rpp_nlp | 3.0708 | -2.5993 | -0.7828 | 0.3106 | 0.0101 |

## Tier II vs tier I (skill vs naive)

- h=1: najlepszy tier II `expectations_comm` (skill 0.6058) vs tier I `ordered_probit` (skill 0.7440) — tier II lepszy: **nie**
- h=3: najlepszy tier II `expectations_comm` (skill 0.6011) vs tier I `ordered_logit` (skill 0.7784) — tier II lepszy: **nie**
- h=12: najlepszy tier II `expectations_comm` (skill 0.5516) vs tier I `ordered_logit` (skill 0.5113) — tier II lepszy: **tak**

Skill = 1 − MSE_model/MSE_benchmark. Złożoność (VAR/ML/NLP) nie gwarantuje poprawy względem tier I; sekcja powyżej porównuje najlepszy skill vs naive w każdym horyzoncie.

Uruchomienie:
```bash
python -m experiments.macro_pl --check --series nbp_rate --series gus_cpi_yoy --series gus_pmi
python -m experiments.run_sgh_nbp_tier2 --cache-only
pytest tests/test_sgh_nbp_tier2.py -q
```
