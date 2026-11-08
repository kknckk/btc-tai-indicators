# TRD-41 — kredyty firmowe, tier II (sgh-credit_corporate-2)

Implementacja: `experiments/run_sgh_credit_corporate_tier2.py`, modele w `experiments/sgh/tier2_credit_corporate.py`, panel `experiments/sgh/credit_corporate_panel.py` (`credit_corporate_tier2_features`).

## Modele (tier II)

| Model | Opis |
|-------|------|
| `var` | VAR na poziomie kredytu firmowego, stopy NBP, CPI, kredyt kons. |
| `bvar` | BVAR (ridge) — ta sama endog, shrinkage na lagach |
| `standards_survey` | Ridge: poziom kredytu + ankieta standardów (firmy) + makro |
| `fci` | Warunki finansowe (expanding z-score stopy/CPI/sektorów) + Ridge |
| `bank_firm` | Kanał bank–firma: Δstopy, standardy, spillovery kons./hip. |
| `lasso` | LASSO na pełnym wektorze cech tier II |
| `random_forest` | Random forest |
| `gradient_boosting` | HistGradientBoosting |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku). Walk-forward jak tier I.

## Dane

Jak TRD-40 (tier I): NBP statystyka monetarna, CPI, opcjonalnie `data/macro_pl/raw/nbp_credit_standards_firms.csv` — **do weryfikacji**.

## Wyniki

- `results/sgh_credit_corporate_tier2.json`
- `results/sgh_credit_corporate_tier2.md` (skill, DM, **porównanie najlepszego tier II vs tier I**)

```bash
pytest tests/test_sgh_credit_corporate_tier2.py -q
python -m experiments.run_sgh_credit_corporate_tier2 --demo --max-folds 2
python -m experiments.run_sgh_credit_corporate_tier1 --demo --max-folds 2   # opcjonalnie, do tier1_comparison
python -m experiments.run_sgh_credit_corporate_tier2 --cache-only
```

Złożoność (VAR/ML) nie musi poprawiać skill względem tier I — sekcja *Tier II vs tier I* w raporcie MD to weryfikuje.
