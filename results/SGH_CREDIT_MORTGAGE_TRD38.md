# TRD-38 — kredyty hipoteczne, tier I (sgh-credit_mortgage-1)

Implementacja: `experiments/run_sgh_credit_mortgage_tier1.py`, modele w `experiments/sgh/tier1_credit_mortgage.py`, panel `experiments/sgh/credit_mortgage_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `ardl` | ARDL — opóźnienia poziomu kredytu hipotecznego i makro (p w siatce, walidacja w foldzie) |
| `vecm` | VECM — układ log(kredyt hip.) + stopa NBP + CPI YoY; dobór `coint_rank` / `k_ar_diff` na walidacji |
| `dynamic_regression` | Regresja dynamiczna — Ridge na lagach hipoteki, stopy, CPI, kredytu kons., depozytów |
| `logit_scorecard` | Mikroekonometria: logit na kierunku wzrostu (scorecard) → prognoza poziomu |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=month)`.

## Dane

| Seria | Loader |
|-------|--------|
| `credit_mortgage`, `credit_consumer`, `deposits_households` | NBP statystyka monetarna (`macro_pl`, XLS) — **do weryfikacji** |
| `nbp_rate`, `gus_cpi_yoy` | NBP API + GUS BDL / CSV — **do weryfikacji** |
| AMRON / SARFiN (ceny/m²) | **do weryfikacji** — brak w loaderze |
| BIK (opcjonalnie) | `data/macro_pl/raw/bik_macro.csv` — **do weryfikacji** |

## Wyniki

- `results/sgh_credit_mortgage_tier1.json`
- `results/sgh_credit_mortgage_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
pytest tests/test_sgh_credit_mortgage_tier1.py -q
python -m experiments.run_sgh_credit_mortgage_tier1 --demo --max-folds 2
python -m experiments.macro_pl --check --series credit_mortgage --cache-only
python -m experiments.run_sgh_credit_mortgage_tier1 --cache-only
```
