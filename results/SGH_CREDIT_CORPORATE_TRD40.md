# TRD-40 — kredyty firmowe, tier I (sgh-credit_corporate-1)

Implementacja: `experiments/run_sgh_credit_corporate_tier1.py`, modele w `experiments/sgh/tier1_credit_corporate.py`, panel `experiments/sgh/credit_corporate_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `ardl` | ARDL — opóźnienia poziomu kredytu firmowego, stopy NBP, CPI, sektor kons./hip. (p w siatce, walidacja w foldzie) |
| `dynamic_regression` | Regresja dynamiczna — Ridge na lagach kredytu firmowego i makro |
| `bank_lending_channel` | Kanał bankowy — OLS: zmiany stopy (1–3 opóźnienia), lag Δlog(kredyt), opcjonalna ankieta standardów |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku kredytów). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=month)`.

## Dane

| Seria | Loader |
|-------|--------|
| `credit_corporate`, `credit_consumer`, `credit_mortgage` | NBP statystyka monetarna (`macro_pl`, XLS) — **do weryfikacji** |
| `nbp_rate`, `gus_cpi_yoy` | NBP API + GUS BDL / CSV — **do weryfikacji** |
| Ankieta standardów kredytowych (firmy) | Ręcznie `data/macro_pl/raw/nbp_credit_standards_firms.csv` (`time`, wartość netto zaostrzania) — **do weryfikacji** |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_credit_corporate_tier1.json`
- `results/sgh_credit_corporate_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
pytest tests/test_sgh_credit_corporate_tier1.py -q
python -m experiments.run_sgh_credit_corporate_tier1 --demo --max-folds 2
python -m experiments.macro_pl --check --series credit_corporate --cache-only
python -m experiments.run_sgh_credit_corporate_tier1 --cache-only
```
