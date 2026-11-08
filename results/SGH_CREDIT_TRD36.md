# TRD-36 — kredyty konsumpcyjne, tier I (sgh-credit_consumer-1)

Implementacja: `experiments/run_sgh_credit_consumer_tier1.py`, modele w `experiments/sgh/tier1_credit_consumer.py`, panel `experiments/sgh/credit_consumer_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `ardl` | ARDL — dobór opóźnień poziomu kredytu i makro (p w siatce, walidacja w foldzie) |
| `ecm` | ECM — relacja długookresowa + ECT (korekta błędem) i krótki bieg |
| `dynamic_regression` | Regresja dynamiczna — Ridge na lagach kredytu, stopy NBP, CPI, depozytów |
| `logit_scorecard` | Logit na kierunku wzrostu (scorecard: standaryzacja cech) → prognoza poziomu |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku kredytów). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=month)`.

## Dane

| Seria | Loader |
|-------|--------|
| `credit_consumer`, `credit_mortgage`, `deposits_households` | NBP statystyka monetarna (`macro_pl`, XLS) — **do weryfikacji** |
| `nbp_rate`, `gus_cpi_yoy` | NBP API + GUS BDL / CSV — **do weryfikacji** |
| BIK (opcjonalnie) | Ręcznie `data/macro_pl/raw/bik_macro.csv` (`time`, `bik_index`) — **do weryfikacji** |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_credit_consumer_tier1.json`
- `results/sgh_credit_consumer_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
pytest tests/test_sgh_credit_consumer_tier1.py -q
python -m experiments.run_sgh_credit_consumer_tier1 --demo --max-folds 2
python -m experiments.macro_pl --check --series credit_consumer --cache-only
python -m experiments.run_sgh_credit_consumer_tier1 --cache-only
```
