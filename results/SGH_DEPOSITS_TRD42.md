# TRD-42 — depozyty osób prywatnych, tier I (sgh-deposits_households-1)

Implementacja: `experiments/run_sgh_deposits_households_tier1.py`, modele w `experiments/sgh/tier1_deposits_households.py`, panel `experiments/sgh/deposits_households_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `ardl` | ARDL — dobór opóźnień poziomu depozytów i makro (p w siatce, walidacja w foldzie) |
| `ecm` | ECM — relacja długookresowa + ECT (korekta błędem) i krótki bieg |
| `dynamic_regression` | Regresja dynamiczna — Ridge na lagach depozytów, stopy NBP, CPI, kredytów |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku depozytów). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=month)`.

## Dane

| Seria | Loader |
|-------|--------|
| `deposits_households`, `credit_consumer`, `credit_mortgage` | NBP statystyka monetarna (`macro_pl`, XLS) — **do weryfikacji** |
| `nbp_rate`, `gus_cpi_yoy` | NBP API + GUS BDL / CSV — **do weryfikacji** |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_deposits_households_tier1.json`
- `results/sgh_deposits_households_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
pytest tests/test_sgh_deposits_households_tier1.py -q
python -m experiments.run_sgh_deposits_households_tier1 --demo --max-folds 2
python -m experiments.macro_pl --check --series deposits_households --cache-only
python -m experiments.run_sgh_deposits_households_tier1 --cache-only
```
