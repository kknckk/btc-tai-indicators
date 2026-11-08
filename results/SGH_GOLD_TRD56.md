# TRD-56 — Złoto (GC=F), tier I (sgh-gold-1)

Implementacja: `experiments/run_sgh_gold_tier1.py`, modele w `experiments/sgh/tier1_gold.py`, panel `experiments/sgh/gold_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `lstm` | LSTM na oknach cech tabularnych; siatka window/hidden/lr na walidacji w foldzie |
| `garch` | GARCH(1,1) na zwrotach (×100 w `arch`); wybór AR mean (0/1) i rozkładu na walidacji |
| `xgboost` | XGBoost na cechach tabularnych; max_depth i learning_rate z walidacji |

Horyzonty: **1 / 7 / 30** dni (poziom GC=F). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry wybierane na końcówce train foldu (`val_frac`), bez dotykania testu.

## Dane

| Seria | Loader |
|-------|--------|
| `gold` | `experiments/multiasset.py` — Yahoo `GC=F` (zweryfikowane 29.10) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_gold_tier1.json`
- `results/sgh_gold_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_gold_tier1 --cache-only

# smoke
python -m experiments.run_sgh_gold_tier1 --demo --max-folds 2

pytest tests/test_sgh_gold_tier1.py -q
```
