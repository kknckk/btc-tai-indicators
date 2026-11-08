# TRD-52 — Pszenica (futures), tier I (sgh-wheat-1)

Implementacja: `experiments/run_sgh_wheat_tier1.py`, modele w `experiments/sgh/tier1_wheat.py`, panel `experiments/sgh/wheat_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `lstm` | LSTM na oknach cech; wybór (window, hidden, lr) na końcówce train foldu |
| `arima` | ARIMA na poziomie kontraktu; wybór rzędu z siatki na walidacji w foldzie |
| `xgboost` | XGBoost na cechach tabularnych; max_depth i learning_rate z walidacji |

Horyzonty: **1 / 7 / 30** dni (poziom ZW=F). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry wybierane na końcówce train foldu (`val_frac`), bez dotykania testu.

## Dane

| Seria | Loader |
|-------|--------|
| `wheat` | `experiments/multiasset.py` — Yahoo `ZW=F` (zweryfikowane 29.10) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_wheat_tier1.json`
- `results/sgh_wheat_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_wheat_tier1 --cache-only

# smoke
python -m experiments.run_sgh_wheat_tier1 --demo --max-folds 2

pytest tests/test_sgh_wheat_tier1.py -q
```

Tier II (TRD-53): `experiments/run_sgh_wheat_tier2.py`, `results/SGH_WHEAT_TRD53.md`.
