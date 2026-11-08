# TRD-54 — Ropa WTI (CL=F), tier I (sgh-oil-1)

Implementacja: `experiments/run_sgh_oil_tier1.py`, modele w `experiments/sgh/tier1_oil.py`, panel `experiments/sgh/oil_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `lstm_svr` | LSTM (okna cech) + SVR na embeddingu; siatka window/hidden/lr oraz C/γ SVR na walidacji w foldzie |
| `xgboost` | XGBoost na cechach tabularnych; max_depth i learning_rate z walidacji |
| `garch_midas` | GARCH(1,1) na zwrotach skalowanych MIDAS (22d RV); wybór K i wag Beta na walidacji |

Horyzonty: **1 / 7 / 30** dni (poziom CL=F). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry wybierane na końcówce train foldu (`val_frac`), bez dotykania testu.

## Dane

| Seria | Loader |
|-------|--------|
| `oil` | `experiments/multiasset.py` — Yahoo `CL=F` (zweryfikowane 29.10) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_oil_tier1.json`
- `results/sgh_oil_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_oil_tier1 --cache-only

# smoke
python -m experiments.run_sgh_oil_tier1 --demo --max-folds 2

pytest tests/test_sgh_oil_tier1.py -q
```

Tier II (TRD-55): `experiments/run_sgh_oil_tier2.py`, `results/SGH_OIL_TRD55.md`.
