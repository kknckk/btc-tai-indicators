# TRD-58 — Bitcoin (PriceUSD + on-chain), tier I (sgh-bitcoin-1)

Implementacja: `experiments/run_sgh_bitcoin_tier1.py`, modele w `experiments/sgh/tier1_bitcoin.py`, panel `experiments/sgh/bitcoin_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `bilstm` | BiLSTM na oknach cech tabularnych; siatka window/hidden/lr na walidacji w foldzie |
| `xgboost` | XGBoost na cechach on-chain (`experiments.data.build_features`); max_depth i learning_rate z walidacji |

Horyzonty: **1 / 7 / 30** dni (poziom `PriceUSD`). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry wybierane na końcówce train foldu (`val_frac`), bez dotykania testu.

## Dane

| Seria | Loader |
|-------|--------|
| `bitcoin` | `experiments/data.py` — `PriceUSD` + on-chain CSV (`data_ingestion/csv`) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_bitcoin_tier1.json`
- `results/sgh_bitcoin_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.data --check
python -m experiments.run_sgh_bitcoin_tier1

# smoke
python -m experiments.run_sgh_bitcoin_tier1 --demo --max-folds 2

pytest tests/test_sgh_bitcoin_tier1.py -q
```
