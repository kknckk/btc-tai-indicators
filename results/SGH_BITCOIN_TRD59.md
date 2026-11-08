# TRD-59 — Bitcoin (PriceUSD + on-chain), tier II (sgh-bitcoin-2)

Implementacja: `experiments/run_sgh_bitcoin_tier2.py`, modele w `experiments/sgh/tier2_bitcoin.py`, panel `experiments/sgh/bitcoin_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `lstm` | LSTM na oknach cech tabularnych (jak commodities tier I); siatka window/hidden/lr na walidacji w foldzie |
| `tft` | Kompaktowy TFT: variable selection + BiLSTM + temporal attention (`tier2_eurpln.TFTForecaster`) |
| `svm` | RBF SVR na cechach on-chain; wybór `C` na końcówce train foldu |
| `hellformer` | Multi-scale heat convolutions + transformer encoder; wybór okna na walidacji |

Horyzonty: **1 / 7 / 30** dni (poziom `PriceUSD`). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry wybierane na końcówce train foldu (`val_frac`), bez dotykania testu.

## Tier II vs tier I

Po uruchomieniu tier I (`run_sgh_bitcoin_tier1`) raport tier II zawiera sekcję porównującą **najlepszy skill vs naive** w tier II z najlepszym w tier I (BiLSTM / XGBoost). Większa złożoność modelu nie gwarantuje poprawy.

## Dane

| Seria | Loader |
|-------|--------|
| `bitcoin` | `experiments/data.py` — `PriceUSD` + on-chain CSV (`data_ingestion/csv`) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_bitcoin_tier2.json`
- `results/sgh_bitcoin_tier2.md` (skill vs RW/naive, DM-p, porównanie z tier I)

```bash
python -m experiments.data --check
python -m experiments.run_sgh_bitcoin_tier2

# smoke
python -m experiments.run_sgh_bitcoin_tier2 --demo --max-folds 2

pytest tests/test_sgh_bitcoin_tier2.py -q
```

Modele głębokie (LSTM/TFT/HellFormer) mogą wymagać GPU na pełnym panelu — wtedy osobne zadanie na sgh-4.
