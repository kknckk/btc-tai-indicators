# TRD-53 — Pszenica (futures), tier II (sgh-wheat-2)

Implementacja: `experiments/run_sgh_wheat_tier2.py`, modele w `experiments/sgh/tier2_wheat.py`, cechy `wheat_tier2_features` w `experiments/sgh/wheat_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `sarima_lstm` | SARIMA (siatka order/seasonal na walidacji) + LSTM na cechach → Ridge meta na końcówce train foldu |
| `fedformer` | Dekompozycja trend/sezon w oknie, filtr Fourier (FFT), transformer encoder |
| `gwrfr` | Random forest z wagami próbek z jądra Gaussa po „wieku” obserwacji (lokalność w czasie) |

Horyzonty: **1 / 7 / 30** dni (poziom ZW=F). Walk-forward jak tier I; porównanie skill vs naive względem najlepszego tier I z `results/sgh_wheat_tier1.json`.

## Dane

| Seria | Loader |
|-------|--------|
| `wheat` | `experiments/multiasset.py` — Yahoo `ZW=F` (zweryfikowane 29.10) |

## Wyniki

- `results/sgh_wheat_tier2.json`
- `results/sgh_wheat_tier2.md` (skill, DM-p, sekcja tier II vs tier I)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_wheat_tier2 --cache-only

# smoke
python -m experiments.run_sgh_wheat_tier2 --demo --max-folds 2

pytest tests/test_sgh_wheat_tier2.py -q
```

Pełny trening głęboki (większe okna/epoki) — opcjonalnie GPU na sgh-4; konfiguracja YAML celuje w CPU walk-forward.
