# TRD-48 — USD/PLN, tier I (sgh-usdpln-1)

Implementacja: `experiments/run_sgh_usdpln_tier1.py`, modele w `experiments/sgh/tier1_eurpln.py` (wspólny stack FX tier I), panel `experiments/sgh/usdpln_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `patchtst` | PatchTST (RevIN + patch transformer), early stopping na końcówce train foldu |
| `stacking` | XGBoost + ARIMA(1,0,1) + GRU → Ridge na walidacji w foldzie |
| `pso_lstm` | PSO nad (hidden, lr, window) GRU; najlepsza konfiguracja trenowana na całym train |

Horyzonty: **1 / 7 / 30** dni (poziom USD/PLN). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

## Dane

| Seria | Loader |
|-------|--------|
| `usdpln` | `experiments/multiasset.py` — NBP tabela A (`usd`) + Yahoo `USDPLN=X` (zweryfikowane 29.10) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_usdpln_tier1.json`
- `results/sgh_usdpln_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_usdpln_tier1 --cache-only

# smoke
python -m experiments.run_sgh_usdpln_tier1 --demo --max-folds 2

pytest tests/test_sgh_usdpln_tier1.py -q
```

Tier II — osobne zadanie.
