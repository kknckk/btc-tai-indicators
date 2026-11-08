# TRD-46 — EUR/PLN, tier I (sgh-eurpln-1)

Implementacja: `experiments/run_sgh_eurpln_tier1.py`, modele w `experiments/sgh/tier1_eurpln.py`, panel `experiments/sgh/eurpln_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `patchtst` | PatchTST (RevIN + patch transformer), early stopping na końcówce train foldu |
| `stacking` | XGBoost + ARIMA(1,0,1) + GRU → Ridge na walidacji w foldzie |
| `pso_lstm` | PSO nad (hidden, lr, window) GRU; najlepsza konfiguracja trenowana na całym train |

Horyzonty: **1 / 7 / 30** dni (poziom EUR/PLN). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

## Dane

| Seria | Loader |
|-------|--------|
| `eurpln` | `experiments/multiasset.py` — NBP tabela A (`eur`) + Yahoo `EURPLN=X` (zweryfikowane 29.10) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_eurpln_tier1.json`
- `results/sgh_eurpln_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_eurpln_tier1 --cache-only

# smoke
python -m experiments.run_sgh_eurpln_tier1 --demo --max-folds 2

pytest tests/test_sgh_eurpln_tier1.py -q
```

Tier II — TRD-47 (`experiments/run_sgh_eurpln_tier2.py`).
