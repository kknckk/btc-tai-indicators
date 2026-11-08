# TRD-47 — EUR/PLN, tier II (sgh-eurpln-2)

Implementacja: `experiments/run_sgh_eurpln_tier2.py`, modele w `experiments/sgh/tier2_eurpln.py`, cechy `eurpln_tier2_features` w `experiments/sgh/eurpln_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `vmd_cnn_bilstm` | Tryby VMD (causal rolling MA) + Conv1d + BiLSTM, poziom EUR/PLN |
| `anomaly_transformer` | IsolationForest na cechach tabularnych + transformer z wagą stanów anomalii |
| `tft` | Uproszczony Temporal Fusion Transformer (variable selection + BiLSTM + attention) |

Horyzonty: **1 / 7 / 30** dni (poziom). Walk-forward jak tier I; porównanie skill vs naive względem najlepszego tier I z `results/sgh_eurpln_tier1.json`.

## Dane

| Seria | Loader |
|-------|--------|
| `eurpln` | `experiments/multiasset.py` — NBP tabela A + Yahoo `EURPLN=X` |

## Wyniki

- `results/sgh_eurpln_tier2.json`
- `results/sgh_eurpln_tier2.md` (skill, DM-p, sekcja tier II vs tier I)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_eurpln_tier2 --cache-only

# smoke
python -m experiments.run_sgh_eurpln_tier2 --demo --max-folds 2

pytest tests/test_sgh_eurpln_tier2.py -q
```

Pełny trening głęboki (większe okna/epoki) — opcjonalnie GPU na sgh-4; konfiguracja YAML celuje w CPU walk-forward.
