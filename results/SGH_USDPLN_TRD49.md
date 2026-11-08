# TRD-49 — USD/PLN, tier II (sgh-usdpln-2)

Implementacja: `experiments/run_sgh_usdpln_tier2.py`, modele w `experiments/sgh/tier2_eurpln.py` (wspólny stack FX tier II), cechy `usdpln_tier2_features` w `experiments/sgh/usdpln_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `vmd_cnn_bilstm` | Tryby VMD (causal rolling MA) + Conv1d + BiLSTM, poziom USD/PLN |
| `anomaly_transformer` | IsolationForest na cechach tabularnych + transformer z wagą stanów anomalii |
| `tft` | Uproszczony Temporal Fusion Transformer (variable selection + BiLSTM + attention) |

Horyzonty: **1 / 7 / 30** dni (poziom). Walk-forward jak tier I; porównanie skill vs naive względem najlepszego tier I z `results/sgh_usdpln_tier1.json`.

## Dane

| Seria | Loader |
|-------|--------|
| `usdpln` | `experiments/multiasset.py` — NBP tabela A (`usd`) + Yahoo `USDPLN=X` (zweryfikowane 29.10) |

## Wyniki

- `results/sgh_usdpln_tier2.json`
- `results/sgh_usdpln_tier2.md` (skill, DM-p, sekcja tier II vs tier I)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_usdpln_tier2 --cache-only

# smoke
python -m experiments.run_sgh_usdpln_tier2 --demo --max-folds 2

pytest tests/test_sgh_usdpln_tier2.py -q
```

Pełny trening głęboki (większe okna/epoki) — opcjonalnie GPU na sgh-4; konfiguracja YAML celuje w CPU walk-forward.
