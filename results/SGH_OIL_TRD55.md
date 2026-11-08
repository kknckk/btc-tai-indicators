# TRD-55 — Ropa WTI (futures), tier II (sgh-oil-2)

Implementacja: `experiments/run_sgh_oil_tier2.py`, modele w `experiments/sgh/tier2_oil.py`, cechy `oil_tier2_features` w `experiments/sgh/oil_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `emd` | Kausalna dekompozycja MA-IMF (HF/MF/LF + trend 63d) → Ridge na każdym kanale → meta Ridge (α z walidacji) |
| `stacking` | Ensembl tier I: LSTM-SVR, XGBoost, GARCH-MIDAS → Ridge meta na końcówce train foldu |
| `sparse_ae` | Okna cech tier II, sparse autoencoder (L1 na latent) + głowica poziomu |

Horyzonty: **1 / 7 / 30** dni (poziom CL=F). Walk-forward jak tier I; porównanie skill vs naive względem najlepszego tier I z `results/sgh_oil_tier1.json`.

## Dane

| Seria | Loader |
|-------|--------|
| `oil` | `experiments/multiasset.py` — Yahoo `CL=F` (zweryfikowane 29.10) |

## Wyniki

- `results/sgh_oil_tier2.json`
- `results/sgh_oil_tier2.md` (skill, DM-p, sekcja tier II vs tier I)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_oil_tier2 --cache-only

# smoke
python -m experiments.run_sgh_oil_tier2 --demo --max-folds 2

pytest tests/test_sgh_oil_tier2.py -q
```

Pełny trening głęboki (większe okna/epoki) — opcjonalnie GPU na sgh-4; konfiguracja YAML celuje w CPU walk-forward.
