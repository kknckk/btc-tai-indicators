# TRD-43 — depozyty osób prywatnych, tier II (sgh-deposits_households-2)

Implementacja: `experiments/run_sgh_deposits_households_tier2.py`, modele w `experiments/sgh/tier2_deposits_households.py`, cechy tier II w `experiments/sgh/deposits_households_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `var` | VAR na depozytach, stopie NBP, CPI, kredycie konsumpcyjnym |
| `bvar` | BVAR (ridge) — ten sam układ endog |
| `portfolio_model` | Proxy portfela: PCA + AR na systemie depozyt/kredyt/stopy |
| `segmentation_model` | Segmentacja: osobne Ridge w reżimie wysokiej vs niskiej stopy NBP |
| `pko_internal` | Proxy modeli PKO: ważona mieszanka ECM + portfel + retencja |
| `random_forest` | Random forest na lagach makro |
| `gradient_boosting` | HistGradientBoosting |
| `retention_model` | Proxy retencji: Ridge na lagach persistencji depozytów |

Horyzonty: **1 / 3 / 12** miesięcy. Raport porównuje najlepszy skill vs naive tier II z tier I (`results/sgh_deposits_households_tier1.json`).

## Dane

NBP statystyka monetarna + makro jak w tier I — **do weryfikacji**.

## Uruchomienie

```bash
pytest tests/test_sgh_deposits_households_tier2.py -q
python -m experiments.run_sgh_deposits_households_tier2 --demo --max-folds 2
python -m experiments.run_sgh_deposits_households_tier2 --cache-only
```

Wyniki: `results/sgh_deposits_households_tier2.json`, `results/sgh_deposits_households_tier2.md`.
