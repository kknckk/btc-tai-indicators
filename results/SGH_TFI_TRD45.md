# TRD-45 — aktywa TFI, tier II (sgh-tfi_assets-2)

Implementacja: `experiments/run_sgh_tfi_assets_tier2.py`, modele w `experiments/sgh/tier2_tfi_assets.py`, cechy tier II w `experiments/sgh/tfi_assets_panel.py` (`tfi_assets_tier2_features`).

## Modele (tier II)

| Model | Opis |
|-------|------|
| `portfolio_model` | PCA + AR na systemie TFI/stopy/CPI/depozyty/kredyt/obligacje → poziom aktywów |
| `relative_attractiveness` | Ridge na spreadach obligacje–depozyty/stopy vs opóźniona performance TFI |
| `segmentation_model` | Ridge w reżimach wysoki vs niski implied net flow |
| `midas` | Ridge na MIDAS-owej agregacji lagów mom TFI (wagi Beta) |
| `mixed_frequency` | Ridge: lag miesięczne TFI + kwartalne/roczne CPI |
| `sentiment_model` | TF-IDF na tekstowym proxy (z-score flow/perf/sentyment) |
| `random_forest` | Las losowy na lagach makro/flow |
| `gradient_boosting` | HistGradientBoosting na tym samym zestawie cech |

Horyzonty: **1 / 3 / 12** miesięcy (poziom aktywów netto TFI). Walk-forward jak tier I.

## Dane

Jak TRD-44 (IZFiA / NBP / GUS) — **do weryfikacji**. Brak dedykowanego sentymentu rynkowego TFI w cache; model sentymentu używa proxy makro/flow.

## Wyniki

- `results/sgh_tfi_assets_tier2.json`
- `results/sgh_tfi_assets_tier2.md` (skill vs RW/naive, DM, **porównanie z tier I**)

```bash
pytest tests/test_sgh_tfi_assets_tier2.py -q
python -m experiments.run_sgh_tfi_assets_tier2 --demo --max-folds 2
python -m experiments.run_sgh_tfi_assets_tier2 --cache-only
```

Złożoność tier II nie musi poprawiać skill vs naive względem tier I — raport MD zawiera flagę `tier2_beats_tier1_skill` per horyzont (gdy istnieje `sgh_tfi_assets_tier1.json`).
