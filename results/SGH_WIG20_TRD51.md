# TRD-51 — WIG20, tier II (sgh-wig20-2)

Implementacja: `experiments/run_sgh_wig20_tier2.py`, modele w `experiments/sgh/tier2_wig20.py`, cechy w `experiments/sgh/wig20_panel.py` (`wig20_tier2_features`).

## Modele (tier II)

| Model | Opis |
|-------|------|
| `har_rv` | Ridge na składowych HAR (rv_d, rv_w, rv_m) + lag poziomu |
| `realized_vol` | Ridge na sumach r², bipower, trailing vol |
| `external_factors` | Ridge: WIG20 + lagi log-zwrotów USD/PLN, EUR/PLN, ropy, złota |
| `stochastic_vol` | AR(1) na log-RV + drift momentum → poziom h-dni |
| `sentiment` | Ridge na proxy fear/stress/momentum (z zwrotów i vol) |
| `random_forest` | Las losowy na pełnym zestawie cech tier II |
| `gradient_boosting` | HistGradientBoosting na tym samym zestawie |

Horyzonty: **1 / 7 / 30** dni (poziom WIG20). Walk-forward jak tier I; hiperparametry (ridge α, φ SV, RF/GB) wybór na końcówce train foldu (`val_frac`).

## Dane

| Seria | Loader |
|-------|--------|
| `wig20` | `experiments/multiasset.py` — Yahoo `WIG20.WA` |
| `usdpln`, `eurpln`, `oil`, `gold` | opcjonalnie dołączane w `build_wig20_daily_panel(with_externals=True)` |

## Wyniki

Po uruchomieniu:

- `results/sgh_wig20_tier2.json`
- `results/sgh_wig20_tier2.md` (skill vs RW/naive, DM-p, **porównanie z tier I**)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_wig20_tier2 --cache-only

# smoke
python -m experiments.run_sgh_wig20_tier2 --demo --max-folds 2

pytest tests/test_sgh_wig20_tier2.py -q
```

Tier I: TRD-50 (`run_sgh_wig20_tier1`).
