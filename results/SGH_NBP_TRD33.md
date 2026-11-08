# TRD-33 — stopa referencyjna NBP, tier II (sgh-nbp_rate-2)

Implementacja: `experiments/run_sgh_nbp_tier2.py`, modele w `experiments/sgh/tier2_nbp.py`, panel `experiments/sgh/nbp_panel.py` (`nbp_tier2_features`, opcjonalnie `rpp_communications.csv`).

## Modele (tier II)

| Model | Opis |
|-------|------|
| `var` | VAR( p ) na [stopa, CPI y/y, PMI], prognoza poziomu stopy |
| `bvar` | Shrinkage VAR (Ridge na układzie lagów — Minnesota-lite) |
| `svar` | SVAR z identyfikacją Cholesky (fallback → VAR) |
| `expectations_comm` | Ridge: proxy oczekiwań stopy + ton/komunikacja RPP |
| `random_forest` | Las losowy na lagach makro + ton |
| `gradient_boosting` | HistGradientBoosting |
| `rpp_nlp` | TF–IDF tekstów komunikatów (+ fallback ton/infl) |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stopy). Walk-forward jak tier I.

## Dane

| Seria | Loader |
|-------|--------|
| `nbp_rate` | NBP API (`macro_pl`) |
| `gus_cpi_yoy`, `gus_pmi` | BDL GUS lub `raw/gus_macro.csv` — **do weryfikacji** |
| RPP komunikaty | opcjonalnie `data/macro_pl/raw/rpp_communications.csv` (kolumny `date`, `tone`, `text`) — **do weryfikacji** |

## Wyniki

- `results/sgh_nbp_rate_tier2.json`
- `results/sgh_nbp_rate_tier2.md` (skill vs RW/naive, DM-p, **porównanie najlepszego tier II vs tier I**)

```bash
python -m experiments.macro_pl --check --series nbp_rate --series gus_cpi_yoy --series gus_pmi
python -m experiments.run_sgh_nbp_tier2 --cache-only
python -m experiments.run_sgh_nbp_tier2 --demo --max-folds 2
pytest tests/test_sgh_nbp_tier2.py -q
```

Tier I: `results/SGH_NBP_TRD32.md`, `results/sgh_nbp_rate_tier1.json`.
