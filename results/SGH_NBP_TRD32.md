# TRD-32 — stopa referencyjna NBP, tier I (sgh-nbp_rate-1)

Implementacja: `experiments/run_sgh_nbp_tier1.py`, modele w `experiments/sgh/tier1_nbp.py`, panel `experiments/sgh/nbp_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `taylor_rule` | Reguła Taylora / funkcja reakcji (CPI y/y, PMI vs cel inflacyjny 2,5%) |
| `forward_linear` | Ridge na lagach stopy, CPI, PMI |
| `ordered_logit` | Ordered logit na kategoriach zmiany stopy → oczekiwany poziom |
| `ordered_probit` | j.w., link probit |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stopy). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=month)`.

## Dane

| Seria | Loader |
|-------|--------|
| `nbp_rate` | NBP API (`macro_pl`) |
| `gus_cpi_yoy`, `gus_pmi` | BDL GUS lub `data/macro_pl/raw/gus_macro.csv` — **do weryfikacji** |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_nbp_rate_tier1.json`
- `results/sgh_nbp_rate_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
# dane
python -m experiments.macro_pl --check --series nbp_rate --series gus_cpi_yoy --series gus_pmi

# produkcyjnie (cache)
python -m experiments.run_sgh_nbp_tier1 --cache-only

# smoke (panel syntetyczny — nie do raportu PKO)
python -m experiments.run_sgh_nbp_tier1 --demo --max-folds 2

pytest tests/test_sgh_nbp_tier1.py -q
```

Raport porównawczy serii (TRD-31) czyta `results/sgh_nbp_rate_tier1.json`.

Tier II (TRD-33): `results/SGH_NBP_TRD33.md`, `results/sgh_nbp_rate_tier2.json`.
