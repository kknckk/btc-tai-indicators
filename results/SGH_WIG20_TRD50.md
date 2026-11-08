# TRD-50 — WIG20, tier I (sgh-wig20-1)

Implementacja: `experiments/run_sgh_wig20_tier1.py`, modele w `experiments/sgh/tier1_wig20.py`, panel `experiments/sgh/wig20_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `garch` | GARCH(1,1) na log-zwrotach (×100); prognoza średniej skumulowanej → poziom indeksu |
| `egarch` | EGARCH(1,1) — ten sam schemat prognozy poziomu |
| `gjr_garch` | GJR-GARCH(1,1,1) — asymetria w wariancji |
| `event_study` | Dni z \|r\| powyżej kwantyla (walidacja w foldzie); inaczej RW (poziom = cena t) |

Horyzonty: **1 / 7 / 30** dni (poziom WIG20). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry (mean AR vs stała, rozkład normal/t, próg event study): wybór na końcówce train foldu (`val_frac`), bez dotykania testu.

## Dane

| Seria | Loader |
|-------|--------|
| `wig20` | `experiments/multiasset.py` — Yahoo `WIG20.WA` (zweryfikowane 29.10; stooq blokuje boty) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_wig20_tier1.json`
- `results/sgh_wig20_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_wig20_tier1 --cache-only

# smoke
python -m experiments.run_sgh_wig20_tier1 --demo --max-folds 2

pytest tests/test_sgh_wig20_tier1.py -q
```

Tier II (TRD-51): `experiments/run_sgh_wig20_tier2.py`, `results/SGH_WIG20_TRD51.md`.
