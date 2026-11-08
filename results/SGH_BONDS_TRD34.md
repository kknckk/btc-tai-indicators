# TRD-34 — rentowności obligacji PL 2/5/7/10Y, tier I (sgh-bonds_pl-1)

Implementacja: `experiments/run_sgh_bonds_tier1.py`, modele w `experiments/sgh/tier1_bonds.py`, panel `experiments/sgh/bonds_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `nelson_siegel` | Czynniki NS z krzywej 2/5/7/10Y, AR(1), λ w siatce na walidacji w foldzie |
| `svensson` | Rozszerzenie Svenssona (4 czynniki), siatka λ₁/λ₂ |
| `diebold_li` | NS + dynamika czynników (λ jak w literaturze DL, m.in. 0.0609) |

Horyzonty: **1 / 7 / 30** dni (poziom rentowności per tenor). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

## Dane

| Seria | Loader |
|-------|--------|
| `bond_pl_2y` … `bond_pl_10y` | `fetch_bonds_pl_yields` → CSV MF gov.pl — **do weryfikacji**; **GPW Benchmark SPW = komercyjny** |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_bonds_pl_tier1.json`
- `results/sgh_bonds_pl_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
python -m experiments.macro_pl --check --series bond_pl_2y --series bond_pl_10y
python -m experiments.run_sgh_bonds_tier1 --cache-only

# smoke
python -m experiments.run_sgh_bonds_tier1 --demo --max-folds 2

pytest tests/test_sgh_bonds_tier1.py -q
```

Tier II (TRD-35): PCA/VAR/LSTM itd. — `results/SGH_BONDS_TRD35.md`, `sgh-bonds_pl-2`.
