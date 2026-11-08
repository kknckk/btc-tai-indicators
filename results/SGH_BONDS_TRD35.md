# TRD-35 — rentowności obligacji PL 2/5/7/10Y, tier II (sgh-bonds_pl-2)

Implementacja: `experiments/run_sgh_bonds_tier2.py`, modele w `experiments/sgh/tier2_bonds.py`, panel `experiments/sgh/bonds_panel.py` (`bonds_tier2_features`, makro `nbp_rate` / `gus_cpi_yoy`).

## Modele (tier II)

| Model | Opis |
|-------|------|
| `pca_curve` | PCA(3) na krzywej 2/5/7/10Y, AR(1) na składowych, rekonstrukcja tenor docelowy |
| `var` | VAR( p ) na czynnikach Nelson–Siegel |
| `bvar` | Ridge-VAR (Minnesota-lite) na czynnikach NS |
| `varx` | Ridge-VARX: czynniki NS + makro (stopa NBP, CPI y/y) |
| `macro_finance` | Ridge: poziom/spread/krzywa + makro |
| `afns` | NS + VAR na czynnikach (AFNS-lite), siatka λ |
| `pca_lstm` | LSTM na sekwencjach składowych PCA |
| `lstm` | LSTM na lagach rentowności (jak tier I commodities) |

Horyzonty: **1 / 7 / 30** dni (poziom per tenor). Walk-forward jak tier I. Pełne LSTM/PCA-LSTM na GPU można skalować osobnym jobem (sgh-4); tutaj lekkie siatki CPU.

## Dane

| Seria | Loader |
|-------|--------|
| `bond_pl_2y` … `bond_pl_10y` | `fetch_bonds_pl_yields` / MF — **do weryfikacji** |
| `nbp_rate`, `gus_cpi_yoy` | `macro_pl` (ffill do dni roboczych) — **do weryfikacji** |

## Wyniki

- `results/sgh_bonds_pl_tier2.json`
- `results/sgh_bonds_pl_tier2.md` (skill vs RW/naive, DM-p, **porównanie najlepszego tier II vs tier I** per tenor)

```bash
python -m experiments.macro_pl --check --series bond_pl_2y --series bond_pl_10y --series nbp_rate
python -m experiments.run_sgh_bonds_tier2 --cache-only

# smoke
python -m experiments.run_sgh_bonds_tier2 --demo --max-folds 2

pytest tests/test_sgh_bonds_tier2.py -q
```

Tier I: `results/SGH_BONDS_TRD34.md`, `results/sgh_bonds_pl_tier1.json`.
