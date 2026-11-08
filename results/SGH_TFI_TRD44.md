# TRD-44 — aktywa TFI, tier I (sgh-tfi_assets-1)

Implementacja: `experiments/run_sgh_tfi_assets_tier1.py`, modele w `experiments/sgh/tier1_tfi_assets.py`, panel `experiments/sgh/tfi_assets_panel.py`.

## Modele (tier I)

| Model | Opis |
|-------|------|
| `flow_performance` | Poziom aktywów vs opóźniona „performance” (rentowność 10Y / depozyty) i implied net flow (ΔA − perf) |
| `panel_regression` | OLS na makro-panelu (NBP, CPI, depozyty, kredyt, obligacje) + opóźnienia poziomu TFI |
| `dynamic_regression` | Regresja dynamiczna — Ridge na wszystkich lagach cech |

Horyzonty: **1 / 3 / 12** miesięcy (poziom aktywów netto TFI). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=month)`.

## Dane

| Seria | Loader |
|-------|--------|
| `tfi_assets` | IZFiA XLS (`macro_pl`, `scripts/macro_pl/fetch_tfi_assets.py`) — **do weryfikacji** |
| `nbp_rate`, `gus_cpi_yoy` | NBP API + GUS — **do weryfikacji** |
| `deposits_households`, `credit_consumer`, `bond_pl_10y` | opcjonalne makro (cache NBP / MF) |

Analizy.pl = płatne; brak automatu — patrz `data/macro_pl/SOURCES.md`.

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_tfi_assets_tier1.json`
- `results/sgh_tfi_assets_tier1.md` (skill vs RW/naive, DM-p vs naive)

```bash
pytest tests/test_sgh_tfi_assets_tier1.py -q
python -m experiments.run_sgh_tfi_assets_tier1 --demo --max-folds 2
python -m experiments.macro_pl --check --series tfi_assets --cache-only
python -m experiments.run_sgh_tfi_assets_tier1 --cache-only
```
