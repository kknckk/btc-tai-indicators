# TRD-57 — Złoto (GC=F), tier II (sgh-gold-2)

Implementacja: `experiments/run_sgh_gold_tier2.py`, modele w `experiments/sgh/tier2_gold.py`, cechy `gold_tier2_features` w `experiments/sgh/gold_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `timegpt` | Nixtla TimeGPT gdy `NIXTLA_API_KEY` / `TIMEGPT_TOKEN`; inaczej Holt–Winters (ETS) dobrany na walidacji w foldzie |
| `var_lstm` | VAR( p ) na `(gold, gold_lr, gold_vol21)` + LSTM tabular → Ridge na walidacji |
| `tvp_var` | VAR z wagami czasowymi (exp-decay ridge); `p` i `halflife` z walidacji |

Horyzonty: **1 / 7 / 30** dni (poziom GC=F). Walk-forward: `evaluation/walk_forward.py` + `TargetSpec(kind=level, unit=day)`.

Hiperparametry wybierane na końcówce train foldu (`val_frac`), bez dotykania testu. Raport tier II porównuje najlepszy skill vs naive z tier I (`results/sgh_gold_tier1.json`).

## Dane

| Seria | Loader |
|-------|--------|
| `gold` | `experiments/multiasset.py` — Yahoo `GC=F` (zweryfikowane 29.10) |

## Wyniki

Po uruchomieniu eksperymentu liczby trafiają do:

- `results/sgh_gold_tier2.json`
- `results/sgh_gold_tier2.md` (skill vs RW/naive, DM-p, tier II vs tier I)

```bash
python -m experiments.multiasset --check --cache-only
python -m experiments.run_sgh_gold_tier2 --cache-only

# smoke
python -m experiments.run_sgh_gold_tier2 --demo --max-folds 2

pytest tests/test_sgh_gold_tier2.py -q
```

Cięższe warianty deep (np. pełny TimeGPT batch / GPU) można skalować osobnym jobem (sgh-4); tutaj lekkie siatki CPU + opcjonalne API TimeGPT.
