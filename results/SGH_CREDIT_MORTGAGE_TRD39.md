# TRD-39 — kredyty hipoteczne, tier II (sgh-credit_mortgage-2)

Implementacja: `experiments/run_sgh_credit_mortgage_tier2.py`, modele w `experiments/sgh/tier2_credit_mortgage.py`, cechy tier II w `experiments/sgh/credit_mortgage_panel.py`.

## Modele (tier II)

| Model | Opis |
|-------|------|
| `var` | VAR — układ kredyt hip. + stopa NBP + CPI + kredyt kons. (p z siatki, walidacja w foldzie) |
| `panel_model` | Panel makro — Ridge na udziałach sektorów (hipoteka/kons./depozyty) i lagach |
| `mortgage_contract` | Kontrakt hipoteczny (proxy) — stopy, zmiana stopy, affordability, momentum stocku |
| `random_forest` | Random forest na lagach tabularnych |
| `gradient_boosting` | HistGradientBoosting |
| `lstm` | LSTM na oknach cech (`mortgage_now` + lags); lekkie siatki CPU |
| `pko_internal` | Proxy modeli PKO — blend ARDL + regresja dynamiczna + momentum |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku). Walk-forward jak tier I. Porównanie skill vs naive względem najlepszego tier I w `tier1_comparison`.

Tier I: `results/SGH_CREDIT_MORTGAGE_TRD38.md`.

## Dane

Jak TRD-38 (NBP monetarna, makro); AMRON/SARFiN — **do weryfikacji** (brak w loaderze).

## Wyniki

- `results/sgh_credit_mortgage_tier2.json`
- `results/sgh_credit_mortgage_tier2.md`

```bash
pytest tests/test_sgh_credit_mortgage_tier2.py -q
python -m experiments.run_sgh_credit_mortgage_tier2 --demo --max-folds 2
python -m experiments.macro_pl --check --series credit_mortgage --cache-only
python -m experiments.run_sgh_credit_mortgage_tier2 --cache-only
```

Pełne LSTM/RNN na GPU — osobny job (sgh-4), nie w tym zadaniu.
