# TRD-37 — kredyty konsumpcyjne, tier II (sgh-credit_consumer-2)

Implementacja: `experiments/run_sgh_credit_consumer_tier2.py`, modele w `experiments/sgh/tier2_credit_consumer.py`, panel `experiments/sgh/credit_consumer_panel.py` (`credit_consumer_tier2_features`).

## Modele (tier II)

| Model | Opis |
|-------|------|
| `var` | VAR na poziomie kredytu + stopa NBP + CPI + kredyt hipoteczny |
| `bvar` | BVAR (ridge / Minnesota-lite) na tym samym układzie endog |
| `factor_model` | PCA + AR na czynnikach makro/kredyt, mapowanie na poziom |
| `fci` | Indeks warunków finansowych (z-score stóp, CPI, dynamiki kredytów) + Ridge |
| `lasso` | LASSO na lagach i FCI |
| `elastic_net` | Elastic net |
| `random_forest` | Las losowy |
| `gradient_boosting` | HistGradientBoosting |
| `pko_internal` | Proxy modeli wewnętrznych PKO (blend ECM + FCI + momentum) — nie zastępuje produkcyjnych modeli banku |

Horyzonty: **1 / 3 / 12** miesięcy (poziom stocku). Walk-forward jak w tier I.

## Dane

Jak TRD-36 (NBP statystyka monetarna, GUS CPI, opcjonalnie BIK) — **do weryfikacji**.

## Wyniki

Po uruchomieniu:

- `results/sgh_credit_consumer_tier2.json`
- `results/sgh_credit_consumer_tier2.md` (skill vs RW/naive, DM-p, **porównanie najlepszego tier II vs tier I**)

```bash
pytest tests/test_sgh_credit_consumer_tier2.py -q
python -m experiments.run_sgh_credit_consumer_tier2 --demo --max-folds 2
python -m experiments.run_sgh_credit_consumer_tier2 --cache-only
```

Złożoność (VAR/ML) nie musi poprawiać skill względem tier I — raport zawiera sekcję `tier1_comparison`.
