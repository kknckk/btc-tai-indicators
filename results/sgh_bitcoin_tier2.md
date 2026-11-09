# TRD-59 — Bitcoin (PriceUSD + on-chain), tier II

Seria: `bitcoin` · panel 2009-01-03 → 2026-06-12 (6370 dni)

Modele: LSTM (okna tabularne), TFT (variable selection + BiLSTM + attention), SVM (RBF SVR na cechach on-chain), HellFormer (multi-scale heat conv + transformer). Harness: `evaluation/walk_forward.py`; dane: `experiments/data.py`.

| h (dni) | model | RMSE | skill vs RW | skill vs naive | DM-p vs naive | DM-p vs RW |
|---:|---|---:|---:|---:|---:|---:|
| 1 | svm | 38367.6015 | -1296.4557 | -672.2799 | 0.0000 | 0.0000 |
| 1 | lstm | 1065.1686 | 0.0000 | 0.4811 | 0.0000 | n/a |
| 1 | tft | 1065.1686 | 0.0000 | 0.4811 | 0.0000 | n/a |
| 1 | hellformer | 1065.1686 | 0.0000 | 0.4811 | 0.0000 | n/a |
| 7 | svm | 38417.5006 | -193.7851 | -168.5606 | 0.0000 | 0.0000 |
| 7 | lstm | 2752.6518 | 0.0000 | 0.1295 | 0.0000 | n/a |
| 7 | tft | 2752.6518 | 0.0000 | 0.1295 | 0.0000 | n/a |
| 7 | hellformer | 2752.6518 | 0.0000 | 0.1295 | 0.0000 | n/a |
| 30 | svm | 38605.1026 | -40.9001 | -39.3038 | 0.0000 | 0.0000 |
| 30 | lstm | 5963.9956 | 0.0000 | 0.0381 | 0.0001 | n/a |
| 30 | tft | 5963.9956 | 0.0000 | 0.0381 | 0.0001 | n/a |
| 30 | hellformer | 5963.9956 | 0.0000 | 0.0381 | 0.0001 | n/a |

## Tier II vs tier I (skill vs naive)

- h=1: najlepszy tier II `lstm` (skill 0.4811) vs tier I `None` (skill n/a) — tier II lepszy: **n/a (brak tier I)**
- h=7: najlepszy tier II `lstm` (skill 0.1295) vs tier I `None` (skill n/a) — tier II lepszy: **n/a (brak tier I)**
- h=30: najlepszy tier II `lstm` (skill 0.0381) vs tier I `None` (skill n/a) — tier II lepszy: **n/a (brak tier I)**

Skill = 1 − MSE_model/MSE_benchmark. Większa złożoność (deep/TFT/transformer/SVM) nie gwarantuje poprawy względem tier I; sekcja powyżej porównuje najlepszy skill vs naive.

Uruchomienie:
```bash
python -m experiments.data --check
python -m experiments.run_sgh_bitcoin_tier2
pytest tests/test_sgh_bitcoin_tier2.py -q
```
