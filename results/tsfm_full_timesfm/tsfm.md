# TSFM zero-shot – wyniki na BTC

- **Tryb:** `real`
- **Model:** `timesfm` (`google/timesfm-2.5-200m-pytorch`)
- **Dane:** `data_ingestion/csv/PriceUSD.csv` – 5789 obserwacji, 2010-07-18 → 2026-05-23
- **Konfiguracja:** context_len=256, origins/h=300, step=5, device=cpu
- **Czas:** 2775.9 s, commit `d7034cb`

## Metryki (punktowo na origin+h)

| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 1704.80 | 1152.52 | 2.01 | 2.01 | 48.3% | 46.0% | 1653.45 | -3.11% | -6.31% | 59.0% |
| 7 | 300 | 4369.53 | 3148.69 | 5.74 | 5.70 | 43.0% | 50.3% | 4014.72 | -8.84% | -18.46% | 54.3% |
| 30 | 300 | 10048.32 | 7187.14 | 13.82 | 13.78 | 44.7% | 45.3% | 8731.45 | -15.08% | -32.44% | 48.7% |

## Agregat

- Średnia DirAcc: **45.3%** (RW-momentum: 47.2%), średni RMSE skill vs RW: **-9.01%**
- Interpretacja: dodatni `skill vs RW` = model bije random walk; ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline kierunku (znak ostatniej dziennej zmiany).
