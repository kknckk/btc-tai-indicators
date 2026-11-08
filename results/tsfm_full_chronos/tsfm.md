# TSFM zero-shot – wyniki na BTC

- **Tryb:** `real`
- **Model:** `chronos-bolt` (`amazon/chronos-bolt-small`)
- **Dane:** `data_ingestion/csv/PriceUSD.csv` – 5789 obserwacji, 2010-07-18 → 2026-05-23
- **Konfiguracja:** context_len=256, origins/h=300, step=5, device=cpu
- **Czas:** 260.6 s, commit `d7034cb`

## Metryki (punktowo na origin+h)

| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 1770.89 | 1172.67 | 2.14 | 2.13 | 53.0% | 46.0% | 1653.45 | -7.10% | -14.71% | 86.7% |
| 7 | 300 | 4212.74 | 2907.75 | 5.34 | 5.27 | 53.7% | 50.3% | 4014.72 | -4.93% | -10.11% | 79.7% |
| 30 | 300 | 9041.35 | 6613.98 | 12.64 | 12.47 | 52.7% | 45.3% | 8731.45 | -3.55% | -7.22% | 77.0% |

## Agregat

- Średnia DirAcc: **53.1%** (RW-momentum: 47.2%), średni RMSE skill vs RW: **-5.19%**
- Interpretacja: dodatni `skill vs RW` = model bije random walk; ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline kierunku (znak ostatniej dziennej zmiany).
