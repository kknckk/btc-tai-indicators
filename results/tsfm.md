# TSFM zero-shot – wyniki na BTC

- **Tryb:** `real`
- **Model:** `chronos-bolt` (`amazon/chronos-bolt-small`)
- **Dane:** `data_ingestion/csv/PriceUSD.csv` – 5789 obserwacji, 2010-07-18 → 2026-05-23
- **Konfiguracja:** context_len=256, origins/h=40, step=1, device=cpu
- **Czas:** 13.1 s, commit `22de988`

## Metryki (punktowo na origin+h)

| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 1372.32 | 1145.59 | 1.47 | 1.47 | 40.0% | 55.0% | 1149.49 | -19.39% | -42.53% | 92.5% |
| 7 | 40 | 3006.73 | 2587.69 | 3.33 | 3.35 | 47.5% | 55.0% | 2876.24 | -4.54% | -9.28% | 100.0% |
| 30 | 40 | 9314.38 | 7698.63 | 9.83 | 10.58 | 52.5% | 52.5% | 7728.69 | -20.52% | -45.24% | 72.5% |

## Agregat

- Średnia DirAcc: **46.7%** (RW-momentum: 54.2%), średni RMSE skill vs RW: **-14.81%**
- Interpretacja: dodatni `skill vs RW` = model bije random walk; ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline kierunku (znak ostatniej dziennej zmiany).
