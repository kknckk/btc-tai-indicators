# TSFM zero-shot – wyniki na BTC

- **Tryb:** `real`
- **Model:** `chronos-bolt` (`amazon/chronos-bolt-small`)
- **Dane:** `api/csv/PriceUSD.csv` – 5789 obserwacji, 2010-07-18 → 2026-05-23
- **Konfiguracja:** context_len=256, origins/h=60, step=1, device=cpu
- **Czas:** 9.0 s, commit `None`

## Metryki (punktowo na origin+h)

| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 1503.39 | 1211.95 | 1.62 | 1.63 | 43.3% | 48.3% | 1346.79 | -11.63% | -24.61% | 88.3% |
| 7 | 60 | 3341.63 | 2911.50 | 3.92 | 3.95 | 41.7% | 53.3% | 3088.16 | -8.21% | -17.09% | 95.0% |
| 30 | 60 | 8044.13 | 6358.43 | 8.35 | 8.81 | 55.0% | 48.3% | 6567.22 | -22.49% | -50.04% | 81.7% |

## Agregat

- Średnia DirAcc: **46.7%** (RW-momentum: 50.0%), średni RMSE skill vs RW: **-14.11%**
- Interpretacja: dodatni `skill vs RW` = model bije random walk; ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline kierunku (znak ostatniej dziennej zmiany).
