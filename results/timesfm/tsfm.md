# TSFM zero-shot – wyniki na BTC

- **Tryb:** `real`
- **Model:** `timesfm` (`google/timesfm-2.5-200m-pytorch`)
- **Dane:** `data_ingestion/csv/PriceUSD.csv` – 5789 obserwacji, 2010-07-18 → 2026-05-23
- **Konfiguracja:** context_len=256, origins/h=40, step=1, device=cpu
- **Czas:** 111.4 s, commit `ee6e415`

## Metryki (punktowo na origin+h)

| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 1309.06 | 1042.28 | 1.34 | 1.33 | 52.5% | 55.0% | 1149.49 | -13.88% | -29.69% | 62.5% |
| 7 | 40 | 2831.36 | 2372.04 | 3.05 | 3.03 | 67.5% | 55.0% | 2876.24 | +1.56% | +3.10% | 62.5% |
| 30 | 40 | 5883.81 | 5175.32 | 6.66 | 6.59 | 95.0% | 52.5% | 7728.69 | +23.87% | +42.04% | 67.5% |

## Agregat

- Średnia DirAcc: **71.7%** (RW-momentum: 54.2%), średni RMSE skill vs RW: **+3.85%**
- Interpretacja: dodatni `skill vs RW` = model bije random walk; ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline kierunku (znak ostatniej dziennej zmiany).
