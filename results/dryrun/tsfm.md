# TSFM zero-shot – wyniki na BTC

- **Tryb:** `dry-run` (ATRAPA – dane nie są wynikiem modelu TSFM)
- **Model:** `stub` (`None`)
- **Dane:** `data_ingestion/csv/PriceUSD.csv` – 5789 obserwacji, 2010-07-18 → 2026-05-23
- **Konfiguracja:** context_len=256, origins/h=40, step=1, device=cpu
- **Czas:** 1.2 s, commit `22de988`

## Metryki (punktowo na origin+h)

| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 1149.49 | 953.97 | 1.22 | 1.22 | 0.0% | 55.0% | 1149.49 | +0.00% | +0.00% | 100.0% |
| 7 | 40 | 2876.24 | 2517.81 | 3.23 | 3.25 | 0.0% | 55.0% | 2876.24 | +0.00% | +0.00% | 100.0% |
| 30 | 40 | 7728.69 | 6705.30 | 8.53 | 9.04 | 0.0% | 52.5% | 7728.69 | +0.00% | +0.00% | 100.0% |

## Agregat

- Średnia DirAcc: **0.0%** (RW-momentum: 54.2%), średni RMSE skill vs RW: **+0.00%**
- Interpretacja: dodatni `skill vs RW` = model bije random walk; ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline kierunku (znak ostatniej dziennej zmiany).

> **UWAGA:** tryb `--dry-run` używa atrapy `stub` (random walk + empiryczne kwantyle log-zwrotów). To test instalacji pipeline'u, **nie** wynik modelu TSFM.

## Ostrzeżenia

- --dry-run nadpisał model 'chronos-bolt' atrapą 'stub'
