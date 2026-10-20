# AGENT_REPORT — sonnet (warstwa danych + harness ewaluacji + baseline'y)

## Co działa (uruchomione na prawdziwych danych)
- `experiments/data.py`: panel 2009-01-03 → 2026-06-12, 6370 dni × 113 kolumn (67 CSV on-chain + PriceUSD + pochodne: `logret_1`, `rv_7`, `rv_30`, `log_price`, `slog1p_*`). `python -m experiments.data --check` drukuje pokrycie i zapisuje `data/panel.parquet` (ignorowany przez git).
- `evaluation/`: walk-forward (expanding, purge = h, dodatkowy `embargo` z configu, scaler/selekcja tylko w `fit`), metryki (RMSE/MAE/MAPE z ceny/R² OOS vs RW/dir-acc/Sharpe/MaxDD/turnover, koszty 10/20 bps), DM z korektą HLN + macierz par + średnie rangi.
- `experiments/run_baselines.py` → `results/baselines.{json,md}`: 4 modele × h∈{1,7,30}, 158–160 foldów (krok 30 dni), test 2013-04 → 2026-05. Całość ~2 min na CPU.
- Testy: `pytest tests --ignore=tests/test_api_endpoints.py` → 28 passed. Testy behawioralne wstrzykują przyszłość (shift(-1), okno centrowane, standaryzacja po całej próbie) i wymagają wykrycia; test „shuffle-future" permutuje dane po dacie cięcia i wymaga niezmienionych prognoz sprzed cięcia (oraz że wyciekający pipeline się zmienia).

## Ważne znaleziska
- **`data_ingestion/csv/dir_t1.csv` to wyciek**: równa się 1[r_{t+1}>0] (100% zgodności znaku). Zostaje w panelu, ale jest zablokowana w cechach (`make_supervised` rzuca błąd; test wykrywa ją automatycznie). `r_t`, `rv_t`, `rv_7d`, `rv_30d` też wyłączone (duplikaty).
- Puste/krótkie serie (zachowane w panelu, poza cechami): `VelCur1yr_Naive` (0 obs), `CapAct1yrUSD`, `FeeMedNtv`, `SplyAct{30d,1yr,180d}` (1 obs). ~10 serii ma próbkowanie co ~4 dni (NVT*, TxTfrVal*, UTXOCnt, ShareExUSD, BlkSizeMeanByte) → kauzalny ffill ≤7 dni. SOPR/CDD/STH/LTH_MVRV tylko od 2022-06.

## Wyniki (pełna próba OOS; koszt 20 bps; szczegóły w `results/baselines.md`)
| model | h | RMSE | R² vs RW | dir-acc | Sharpe net | DM-p vs naive |
|---|---|---|---|---|---|---|
| naive (B&H) | 1/7/30 | 0.0383/0.1018/0.2390 | 0 | – | 1.07/1.04/1.06 | – |
| xgboost | 1 | 0.0381 | +0.007 | 0.518 | 0.74 | 0.58 |
| xgboost | 7 | 0.1009 | +0.017 | 0.528 | 0.98 | 0.60 |
| xgboost | 30 | 0.2688 | −0.265 | 0.545 | 0.93 | 0.12 |
- Wariancja (QLIKE, niżej = lepiej; naive = zmienność historyczna 30d): GARCH(GJR-t) 1.987 / 0.622 / 0.420 vs naive 2.153 / 0.822 / 0.692 (h=1/7/30); DM-p 0.0000 / 0.0005 / 0.011. HAR-RV 2.071 / 0.761 / 0.550, nieistotny (p 0.19–0.46). Vol-targeting (HAR/GARCH) obniża MaxDD z −0.85 (B&H) do −0.47…−0.70 przy niższym Sharpe (0.81–1.02).
- Wniosek: żaden model nie bije random walk w prognozie zwrotu istotnie statystycznie; XGBoost na h=30 jest gorszy od RW (przeuczenie). Sub-okres od 2018 jest jeszcze słabszy (XGB R² −0.05 dla h=7).

## Braki / ryzyka
- HAR-RV na r² dziennym (brak danych intraday) – RV to szumowy proxy. „naive" w tabeli ekonomicznej = buy&hold (zerowa prognoza nie daje sygnału) – zaznaczone w md.
- Pozycja sign(pred) z h>1 rebalansowana codziennie; wyniki ekonomiczne xgboost zależą od tej reguły. Brak MCS i modelowania slippage'u.
- `purge = h`, `embargo` domyślnie 0 (dodatkowy odstęp); jeśli oczekiwano łącznie h+h, ustaw `walk_forward.embargo` w YAML.
- Progi pokrycia cech (`min_feature_obs=500`) to reguła na dostępności danych, nie na wartościach. Hiperparametry XGB nie są tuningowane.
- `tests/test_api_endpoints.py` nie importuje się (brak `get_econometrics_results` w `api/routers/literature.py`) – błąd istniejący wcześniej, `api/` poza moim zakresem.

## Dalsze kroki
Inne agenty podpinają modele przez `Forecaster` (`fit/predict`, opcjonalnie `predict_vol`) i `run_walk_forward`; predykcje baseline'ów leżą w `results/preds/*.parquet` (lokalnie, niecommitowane) do testów DM. Następnie: MCS, tuning XGB w zagnieżdżonym CV, ablacja cech on-chain vs same zwroty, dodanie SPY/obligacji.
