# BTC Indicators Documentation

Ten plik zawiera pełną listę wskaźników On-Chain używanych w projekcie, ze wskazaniem źródła ich pochodzenia, zalecanego interwału odświeżania w środowisku produkcyjnym (GCP) oraz linkiem do wygenerowanych plików CSV.

## Struktura plików CSV
Każdy plik CSV w folderze `data_ingestion/csv/` zawiera zawsze dwie kolumny:
- `time` - data w formacie `YYYY-MM-DD`
- `[Nazwa_Wskaznika]` - wartość numeryczna wskaźnika dla danego dnia

Dane w plikach CSV obejmują pełną historię (od początku rejestracji wskaźnika) do dnia **2026-06-01**.

## Lista Wskaźników

| Wskaźnik | Opis | Źródło Danych | Interwał Workera (GCP) | Plik CSV (do 2026-06-01) |
|---|---|---|---|---|
| **CapRealUSD** | Realized Capitalization (USD). Wycena UTXO po cenie w momencie ich powstania. | CoinMetrics | 1D (Dziennie) | [csv/CapRealUSD.csv](data_ingestion/csv/CapRealUSD.csv) |
| **SplyAct30d** | Active Supply 30d. Ilość BTC, która zmieniła adres w ciągu ostatnich 30 dni. | CoinMetrics | 1D (Dziennie) | [csv/SplyAct30d.csv](data_ingestion/csv/SplyAct30d.csv) |
| **SplyAct180d** | Active Supply 180d. | CoinMetrics | 1D (Dziennie) | [csv/SplyAct180d.csv](data_ingestion/csv/SplyAct180d.csv) |
| **SplyAct1yr** | Active Supply 1y. | CoinMetrics | 1D (Dziennie) | [csv/SplyAct1yr.csv](data_ingestion/csv/SplyAct1yr.csv) |
| **TxTfrValDayDst** | Coin Days Destroyed (CDD). Suma (wiek_utxo_w_dniach * wartość_btc). | BigQuery (SQL) | 1D (Dziennie) | *Do wyliczenia via BQ* |
| **FeeMedNtv** | Median Fee (BTC). | mempool.space | Per-Block -> Agg do 1D | *Czeka na pobranie* |
| **FeeMedUSD** | Median Fee (USD). | mempool.space / BQ | Per-Block -> Agg do 1D | *Czeka na pobranie* |
| **TxTfrValAdjNtv** | Adjusted Transfer Volume (BTC). Usuwa tzw. change outputs. | CoinMetrics | 1D (Dziennie) | [csv/TxTfrValAdjNtv.csv](data_ingestion/csv/TxTfrValAdjNtv.csv) |
| **TxTfrValAdjUSD** | Adjusted Transfer Volume (USD). | CoinMetrics | 1D (Dziennie) | [csv/TxTfrValAdjUSD.csv](data_ingestion/csv/TxTfrValAdjUSD.csv) |
| **CapAct1yrUSD** | Active Cap 1y (USD). Realized Cap tylko dla monet aktywnych w < 1y. | CoinMetrics | 1D (Dziennie) | [csv/CapAct1yrUSD.csv](data_ingestion/csv/CapAct1yrUSD.csv) |
| **SOPR** | Spent Output Profit Ratio. Zysk/strata na przenoszonych monetach. | Wyliczane / BQ | 1D (Dziennie) | *Do wyliczenia via BQ* |
| **NUPL** | Net Unrealized Profit/Loss. | Wyliczane (MVRV pochodna)| 1D (Dziennie) | *Do wyliczenia via BQ* |
| **MVRV_Z** | MVRV Z-Score. Odchylenie standardowe Market Cap od Realized Cap. | Wyliczane z CSV | 1D (Dziennie) | *Do wyliczenia via BQ* |
| **RealizedPrice** | CapRealUSD / SplyCur | Wyliczane z CSV | 1D (Dziennie) | *Do wyliczenia via BQ* |
| **PuellMultiple** | Przychód z wydobycia / 365d_MA(Przychód). | Wyliczane z CSV | 1D (Dziennie) | *Do wyliczenia via BQ* |
| **ExchangeFlowIn** | Napływ na giełdy. Wymaga clusteringu adresów. | Zewn. API (np. Dune)| 1D (Dziennie) | *Brak otwartego darmowego źródła* |
| **ExchangeFlowOut**| Odpływ z giełd. | Zewn. API (np. Dune)| 1D (Dziennie) | *Brak otwartego darmowego źródła* |
| **AdrActCnt** | Liczba aktywnych adresów (wysyłające + odbierające). | CoinMetrics | 1D (Dziennie) | [csv/AdrActCnt.csv](data_ingestion/csv/AdrActCnt.csv) |
| **AdrBalCnt** | Liczba adresów z niezerowym balansem. | CoinMetrics | 1D (Dziennie) | [csv/AdrBalCnt.csv](data_ingestion/csv/AdrBalCnt.csv) |
| **BlkCnt** | Liczba bloków wydobytych dziennie. | CoinMetrics | 1D (Dziennie) | [csv/BlkCnt.csv](data_ingestion/csv/BlkCnt.csv) |
| **BlkSizeMeanByte**| Średni rozmiar bloku (bajty). | CoinMetrics | 1D (Dziennie) | [csv/BlkSizeMeanByte.csv](data_ingestion/csv/BlkSizeMeanByte.csv) |
| **UTXOCnt** | Całkowita liczba niespendowanych wyjść. | CoinMetrics | 1D (Dziennie) | [csv/UTXOCnt.csv](data_ingestion/csv/UTXOCnt.csv) |
| **CapMrktCurUSD** | Bieżąca kapitalizacja rynkowa (USD). | CoinMetrics | 1D (Dziennie) | [csv/CapMrktCurUSD.csv](data_ingestion/csv/CapMrktCurUSD.csv) |
| **DiffMean** | Średnia trudność wydobycia. | CoinMetrics | 1D (Dziennie) | [csv/DiffMean.csv](data_ingestion/csv/DiffMean.csv) |
| **HashRate** | Szacunkowy hashrate. | CoinMetrics | 1D (Dziennie) | [csv/HashRate.csv](data_ingestion/csv/HashRate.csv) |
| **SplyCur** | Bieżąca podaż BTC w obiegu. | CoinMetrics | 1D (Dziennie) | [csv/SplyCur.csv](data_ingestion/csv/SplyCur.csv) |
| **TxCnt** | Liczba potwierdzonych transakcji dziennie. | CoinMetrics | 1D (Dziennie) | [csv/TxCnt.csv](data_ingestion/csv/TxCnt.csv) |
| **FeeTotNtv** | Całkowita suma opłat (BTC). | CoinMetrics | 1D (Dziennie) | [csv/FeeTotNtv.csv](data_ingestion/csv/FeeTotNtv.csv) |
| **FeeTotUSD** | Całkowita suma opłat (USD). | CoinMetrics | 1D (Dziennie) | [csv/FeeTotUSD.csv](data_ingestion/csv/FeeTotUSD.csv) |

## Uwagi
- Wskaźniki pochodne ("DERIVED") takie jak MVRV Z-Score, Puell Multiple, czy Realized Price są w architekturze GCP wyliczane w locie za pomocą zapytań SQL z wykorzystaniem metryk podstawowych (takich jak `CapRealUSD` czy `SplyCur`).
- CSV dla wskaźników podstawowych zostało wygenerowane w folderze `data_ingestion/csv` i zatrzymuje się na dacie `2026-06-01`. Następne pobrania dopisują wiersze codziennie poprzez Cloud Functions.
