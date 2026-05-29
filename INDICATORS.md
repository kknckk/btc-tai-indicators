# BTC Indicators Documentation

Ten plik zawiera pełną listę wskaźników On-Chain używanych w projekcie, ze wskazaniem źródła ich pochodzenia, zalecanego interwału odświeżania w środowisku produkcyjnym (GCP) oraz linkiem do wygenerowanych plików CSV.

## Struktura plików CSV
Każdy plik CSV w folderze `data_ingestion/csv/` zawiera zawsze dwie kolumny:
- `time` - data w formacie `YYYY-MM-DD`
- `[Nazwa_Wskaznika]` - wartość numeryczna wskaźnika dla danego dnia

Dane w plikach CSV obejmują pełną historię (od początku sieci) do dnia **2026-06-01**.

## Wygenerowane Wskaźniki (Gotowe w repozytorium jako CSV)

Poniższe 38 wskaźników zostało skutecznie pobranych i wygenerowanych (Faza 0 oraz Faza 1):
1. Podstawowe z **CoinMetrics**.
2. Pochodne wyliczone za pomocą pandasa.
3. Dodatkowe z darmowego API **blockchain.com**.

| Wskaźnik | Opis | Źródło / Formuła (dla DERIVED) | Link do CSV |
|---|---|---|---|
| **CapRealUSD** | Realized Capitalization (USD). Wycena UTXO po cenie w momencie ich powstania. | *Derived: CapMrktCurUSD / CapMVRVCur* | [CapRealUSD.csv](data_ingestion/csv/CapRealUSD.csv) |
| **CapMrktCurUSD** | Bieżąca kapitalizacja rynkowa (USD). | CoinMetrics | [CapMrktCurUSD.csv](data_ingestion/csv/CapMrktCurUSD.csv) |
| **CapMVRVCur** | Wskaźnik MVRV (Market Value to Realized Value). | CoinMetrics | [CapMVRVCur.csv](data_ingestion/csv/CapMVRVCur.csv) |
| **MVRV_Z** | MVRV Z-Score. Odchylenie standardowe Market Cap od Realized Cap. | *Derived: (CapMrkt - CapReal) / std(CapMrkt)* | [MVRV_Z.csv](data_ingestion/csv/MVRV_Z.csv) |
| **NUPL** | Net Unrealized Profit/Loss. | *Derived: (CapMrkt - CapReal) / CapMrkt* | [NUPL.csv](data_ingestion/csv/NUPL.csv) |
| **RealizedPrice** | Zrealizowana cena rynkowa. | *Derived: CapRealUSD / SplyCur* | [RealizedPrice.csv](data_ingestion/csv/RealizedPrice.csv) |
| **PuellMultiple** | Puell Multiple (Przychód z wydobycia vs jego średnia 365d). | *Derived: RevUSD / MA365(RevUSD)* | [PuellMultiple.csv](data_ingestion/csv/PuellMultiple.csv) |
| **RevUSD** | Całkowity przychód górników w USD (subsydium + opłaty). | *Derived: IssTotUSD + FeeTotUSD* | [RevUSD.csv](data_ingestion/csv/RevUSD.csv) |
| **FeeTotNtv** | Całkowita suma opłat zapłacona górnikom (w BTC). | CoinMetrics | [FeeTotNtv.csv](data_ingestion/csv/FeeTotNtv.csv) |
| **FeeTotUSD** | Całkowita suma opłat zapłacona górnikom (w USD). | *Derived: FeeTotNtv * PriceUSD* | [FeeTotUSD.csv](data_ingestion/csv/FeeTotUSD.csv) |
| **FeeMeanNtv** | Średnia opłata za transakcję (BTC). | *Derived: FeeTotNtv / TxCnt* | [FeeMeanNtv.csv](data_ingestion/csv/FeeMeanNtv.csv) |
| **FeeMeanUSD** | Średnia opłata za transakcję (USD). | *Derived: FeeTotUSD / TxCnt* | [FeeMeanUSD.csv](data_ingestion/csv/FeeMeanUSD.csv) |
| **FeeRevPct** | Procentowy udział opłat w całkowitym przychodzie górników. | *Derived: 100 * FeeTotUSD / RevUSD* | [FeeRevPct.csv](data_ingestion/csv/FeeRevPct.csv) |
| **AdrActCnt** | Liczba aktywnych adresów (wysyłające + odbierające). | CoinMetrics | [AdrActCnt.csv](data_ingestion/csv/AdrActCnt.csv) |
| **AdrBalCnt** | Liczba adresów z niezerowym balansem. | CoinMetrics | [AdrBalCnt.csv](data_ingestion/csv/AdrBalCnt.csv) |
| **TxCnt** | Liczba potwierdzonych transakcji dziennie. | CoinMetrics | [TxCnt.csv](data_ingestion/csv/TxCnt.csv) |
| **TxCntSec** | Transakcje na sekundę (TPS) jako średnia dzienna. | *Derived: TxCnt / 86400* | [TxCntSec.csv](data_ingestion/csv/TxCntSec.csv) |
| **TxTfrCnt** | Ilość transferów BTC (po usunięciu change outputs CoinMetrics).| CoinMetrics | [TxTfrCnt.csv](data_ingestion/csv/TxTfrCnt.csv) |
| **TxTfrValNtv** | Wolumen transferu w BTC (adjusted/estimated). | blockchain.com | [TxTfrValNtv.csv](data_ingestion/csv/TxTfrValNtv.csv) |
| **TxTfrValUSD** | Wolumen transferu w USD (adjusted/estimated). | blockchain.com | [TxTfrValUSD.csv](data_ingestion/csv/TxTfrValUSD.csv) |
| **BlkCnt** | Liczba bloków wydobytych dziennie. | CoinMetrics | [BlkCnt.csv](data_ingestion/csv/BlkCnt.csv) |
| **BlkIntMean** | Średni czas bloku (w sekundach). | *Derived: 86400 / BlkCnt* | [BlkIntMean.csv](data_ingestion/csv/BlkIntMean.csv) |
| **BlkSizeMeanByte** | Średni rozmiar bloku (w Bajtach). | blockchain.com | [BlkSizeMeanByte.csv](data_ingestion/csv/BlkSizeMeanByte.csv) |
| **HashRate** | Szacunkowy hashrate sieci (EH/s). | CoinMetrics | [HashRate.csv](data_ingestion/csv/HashRate.csv) |
| **SplyCur** | Bieżąca podaż BTC w obiegu. | CoinMetrics | [SplyCur.csv](data_ingestion/csv/SplyCur.csv) |
| **IssTotNtv** | Dzienna emisja BTC (subsydium z nowych bloków). | CoinMetrics | [IssTotNtv.csv](data_ingestion/csv/IssTotNtv.csv) |
| **IssTotUSD** | Dzienna emisja BTC w przeliczeniu na USD. | CoinMetrics | [IssTotUSD.csv](data_ingestion/csv/IssTotUSD.csv) |
| **IssContPctDay** | Roczna stopa inflacji na dzień wczorajszy (Dzienna Emisja / Podaż). | *Derived: IssTotNtv / SplyCur* | [IssContPctDay.csv](data_ingestion/csv/IssContPctDay.csv) |
| **IssContPctAnn** | Przewidywana roczna stopa inflacji w ujęciu rocznym. | *Derived: IssContPctDay * 365* | [IssContPctAnn.csv](data_ingestion/csv/IssContPctAnn.csv) |
| **SplyExpFut10yr**| Oczekiwana nowa podaż przez następne 10 lat (bazowana na harmonogramie). | CoinMetrics | [SplyExpFut10yr.csv](data_ingestion/csv/SplyExpFut10yr.csv) |
| **CapFutExp10yrUSD**| Wartość oczekiwanej nowej podaży 10yr w USD (Cap Dilution). | *Derived: SplyExpFut10yr * PriceUSD* | [CapFutExp10yrUSD.csv](data_ingestion/csv/CapFutExp10yrUSD.csv) |
| **UTXOCnt** | Całkowita liczba niespendowanych wyjść. | blockchain.com | [UTXOCnt.csv](data_ingestion/csv/UTXOCnt.csv) |
| **FlowInExNtv** | Napływ na znane giełdy w BTC. | CoinMetrics | [FlowInExNtv.csv](data_ingestion/csv/FlowInExNtv.csv) |
| **FlowInExUSD** | Napływ na znane giełdy w USD. | CoinMetrics | [FlowInExUSD.csv](data_ingestion/csv/FlowInExUSD.csv) |
| **FlowOutExNtv**| Odpływ ze znanych giełd w BTC. | CoinMetrics | [FlowOutExNtv.csv](data_ingestion/csv/FlowOutExNtv.csv) |
| **FlowOutExUSD**| Odpływ ze znanych giełd w USD. | CoinMetrics | [FlowOutExUSD.csv](data_ingestion/csv/FlowOutExUSD.csv) |
| **PriceBTC** | Oczywiście 1 | CoinMetrics | [PriceBTC.csv](data_ingestion/csv/PriceBTC.csv) |
| **PriceUSD** | Dzienna referencyjna cena zamknięcia w USD. | CoinMetrics | [PriceUSD.csv](data_ingestion/csv/PriceUSD.csv) |

---

## Metryki Premium / Zaawansowane (Czekające na wdrożenie GCP)

Pozostałe wskaźniki (Faza 2 i Faza 3 w ROADMAP.md), **nie są wylistowane jako gotowe CSV**, ponieważ ich samodzielne policzenie lokalnie jest niemożliwe bez analizy blok po bloku. Zostaną zrealizowane z poziomu BigQuery jako oddzielny Data Ingestion pipeline w GCP:

- **SplyAct30d / SplyAct180d / SplyAct1yr** (Active Supply - wymaga zmapowania wieku każdego pojedynczego UTXO).
- **TxTfrValDayDst** (CDD - wymaga złączenia każdego wejścia transakcyjnego z dniem utworzenia jego oryginalnego UTXO).
- **CapAct1yrUSD** (Podzbiór Realized Cap - wymaga UTXO set analysis).
- **FeeMedNtv / FeeMedUSD** (Mediany Opłat blokowych. Będą w GCP pobierane codziennie skryptem z API `mempool.space`).

## Instrukcje dla Agenta AI na innej maszynie
1. Aby wygenerować na nowo całą paczkę wyżej wylistowanych CSV, użyj skryptów: 
   - `python data_ingestion/derive_metrics.py` (CoinMetrics)
   - `python data_ingestion/blockchain_fetcher.py` (Blockchain.com)
2. Skrypty automatycznie zapisują dane do folderu `data_ingestion/csv/`. Upewnij się, że używasz środowiska wirtualnego opisanego w `SCRIPTS.md`.
