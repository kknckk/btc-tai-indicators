# Roadmapa Projektu (Wdrożenia i Analityka)

Ten dokument opisuje plan działania na kolejne iteracje projektu. Do tej pory zrealizowaliśmy i wyeksportowaliśmy do plików CSV 34 podstawowe i zaawansowane wskaźniki przy użyciu danych darmowych. Poniżej znajduje się szczegółowa architektura i logika, którą należy wdrożyć, by uzyskać te najbardziej zaawansowane (zależne od struktury UTXO i pełnego węzła Bitcoin).

## 1. Wyzwanie: Pełny Węzeł (Full Node) w chmurze
Wskaźniki takie jak:
- **SplyAct30d / SplyAct180d / SplyAct1yr** (Podaż Aktywna w oknach czasowych)
- **TxTfrValDayDst** (Coin Days Destroyed - CDD)
- **ExchangeFlowIn / ExchangeFlowOut** (Przepływy na giełdach)
- **TxTfrValAdjUSD** (Wartość transferów z odrzuceniem tzw. change outputs)

Wymagają one dostępu do pełnego zestawu UTXO (Unspent Transaction Outputs) lub skomplikowanej heurystyki (clustering adresów). 

### Planowane rozwiązanie infrastrukturalne w GCP
Zamiast stawiać własny serwer za 100$ miesięcznie z 1TB dyskiem, użyjemy **Google BigQuery Public Data** (`bigquery-public-data.crypto_bitcoin`). Ten dataset zawiera sparsowane i zindeksowane wszystkie bloki i transakcje Bitcoina, odświeżane każdego dnia.

## 2. Harmonogram Prac (Kamienie Milowe)

### Etap A: Implementacja logiki zapytań SQL w BigQuery (CDD i SplyAct)
1. Napisz zaawansowane zapytania SQL do `crypto_bitcoin.inputs` i `crypto_bitcoin.outputs`.
2. Dla **CDD**: Należy złączyć tabele inputów z tabelami oryginalnych outputów, by obliczyć czas życia każdej monety w dniach (różnica block_timestamp) i pomnożyć przez `value`.
3. Dla **Active Supply**: Zidentyfikuj i zsumuj wszystkie wyjścia (outputs), które nie zostały wydane przez X dni od momentu zapytania.

### Etap B: Cloud Functions i Cloud Scheduler (Daily Workers)
1. Obecne skrypty (jak `mempool_fetcher.py`) oraz nowe zapytania SQL z Etapu A należy zamknąć w osobnych, ustandaryzowanych funkcjach Google Cloud Functions (Python 3.11).
2. Dodaj Cloud Scheduler wyzwalany cronem (np. `0 2 * * *` - o 02:00 codziennie).
3. Logika Workera: 
   - Obudź się
   - Oblicz wskaźniki dla minionego pełnego dnia (`CURRENT_DATE() - 1`)
   - Wykonaj `INSERT INTO` (append) do naszej tabeli `btc_indicators.daily_metrics`
   - Zakończ działanie (koszt znikomy).

### Etap C: Clustering Giełd i Heurystyki Change Outputs
1. Pozyskanie z zewnętrznych źródeł otwartych (np. repozytoria uniwersyteckie, datasety z Kaggle lub Glassnode Free Tier) bazy adresów należących do giełd (Binance, Coinbase etc.).
2. Napisanie workera w Pythonie uruchamianego w Cloud Run Job, który po pobraniu dziennych transakcji z BQ będzie stosował heurystykę wykrywania adresów resztowych (Change Outputs) - m.in. zasada największego zwrotu do tego samego rodzaju skryptu (np. z Segwit do Segwit).

## Zakończenie
Dzięki wykorzystaniu BigQuery ominiemy problem pełnego węzła, zachowując niemal 100% serverless workflow. Skup się w pierwszej kolejności na poprawnym napisaniu skryptów SQL do obliczenia CDD.
