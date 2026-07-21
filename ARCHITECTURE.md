# Architektura Systemu Wskaźników BTC (BTC On-Chain Indicators)

Ten plik stanowi punkt wyjścia dla deweloperów oraz asystentów AI rozpoczynających pracę nad projektem. Zawiera pełen obraz systemu po jego przebudowie w czerwcu 2026.

## 1. Architektura i Przepływ Danych (Data Ingestion)
Projekt opiera się na pobieraniu i agregacji 45 potężnych wskaźników analitycznych z różnych źródeł:
*   **CoinMetrics:** Skrypt `data_ingestion/coinmetrics_fetcher.py` pobiera podstawowe darmowe wskaźniki (np. `CapRealUSD`, `SplyCur`).
*   **BGeometrics API:** Ponieważ CoinMetrics nie udostępnia UTXO Age Bands za darmo, dodano `bgeometrics_fetcher.py`, który pobiera wskaźniki gęstości rynkowej takie jak **SOPR** czy **CDD**.
*   **Mempool.space:** Skrypt `mempool_fetcher.py` dostarcza informacji o medianie opłat (FeeMedNtv).
*   **BigQuery Public Data:** Skrypt `bq_active_supply.py` samodzielnie wylicza złożone wskaźniki "Aktywnej Podaży" (np. `SplyAct1yr`) pisząc zapytania SQL bezpośrednio do darmowych logów łańcucha Bitcoina (darmowy terabajt GCP).
*   **Agregator:** Skrypt `upload_csv_to_bq.py` skanuje folder `/csv/`, wykonuje złączenie typu Outer Join (na dacie) na wszystkich plikach i ładuje pełną tabelę `WRITE_TRUNCATE` do bazy `btc-ind.btc_indicators.coinmetrics_daily` w Google BigQuery.

## 2. Automatyzacja (CI/CD)
System posiada plik konfiguracyjny `.github/workflows/daily_update.yml`, który po stronie GitHuba uruchamia proces Ingestion codziennie o 1:00 AM (ET). Uruchamia on wszystkie skrypty pobierania i wysyła złączony zbiór do BigQuery. Ze względów bezpieczeństwa (OAuth) nałożenie pliku workflow mogło wymagać manualnego commitu przez właściciela repo.

## 3. Backend (FastAPI na Google Cloud Run)
Backend znajduje się w folderze `/api/` i jest wdrożony jako Serverless Cloud Run App (`btc-api`). Został mocno zoptymalizowany:
*   **Dynamic Discovery:** Zamiast sztywno wpisanej listy, API dynamicznie odpytuje `INFORMATION_SCHEMA.COLUMNS` BigQuery o to, jakie wskaźniki faktycznie istnieją w tabeli. Obsługuje dowolną ilość kolumn (obecnie 45).
*   **Buforowanie (Caching):** Endpointy BQ zostały zabezpieczone przez `@cached(TTLCache)` z biblioteki `cachetools`. Zapytania historyczne są w 100% cachowane w RAM na 12 godzin, zdejmując całkowicie koszty i obciążenie z BigQuery.
*   **Korelacje (SciPy):** Endpoint `/api/v1/metrics/correlation` na żywo wczytuje z pamięci ceny BTC oraz dany wskaźnik za ostatnie N dni, obliczając naukową Korelację Pearsona. Zwraca tablicę posortowanych sił korelacji ułatwiającą określenie, które sygnały działają najlepiej.

## 4. Frontend (Next.js na Google Cloud Run)
Aplikacja kliencka w folderze `/dashboard/` została wdrożona na Cloud Run (`btc-dashboard`). Główne widoki to:
*   **Wskaźniki:** Widok pojedynczych wskaźników z opisami.
*   **Porównaj:** Zaawansowana zakładka do rysowania wieloliniowych wykresów w bibliotece Recharts (możliwość nakładania kliku wskaźników on-chain z dedykowaną lewą i prawą osią Y dla Ceny BTC).
*   **Korelacje:** Dynamiczna Heatmapa (pasek postępu i procenty) przedstawiająca wyniki z endpointu korelacji, pokazująca zależność rynkową od 30 do 365 dni.
*   **Baza Danych:** Panel oferujący pobranie każdego ze wskaźników w wygodnym formacie `nazwa.csv` zaciąganym z `/public/csv/`.

## Kolejne Kroki / Rozwój
Nowy programista rozwijający system powinien zapoznać się z kodem w `api/main.py` oraz `dashboard/src/app/` by zobaczyć sposób komunikacji. Zastosowany React korzysta z nowoczesnego SWR do cachingu zapytań HTTP po stronie interfejsu.
Możliwości rozwoju to m.in. wsparcie dla nowych baz danych, system AI ostrzegający mailowo o silnych sygnałach w korelacji lub bardziej rygorystyczne testowanie w GitHub Actions.
