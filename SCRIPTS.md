# Dokumentacja Skryptów Ingestji Danych (Data Ingestion)

Ten plik zawiera opisy skryptów odpowiedzialnych za pobieranie, przetwarzanie i eksportowanie wskaźników sieci Bitcoin. Skrypty znajdują się w katalogu `data_ingestion/`.

## Wymagania

Do uruchomienia skryptów lokalnie wymagany jest Python (zalecany w wersji 3.9+) oraz biblioteki zdefiniowane w pliku `requirements.txt`.

Instalacja środowiska:
```bash
cd data_ingestion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 1. `derive_metrics.py` (Główny generator CSV)

**Opis**: Jest to najważniejszy skrypt w obecnej fazie projektu. Służy do masowego generowania plików `.csv` dla wskaźników. Ze względu na ograniczenia darmowego API CoinMetrics, skrypt ten pobiera tylko podstawowe dane, a brakujące wskaźniki (takie jak Realized Cap, MVRV_Z, NUPL, Puell Multiple) **wylicza samodzielnie (matematycznie) za pomocą biblioteki Pandas**.

**Jak działa**:
1. Pobiera plik `btc.csv` z repozytorium CoinMetrics.
2. Filtruje dane odrzucone (ścina do 01.06.2026).
3. Wykorzystuje surowe wskaźniki (np. `CapMrktCurUSD`, `CapMVRVCur`) do wyliczenia brakujących wskaźników pochodnych (DERIVED).
4. Przycina wynikowe kolumny i każdą z nich zapisuje jako osobny plik w folderze `csv/` (np. `CapRealUSD.csv`).

**Uruchomienie**:
```bash
python derive_metrics.py
```

---

## 2. `coinmetrics_fetcher.py` (Eksport do BigQuery)

**Opis**: Ten skrypt jest przygotowany pod docelowe wdrożenie serwerowe (GCP Cloud). Pobiera podstawowe wskaźniki i wpycha je natywnie do tabel hurtowni danych **Google BigQuery**, skąd potem łatwo (i tanio) serwuje je nasze REST API. Posiada również prosty mechanizm zrzutu CSV przed załadowaniem do bazy.

**Zmienne środowiskowe**:
- `GCP_PROJECT_ID` - ID Twojego projektu w konsoli Google Cloud Platform. (domyślnie "twoj-projekt-gcp")

**Uruchomienie lokalne (wymaga uwierzytelnienia GCP)**:
```bash
gcloud auth application-default login
export GCP_PROJECT_ID="moj-projekt"
python coinmetrics_fetcher.py
```

---

## 3. `mempool_fetcher.py` (Daily Worker)

**Opis**: Szablon / szkielet skryptu typu "Daily Worker". Skrypt ten łączy się z darmowym API portalu `mempool.space`, aby wyciągać medianę opłat dla sieci (FeeMedNtv). Zaprojektowany jest tak, by uruchamiać go jako chmurową funkcję (np. z Cloud Schedulera) dokładnie raz dziennie, o północy. Otrzymany wynik dokleja (metoda `WRITE_APPEND`) jako pojedynczy nowy wiersz do tabeli BigQuery.

**Uruchomienie**:
```bash
python mempool_fetcher.py
```

---

## Integracja w docelowym ekosystemie (Cloud)

Żaden z tych skryptów docelowo **nie musi działać na Twoim komputerze**.
Plan wdrożeniowy:
1. `derive_metrics.py` posłużył do jednorazowego wygenerowania paczki historii danych "na start".
2. Na maszynie docelowej (GCP) skrypty takie jak `coinmetrics_fetcher.py` zostaną spakowane do tzw. Cloud Functions.
3. Usługa Google Cloud Scheduler będzie wysyłać sygnał o północy, który "obudzi" skrypt pobierający małą paczuszkę danych z minionego dnia i doda ten jeden wiersz do BigQuery. Zapewnia to utrzymanie bazy z kosztami rzędu ułamków centa miesięcznie.
