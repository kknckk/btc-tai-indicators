# Dokumentacja Wskaźników On-Chain (Lista 56 wskaźników)

dokument zawiera podsumowanie 56 metryk. 

---

## 1. Wskaźniki pobierane bezpośrednio z darmowych API

Są to surowe dane zaciągane prosto z zewnętrznych dostawców. **API odgrywają tu rolę źródła prawdy.** Nie dokonujemy na nich własnych manipulacji analitycznych, ufając danym dostawcy. 

### Źródło: API CoinMetrics (Plik: btc.csv)
Skrypt w naszym projekcie: `coinmetrics_fetcher.py`

*Uwaga do adresów giełd (FlowInEx/OutEx):* Poniższe wskaźniki dotyczące giełd wymagają zaawansowanej heurystyki i tzw. **labelingu adresów** (kategoryzowania, który adres należy do Binance, Coinbase itp.). Nie wiemy, w jaki dokładnie sposób CoinMetrics przeprowadza ten labeling i jak bardzo jest on dokładny w darmowym wariancie ich API, dlatego traktujemy te dane z lekką dozą rezerwy.

1. **CapMrktCurUSD** – Bieżąca kapitalizacja rynkowa (USD). Obliczana jako podaż * cena.
2. **CapMVRVCur** – Wskaźnik MVRV (Market Value to Realized Value).
3. **FeeTotNtv** – Całkowita suma opłat transakcyjnych zapłacona górnikom (w BTC).
4. **AdrActCnt** – Liczba aktywnych adresów w danym dniu.
5. **AdrBalCnt** – Liczba adresów z niezerowym saldem.
6. **TxCnt** – Liczba potwierdzonych transakcji w ciągu dnia.
7. **TxTfrCnt** – Ilość transferów BTC (po odfiltrowaniu change outputs).
8. **BlkCnt** – Liczba bloków wydobytych w ciągu doby.
9. **HashRate** – Szacunkowa moc obliczeniowa sieci (EH/s).
10. **SplyCur** – Podaż BTC aktualnie znajdująca się w obiegu.
11. **IssTotNtv** – Dzienna emisja BTC (subsydium).
12. **IssTotUSD** – Dzienna emisja BTC (w USD).
13. **SplyExpFut10yr** – Oczekiwana nowa podaż przez kolejne 10 lat.
14. **FlowInExNtv** / **ExchangeFlowIn** – Wolumen napływający na znane giełdy (w BTC). ryzyko poprawności danych- nieznany labeling
15. **FlowInExUSD** / **ExchangeFlowInUSD** – Wolumen napływający na giełdy (w USD). ryzyko poprawności danych- nieznany labeling
16. **FlowOutExNtv** / **ExchangeFlowOut** – Wolumen wypływający z giełd (w BTC). ryzyko poprawności danych- nieznany labeling
17. **FlowOutExUSD** / **ExchangeFlowOutUSD** – Wolumen wypływający z giełd (w USD). ryzyko poprawności danych- nieznany labeling
18. **PriceBTC** – Cena Bitcoina wyrażona w Bitcoinie (zawsze 1).
19. **PriceUSD** – Dzienna referencyjna cena zamknięcia w USD.
20. **TxTfrValAdjUSD** – Skorygowany wolumen transferu (CoinMetrics usuwa szum z mikrotx) w USD.
21. **TxTfrValAdjNtv** – Skorygowany wolumen transferu w BTC.

### Źródło: API Blockchain.com
Skrypt w naszym projekcie: `blockchain_fetcher.py`

22. **TxTfrValNtv** – Szacowany faktyczny wolumen transferu (w BTC), po odfiltrowaniu przesyłania środków do samego siebie.
23. **TxTfrValUSD** – Szacowany wolumen transferu w dolarach (USD).
24. **BlkSizeMeanByte** – Średni rozmiar bloku. Pobierany w MB i rzutowany przez nas na Bajty.
    *Działanie:* `df[metric_name] = df[metric_name] * 1_000_000`
25. **UTXOCnt** – Całkowita liczba niespendowanych wyjść (UTXO) w sieci.

### Źródło: API BGeometrics (api.bitcoin-data.com)
Skrypt w naszym projekcie: `bgeometrics_fetcher.py`

26. **SOPR** (Spent Output Profit Ratio) – Pokazuje, czy monety wydawane danego dnia były średnio w zysku, czy na stracie.
27. **CDD** / **TxTfrValDayDst** (Coin Days Destroyed) – Mierzy "aktywność starych monet". Ilość przesłanych BTC pomnożona przez liczbę dni od ich poprzedniego transferu.
28. **LTH_MVRV** (Long-Term Holder MVRV) – Część wskaźnika LTH/STH split. Analiza MVRV tylko dla monet nie ruszanych od dawna.
29. **STH_MVRV** (Short-Term Holder MVRV) – Analiza MVRV dla monet świeżo wygenerowanych lub niedawno przeniesionych.

### Źródło: API Mempool.space
Skrypt w naszym projekcie: `mempool_fetcher.py`

30. **FeeMedNtv** – Mediana opłat sieciowych (w BTC). Wyciągnięta jako uśredniona mediana opłat z poszczególnych bloków w ciągu dnia.

---

## 2. Wskaźniki wyliczane u nas ("Derywaty" i agregacje)

Są to wskaźniki powstałe na bazie skryptów wyliczających z innych danych.

### Środowisko: Chmura Google (GCP - BigQuery)
Skrypt: `bq_easy_metrics.py`

31. **DiffMean** (Średnia Trudność Sieci)
    - *Źródło:* Publiczny dataset BigQuery (`bigquery-public-data.crypto_bitcoin.blocks`)
    - *Działanie:* Wyliczana poprzez konwersję kompresowanego szesnastkowego pola `bits` bloków.
    ```python
    exponent = val >> 24
    coefficient = val & 0xffffff
    target = coefficient * (256 ** (exponent - 3))
    max_target = 0xFFFF * (256 ** (0x1d - 3))
    difficulty = max_target / target
    ```
32. **TxMeanByte**
    - *Źródło:* BQ (`crypto_bitcoin.transactions`)
    - *Działanie:* Wyliczane jako średni rozmiar transakcji (w bajtach) z danego dnia.
    ```sql
    SELECT AVG(size) as TxMeanByte FROM `bigquery-public-data.crypto_bitcoin.transactions` GROUP BY DATE(block_timestamp)
    ```
33. **FeeMedUSD**
    - *Działanie:* Otrzymana lokalnie z wymnożenia pobranego wcześniej (punkt 30) medianowego `FeeMedNtv` i dziennej ceny.
    ```python
    df_fee["FeeMedUSD"] = (df_fee["FeeMedNtv"] / 100_000_000) * df_fee["PriceUSD"]
    ```

### Derywaty Lokalne: "Oszukane" czyli matematyczny reverse engineering)
Skrypt: `derive_metrics.py`

Wyliczenie tych metryk od zera wymagałoby śledzenia daty każdego UTXO aż od bloku Genesis. Jako że nie posiadamy takich możliwości obliczeniowych bez skanowania całego BQ, odzyskujemy je ze wskaźników z CoinMetrics dzięki algebrze (z tzw. ułamka MVRV).

34. **CapRealUSD** (Realized Capitalization)
    - *Działanie:* Wycena UTXO po cenie z dnia ich powstania. Zamiast weryfikować wiek, odwracamy wzór na MVRV.
    ```python
    df["CapRealUSD"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
    ```
35. **RealizedPrice** (Zrealizowana Cena)
    ```python
    df["RealizedPrice"] = df["CapRealUSD"] / df["SplyCur"]
    ```
36. **NUPL** (Net Unrealized Profit/Loss)
    - *Działanie:* Wskazuje sumę niezrealizowanych zysków minus strat. Odzyskane za pomocą odejmowania obu kapitalizacji.
    ```python
    df["NUPL"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
    ```
37. **MVRV_Z** (MVRV Z-Score)
    - *Działanie:* Standardowe odchylenie używające Expanding Window na Market Cap.
    ```python
    market_cap_std = df["CapMrktCurUSD"].expanding().std()
    df["MVRV_Z"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / market_cap_std
    ```

### Derywaty Lokalne: tylko matematyka
Skrypty: `derive_metrics.py` oraz `compute_derived_metrics.py`

38. **PuellMultiple** – Stosunek przychodów do średniej 365-dniowej.
    ```python
    ma365_rev = df["RevUSD"].rolling(window=365, min_periods=1).mean()
    df["PuellMultiple"] = df["RevUSD"] / ma365_rev
    ```
39. **RevUSD** – Całkowity zysk w USD. `IssTotUSD + FeeTotUSD`
40. **FeeTotUSD** – Suma opłat zwaloryzowana z Ntv na USD. `FeeTotNtv * PriceUSD`
41. **FeeMeanNtv** – Średnia zapłacona opłata. `FeeTotNtv / TxCnt`
42. **FeeMeanUSD** – Średnia opłata w USD. `FeeTotUSD / TxCnt`
43. **FeeRevPct** – Ile % zarobku górników pochodzi z opłat. `100 * (FeeTotUSD / RevUSD)`
44. **TxCntSec** – Ilość transakcji na sekundę (TPS). `TxCnt / 86400`
45. **BlkIntMean** – Czas bloku. `86400 / BlkCnt`
46. **IssContPctDay** – Bieżąca roczna inflacja ujęta 1-dniowo. `IssTotNtv / SplyCur`
47. **IssContPctAnn** – Prognoza rocznej inflacji. `IssContPctDay * 365`
48. **CapFutExp10yrUSD** – Zabezpieczony nakład nowej podaży 10-letniej. `SplyExpFut10yr * PriceUSD`
49. **IssContNtv** – Dokładna kopia wskaźnika `IssTotNtv` w celach dopasowania do starych nazw.
50. **TxTfrValMeanNtv** – Średnia wartość transferu w BTC. `TxTfrValNtv / TxTfrCnt`
51. **TxTfrValMeanUSD** – Średnia wartość transferu w USD. `TxTfrValUSD / TxTfrCnt`
52. **RevAllTimeUSD** – Skumulowany zysk górników (All-time sum). `(FeeTotUSD + IssTotUSD).cumsum()`
53. **NVTAdj / NVT_Naive** (Network Value to Transactions) – Miernik przewartościowania (czy wartość sieci nie rośnie szybciej niż adopcja). `CapMrktCurUSD / TxTfrValUSD`
54. **VelCur1yr / VelCur1yr_Naive** – Prędkość roczna obiegu monet (Naiwna). `TxTfrValNtv / SplyAct1yr`
55. **NVTAdj90** – Analogiczny NVT wyrównany 90-dniową średnią (jeśli występuje w skryptach, liczony jako 90-d SMA).

---

## 3. Wskaźniki, których NIE udało się policzyć i są symulowane (Mockowane)

Mamy problem z pełnym wyliczaniem Active Supply --- wymaga zapytania o rozmiarze ok. 2 TB na BigQuery public dataset. 
https://cloud.google.com/blog/topics/public-datasets/bitcoin-in-bigquery-blockchain-analytics-on-public-data

Nawiasem mówiąc to bardzo wygodne źródło i ma sporo innych chainów https://github.com/blockchain-etl/public-datasets 

skrypt `bq_active_supply.py`.

56. **SplyAct30d / SplyAct180d / SplyAct1yr / CapAct1yrUSD** (Podaż Aktywna i związana z nią Kapitalizacja)
    - *Dlaczego się nie udało:* Dokładne wyliczenie wieku *każdego* UTXO i zagregowanie aktywnej grupy to gigantyczne zapytanie JOIN tabel Inputs z Outputs dla milionów operacji dziennie.
    - *Działanie (Mockowanie):* W kodzie zwracamy statyczne stałe w miejsce trudnych obliczeń:
    ```sql
    SELECT 
        CAST(CURRENT_DATE() AS STRING) as time,
        18500000.0 as SplyAct1yr,
        12000000.0 as SplyAct180d,
        5000000.0 as SplyAct30d,
        1000000000000.0 as CapAct1yrUSD
    ```
    Dzięki temu wskaźniki te de facto istnieją jako pliki CSV, a pipeline nie pada z braku danych, ale nie przedstawiają faktycznej wartości historycznej do momentu zbudowania dedykowanego DAGa w GCP (np. z Dataform), który inkrementalnie zbuduje te tabele.

**Propozycje co zrobić ze wskaźnikami, których nie da się lokalnie policzyć:**
1. **Podejście Inżynieryjne (GCP/Dataform):** Utworzenie inkrementalnego widoku materializowanego (np. przy pomocy dbt lub Dataform), który co noc analizuje tylko różnicę (delta) w nowych blokach, aktualizując stan UTXO, zamiast skanować całą 2 TB bazę danych z `crypto_bitcoin`.
2. **Użycie innych źródeł API:** Możemy spróbować pobierać pre-kalkulowane Active Supply z innych API wspierających dane Glassnode lub CryptoQuant (często płatne, ale można szukać alternatyw wektorowych jak w BGeometrics).
3. **Podejście Proxy (Heurystyki):** Zamiana trudnego `SplyAct1yr` na agregaty takie jak "ilość transakcji o wysokim CDD" z innych endpointów. Jeśli brakuje nam dokładnego Active Supply, czasem wystarczy nam wiedza, czy w danym dniu ruszyło się ponadprzeciętnie dużo starych monet (co zapewnia np. wskaźnik CDD z BGeometrics).

---

## 4. Analiza Literatury i Planowane Eksperymenty (Zakres Prac Technicznych)

Na podstawie zebranych prac badawczych, wyłoniliśmy konkretne eksperymenty do zrobienia. to by miało potwierdzić 'układ odniesienia' i jakość danych. tzn czy umiemy reprodukować inne znane wyniki (recenzowane), zanim zaproponujemy swoje miary czy modele. 


### 1. Bitcoin Network Activity and Bitcoin Price Return Volatility
- **Tytuł i URL:** [Bitcoin Network Activity and Bitcoin Price Return Volatility (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619010)
- **Teza do zreprodukowania:** Aktywność sieciowa (TxCnt, AdrActCnt, opłaty) jest silnie dodatnio powiązana z dzienną zmiennością zwrotów Bitcoina.
- **Kroki badania:** Zbudowanie modelu GARCH(1,1) na zwrotach; regresja warunkowej wariancji na zlogarytmowane metryki aktywności.
- **Zakres prac technicznych (Skrypty):** 
  - Skrypt ekonometryczny używający biblioteki `arch` (w Pythonie) do dopasowania modelu GARCH(1,1) do log-zwrotów ceny (PriceUSD).
  - Skrypt analizy statystycznej (`statsmodels`), który wyciągnie szereg warunkowej wariancji ($h_t$) i wykona regresję OLS przeciwko zlogarytmowanym wskaźnikom on-chain.
  - Wygenerowanie korelacyjnej macierzy (Heatmapy) wielowspółliniowości.

### 2. Return and Volatility Forecasting Using On-Chain Flows
- **Tytuł i URL:** [Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets (arXiv)](https://ideas.repec.org/p/arx/papers/2411.06327.html)
- **Teza do zreprodukowania:** Wpływy netto na giełdy (NetFlowUSD) wyprzedzają krótkoterminowe spadki cen i wzrost zmienności.
- **Kroki badania:** Obliczenie różnic między wpływami a wypływami z giełd; dopasowanie modeli OLS ze zmiennymi opóźnionymi.
- **Zakres prac technicznych (Skrypty):**
  - Skrypt budujący nową cechę `NetFlow = FlowInEx - FlowOutEx`.
  - Skrypt regresji liniowej (`statsmodels.api.OLS`) z tzw. lagami (opóźnieniami: $t$, $t-1$, $t-2$) sprawdzający predykcyjność napływów na giełdę.

### 3. Blockchain Analiz Göstergelerinin Bitcoin Fiyatı Üzerindeki Etkisi
- **Tytuł i URL:** [Blockchain Analiz Göstergelerinin Bitcoin Fiyatı Üzerindeki Etkisi](https://dergipark.org.tr/tr/pub/esad/issue/74371/1181992)
- **Teza do zreprodukowania:** Wskaźnik SOPR, zyskowność górników (PM) i adresy aktywne wpływają krótko- i długoterminowo na cenę w modelu ARDL.
- **Kroki badania:** Testy stacjonarności (ADF); budowa modelu autoregresyjnego ARDL i testy przyczynowości Toda-Yamamoto.
- **Zakres prac technicznych (Skrypty):**
  - Moduł testów stacjonarności wykorzystujący Augmented Dickey-Fuller test (`statsmodels.tsa.stattools.adfuller`).
  - Zbudowanie modelu Autoregressive Distributed Lag (ARDL) z automatycznym doborem opóźnień.
  - Skrypt weryfikujący przyczynowość (Granger causality / Toda-Yamamoto).

### 4. Using On-Chain Data to Predict Cryptocurrency Cycles
- **Tytuł i URL:** [Using on-chain data to predict Bitcoin cycles (Univ. of Vaasa)](https://osuva.uwasa.fi/server/api/core/bitstreams/cb668ce8-05ec-4be2-9062-d503386079f3/content)
- **Teza do zreprodukowania:** Ekstremalne odczyty NUPL i MVRV Z-score pozwalają stworzyć kontrariańskie strategie inwestycyjne lepsze od "buy-and-hold".
- **Kroki badania:** Ustalenie progów kwantylowych paniki/euforii; symulacja backtestów (Long/Exit) na danych historycznych.
- **Zakres prac technicznych (Skrypty):**
  - Skrypt analityczny kalibrujący historyczne progi (kwantyle) dla MVRV_Z i NUPL.
  - Środowisko Backtestingowe (użycie biblioteki `vectorbt` lub klasycznego `pandas`), implementujące logikę wejść i wyjść na podstawie progów.
  - Skrypt symulacji Monte Carlo do weryfikacji, czy wyniki strategii różnią się statystycznie od losowego wchodzenia w rynek.

### 5. On-chain analysis and price forecasting using on-chain metrics
- **Tytuł i URL:** [On-chain analysis and cryptocurrency price forecasting using on-chain metrics (NTU)](https://dr.ntu.edu.sg/entities/publication/f393fcc8-1fdf-4fd6-8e4b-272a77097039)
- **Teza do zreprodukowania:** Dodanie wskaźników on-chain do modeli (GARCH, Prophet, LSTM) poprawia trafność prognoz krótkoterminowych.
- **Kroki badania:** Budowa klasycznych modeli TS bez danych on-chain oraz z nimi; porównanie błędów RMSE/MAE.
- **Zakres prac technicznych (Skrypty):**
  - Integracja biblioteki Prophet (`prophet`) z dodaniem zewnętrznych regresorów (on-chain).
  - Klasyczna sieć rekurencyjna BiLSTM w `PyTorch` lub `TensorFlow/Keras` zdefiniowana dla okien 30-90 dniowych.
  - Kod do ewaluacji modeli (RMSE, MAE) i analizy ważności cech (Permutation Feature Importance / SHAP).

### 6. Cryptocurrency Volatility Forecasting Using Transformer Models
- **Tytuł i URL:** [Cryptocurrency Volatility Forecasting Using Transformer-Based Deep Learning Models and On-Chain Metrics](https://www.al-kindipublisher.com/index.php/jefas/article/view/10158)
- **Teza do zreprodukowania:** Nowoczesne modele oparte na atencji (Transformers) deklasują sieci LSTM w modelowaniu zmienności w oparciu o on-chain.
- **Kroki badania:** Implementacja LSTM i Transformera prognozującego 7-dniową zmienność; analiza zachowania w reżimach stresowych.
- **Zakres prac technicznych (Skrypty):**
  - Zbudowanie modelu *Transformer Encoder* dedykowanego pod szeregi czasowe (PyTorch).
  - Skrypt trenujący obydwa modele (Transformer vs LSTM) z naciskiem na out-of-sample testing podczas krachów rynkowych (tzw. high-turbulence periods).
  - Ekstrakcja i wizualizacja "Attention Weights" pokazujących, które wskaźniki były najważniejsze w momentach rynkowej paniki.

### 7. Predicting Bitcoin Price Direction and Magnitude (ML/DL + TA)
- **Tytuł i URL:** [Using Machine and Deep Learning Models, On-chain Data, and Technical Analysis for Predicting Bitcoin Price Direction and Magnitude](https://scholarsmine.mst.edu/engman_syseng_facwork/1602/)
- **Teza do zreprodukowania:** Fuzja wskaźników analizy technicznej (TA) z on-chain optymalizuje kategoryzację kierunku ceny (Up/Down).
- **Kroki badania:** Zdefiniowanie etykiet (wzrost/spadek); selekcja cech przez algorytmy (Boruta / SHAP) dla Random Forest.
- **Zakres prac technicznych (Skrypty):**
  - Feature Engineering Skrypt do wyliczenia klasycznych wskaźników AT (SMA, RSI, MACD) z użyciem np. biblioteki `ta`.
  - Budowa pipeline'u klasyfikacyjnego w `scikit-learn` (SVM, Random Forest, Gradient Boosting).
  - Implementacja algorytmu Boruta (lub SHAP values z biblioteki `shap`), aby bezspornie udowodnić, czy wskaźniki on-chain są ważniejsze niż wskaźniki analizy technicznej.

### 8. Deciphering Bitcoin Blockchain Data by Cohort Analysis
- **Tytuł i URL:** [Deciphering Bitcoin blockchain data by cohort analysis (Nature Scientific Data)](https://www.nature.com/articles/s41597-022-01254-0)
- **Teza do zreprodukowania:** Ekstrema zjawiska Coin Days Destroyed (CDD) wyprzedzają kluczowe pęknięcia cykli.
- **Kroki badania:** Identyfikacja 95-percentyla CDD i nałożenie na historyczną zmienność rynkową.
- **Zakres prac technicznych (Skrypty):**
  - Prosty skrypt analityczny (Pandas) mapujący górny centyl (powyżej 95%) wskaźnika CDD przeciwko Realized Volatility.
  - (Opcjonalnie) implementacja prostego modelu Markowa (Markov Switching Model) kategoryzującego rynek w zależności od wieku wydawanych monet.

### 9. Blockchain Analysis of the Bitcoin Market
- **Tytuł i URL:** [Blockchain Analysis of the Bitcoin Market (NBER)](https://www.nber.org/system/files/working_papers/w29396/w29396.pdf)
- **Teza do zreprodukowania:** Duża część wolumenu on-chain to szum transakcji powiązanych z giełdami.
- **Kroki badania:** Porównanie całkowitego ztransferowanego wolumenu ($) ze zidentyfikowanym z/do giełd.
- **Zakres prac technicznych (Skrypty):**
  - Obliczenie miary koncentracji (Udział wolumenu giełdowego = Wpływy + Wypływy / Całkowity Wolumen).
  - Skrypt liczący kroczącą korelację (Rolling Correlation) między miarą koncentracji giełd a strukturą zwrotów cenowych.

### 10. Inferring Short-Term Volatility Indicators from Blockchain
- **Tytuł i URL:** [Inferring short-term volatility indicators from Bitcoin blockchain (arXiv)](https://arxiv.org/abs/1809.07856)
- **Teza do zreprodukowania:** Metody redukcji wymiarów (PCA, NMF) na zgranym zbiorze cech grafowych pozwalają wyodrębnić sygnał wczesnego ostrzegania (Early Warning Indicator - EWI).
- **Kroki badania:** Agregacja wielu zaszumionych wskaźników poprzez PCA; dopasowanie klasyfikatora wyłapującego piki zmienności.
- **Zakres prac technicznych (Skrypty):**
  - Skrypt używający Principal Component Analysis (`sklearn.decomposition.PCA`) redukujący 20-30 zaszumionych metryk do 2-3 głównych komponentów.
  - Dopasowanie modelu Regresji Logistycznej, który będzie na bieżąco klasyfikował i zwracał prawdopodobieństwo nadejścia dnia o wysokiej zmienności (RV > 95%).
