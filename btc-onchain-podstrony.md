# Przegląd kluczowych prac on-chain i plan analiz

Poniżej 10 sekcji odpowiadających podstronom aplikacji. Każda zawiera: tytuł pracy, autorów i URL, tezę, listę miar (ze wskazaniem na Twoje wskaźniki), opis modeli/statystycznych relacji oraz szczegółowy **prompt dla programisty** (jakie wykresy/tabele, co pokolorować, co monitorować).

---

## 1. Bitcoin Network Activity and Bitcoin Price Return Volatility

**Pełny tytuł:** Bitcoin Network Activity and Bitcoin Price Return Volatility  
**Autorzy:** [sprawdź w PDF SSRN – typowo 2–3 autorów, ekonomia/finanse]  
**URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619010

### Teza (intuicja paperu)

Aktywność sieciowa użytkowników Bitcoina (on‑chain activity metrics) jest silnie współzależna i dodatnio związana z dzienną zmiennością zwrotów; można ją traktować jako proxy intensywności informacji/handlu.

### Miary statystyczne i mapowanie na wskaźniki aplikacji

1. **Zmienne wynikowe (rynkowe)**
   - Dzienna cena: `PriceUSD` / `PriceBTC`.
   - Dzienny log‑zwrot: \( r_t = \log(P_t) - \log(P_{t-1}) \).
   - Dzienna zmienność: np. \( rv_t = r_t^2 \) lub rolling std log‑zwrotów.

2. **Miary aktywności sieciowej (Twoje wskaźniki)**
   - Liczba transakcji: `TxCnt`, `TxTfrCnt`.
   - Intensywność w czasie: `TxCntSec`.
   - Aktywne adresy: `AdrActCnt`.
   - Wolumen wartości: `TxTfrValUSD`, `TxTfrValNtv`.
   - Zajętość blokspace: `BlkCnt`, `BlkSizeMeanByte`, `BlkIntMean`.
   - Opłaty: `FeeTotUSD`, `FeeTotNtv`, `FeeMeanUSD`, `FeeMeanNtv`, `FeeMedNtv`.

3. **Model bazowy GARCH(1,1)**

Model zwrotów:
\[
 r_t = \mu + \epsilon_t, \quad \epsilon_t \sim N(0, h_t)
\]

Równanie wariancji:
\[
 h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}
\]

4. **Regresja zmienności na metryki on‑chain**

Po wyestymowaniu \(h_t\) dopasuj model:
\[
 h_t = \gamma_0 + \sum_k \gamma_k X_{k,t-lag} + u_t
\]

gdzie \(X_k\) to wybrane logowane metryki aktywności (np. \(\log(AdrActCnt_t)\), \(\log(TxCnt_t)\), \(\log(FeeTotUSD_t)\)).

### Co obserwować w danych / wynikach

- Wysokie korelacje między wszystkimi metrykami aktywności (multikolinearność, klastry miar opisujących to samo zjawisko).
- Znak i istotność \(\gamma_k\): oczekiwane \(\gamma_k > 0\) dla większości miar aktywności.
- Stabilność parametrów w podokresach (np. przed 2020 vs 2020–2021 vs po 2022).

### Prompt dla programisty (podstrona 1)

1. Załaduj dzienne szeregi: `PriceUSD`, `AdrActCnt`, `TxCnt`, `TxCntSec`, `TxTfrCnt`, `TxTfrValUSD`, `UTXOCnt`, `BlkCnt`, `BlkSizeMeanByte`, `FeeTotUSD`, `FeeMeanUSD`, `FeeMedNtv`.
2. Oblicz dzienne log‑zwroty i serię \(rv_t = r_t^2\) oraz 30‑dniową realized volatility (rolling std log‑zwrotów).
3. Dopasuj model GARCH(1,1) do log‑zwrotów i wyeksportuj serię \(h_t\) (warunkowa wariancja).
4. Zbuduj korelacyjną macierz metryk aktywności oraz heatmapę (kolory: silne dodatnie korelacje na czerwono, ujemne na niebiesko).
5. Zrób regresję \(h_t\) na wybrane logowane metryki aktywności (oddzielne modele lub LASSO) i wyświetl tabelę współczynników z p‑value.
6. UI: 
   - Sekcja „Heatmap korelacji aktywności” – wykres macierzowy, hover pokazuje parę miar.
   - Sekcja „Wpływ aktywności na zmienność” – tabela współczynników z podświetleniem w **zielonym** tych zmiennych, których \(\gamma_k\) jest istotnie dodatnie (p < 0.05), i w **czerwonym** tych istotnie ujemnych.
   - Dodatkowo wykres liniowy `PriceUSD` vs \(h_t\) z zaznaczonymi okresami, gdy \(h_t\) jest powyżej np. 90‑percentyla (pokolorowane tło).

---

## 2. Return and Volatility Forecasting Using On‑Chain Flows in Cryptocurrency Markets

**Pełny tytuł:** Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets  
**Autorzy:** [sprawdź arXiv/SSRN – kilku autorów, econ/fin]  
**URL:** https://ideas.repec.org/p/arx/papers/2411.06327.html

### Teza (intuicja paperu)

Net flows na giełdy (wpływy/odpływy on‑chain) zawierają informację predykcyjną dla krótkoterminowych stóp zwrotu i zmienności; dla BTC efekty są słabsze niż dla ETH, ale wciąż istotne dla vol.

### Miary statystyczne i mapowanie

1. **Net flows**
   - \( NetFlowUSD_t = FlowInExUSD_t - FlowOutExUSD_t \)
   - \( NetFlowNtv_t = FlowInExNtv_t - FlowOutExNtv_t \)

2. **Zmienne wynikowe**
   - \( r_t = \log(P_t) - \log(P_{t-1}) \) z `PriceUSD`.
   - Realized volatility: np. \(RV_t = r_t^2\) lub 7‑dniowa rolling std.

3. **Modele regresyjne**

Zwroty:
\[
 r_t = \alpha + \beta_1 NetFlowUSD_t + \beta_2 NetFlowUSD_{t-1} + \epsilon_t
\]

Zmienność:
\[
 RV_t = \delta_0 + \delta_1 NetFlowUSD_t + \delta_2 NetFlowUSD_{t-1} + v_t
\]

### Co obserwować

- Znak \(\beta_1, \delta_1\): czy napływ netto na giełdy poprzedza spadki cen i wzrost vol.
- Istotność w różnych lagach (0, 1, 2 dni).
- Okresy, w których relacje się załamują (np. duże zdarzenia makro).

### Prompt dla programisty (podstrona 2)

1. Dodaj do dataframe kolumny `NetFlowUSD` i `NetFlowNtv` jako różnice FlowIn – FlowOut.
2. Oblicz \(r_t\) oraz \(r_t^2\) i/lub 7‑dniową realized volatility.
3. Zbuduj dwie regresje OLS (zwroty vs net flows, vol vs net flows) z lagami 0 i 1.
4. UI: 
   - Wykres time‑series `NetFlowUSD` nałożony na `PriceUSD` – użyj dwóch osi Y; okresy z bardzo wysokim dodatnim/ujemnym net flow zaznacz pionowymi liniami.
   - Tabela „Wpływ net flows” z kolumnami: cel (r lub vol), zmienna, lag, współczynnik, p‑value. Podświetl w **zielonym** wiersze z istotnością p < 0.05.
   - Wykres scatter (NetFlowUSD vs dzisiejszy zwrot) z linią regresji.

---

## 3. Blockchain Analiz Göstergelerinin Bitcoin Fiyatı Üzerindeki Etkisi

**Pełny tytuł:** Blockchain Analiz Göstergelerinin Bitcoin Fiyatı Üzerindeki Etkisi  
**Autorzy:** Yunus Kalkan, Halim Tatlı  
**URL:** https://dergipark.org.tr/tr/pub/esad/issue/74371/1181992

### Teza (intuicja)

SOPR, zyskowność górników (PM) i liczba aktywnych adresów (BAA) mają istotny wpływ na cenę Bitcoina w krótkim i długim okresie w modelu ARDL.

### Miary i mapowanie

- SOPR → `SOPR`.
- PM (profitability miners) → proxy z `RevUSD`, `FeeRevPct`, `HashRate`, `PuellMultiple`.
- BAA (Bitcoin Active Addresses) → `AdrActCnt`.
- Cena: `PriceUSD`.

### Model ARDL

Ogólna postać ARDL(p, q1, q2, q3):
\[
 Price_t = \sum_{i=1}^p \phi_i Price_{t-i} + \sum_{k=0}^{q_1} \theta_{1k} SOPR_{t-k} + \sum_{k=0}^{q_2} \theta_{2k} PM_{t-k} + \sum_{k=0}^{q_3} \theta_{3k} AdrActCnt_{t-k} + \epsilon_t
\]

### Co obserwować

- Istotność krótkoterminową (lagowe \(\theta_{jk}\)) i długoterminową (z części kointegracyjnej) dla SOPR, PM, AdrActCnt.
- Wyniki testu przyczynowości Toda–Yamamoto między Price a SOPR / AdrActCnt.

### Prompt dla programisty (podstrona 3)

1. Zresampluj dane do częstotliwości miesięcznej lub tygodniowej.
2. Zbuduj proxy `PM` np. jako \( PM_t = RevUSD_t / IssTotUSD_t \).
3. Sprawdź stacjonarność (ADF) dla `PriceUSD`, `SOPR`, `PM`, `AdrActCnt`.
4. Dopasuj model ARDL z automatycznym doborem opóźnień; policz long‑run multipliers.
5. Przeprowadź Toda–Yamamoto causality test.
6. UI:
   - Wykres 4 time‑series z zaznaczonymi okresami, gdy SOPR, PM i AdrActCnt są w najwyższych/lub najniższych kwantylach.
   - Tabela „ARDL – krótkoterminowy wpływ” z parametrami i p‑value; podświetl na **zielono** istotne dodatnie, na **czerwono** istotne ujemne.
   - Blok „Długoterminowe relacje” – przedstaw estymowany long‑run wpływ i jego znak.

---

## 4. Using On-Chain Data to Predict Bitcoin Cycles

**Pełny tytuł:** Using on-chain data to predict Bitcoin cycles  
**Autor:** Sebastian Näsman  
**URL (thesis):** https://osuva.uwasa.fi/handle/10024/16509

### Teza (intuicja)

Trzy wskaźniki – NUPL, MVRV Z‑score, CVDD – mogą być użyte jako kontrariańskie sygnały wejścia/wyjścia, generując strategie pobijające buy‑and‑hold.

### Miary i mapowanie

- `NUPL`.
- `MVRV_Z`, `CapMVRVCur`, `CapRealUSD`, `CapMrktCurUSD`.
- CVDD – brak wprost; można oszacować na bazie CDD oraz wartości emitowanych monet.
- `PriceUSD`, `RealizedPrice`, `CDD`, `SplyAct1yr`, `CapAct1yrUSD`.

### Reguły sygnałowe (uogólnienie)

- Sygnał LONG, gdy:
  - \( MVRV\_Z < z_{low} \) oraz \( NUPL < n_{low} \) (akumulacja / panika).
- Sygnał wyjścia / redukcji, gdy:
  - \( MVRV\_Z > z_{high} \) lub \( NUPL > n_{high} \) (euforia / przegrzanie).

\(z_{low}, z_{high}, n_{low}, n_{high}\) – progi kalibrowane na danych historycznych (np. kwantyle).

### Co obserwować

- Czy strategie sygnałowe NUPL/MVRV biją buy‑and‑hold po 2021/2022.
- Zależność CDD / SplyAct1yr od momentów sygnałów (czy stara podaż wychodzi na szczytach/bottomach).

### Prompt dla programisty (podstrona 4)

1. Oblicz kwantylowe progi dla `MVRV_Z` i `NUPL` (np. 10%, 25%, 75%, 90%).
2. Zdefiniuj kilka strategii:
   - Strategia A: Long tylko gdy `MVRV_Z` < dolny próg i `NUPL` < dolny próg; wyjście gdy obie metryki > górnych progów.
   - Strategia B: tylko MVRV_Z; Strategia C: tylko NUPL.
3. Przeprowadź backtest od np. 2013 r. do dziś, porównując z buy‑and‑hold i strategiami losowych wejść (Monte Carlo).
4. UI:
   - Wykres log‑equity curves dla wszystkich strategii vs buy‑and‑hold z zaznaczonymi punktami wejścia/wyjścia.
   - Tabela metryk: CAGR, max drawdown, Sharpe/Sortino, liczba transakcji.
   - Dodatkowy wykres `MVRV_Z` i `NUPL` z poziomami progowymi, punkty sygnałów pokolorowane (zielony = wejście, czerwony = wyjście).

---

## 5. On-chain analysis and cryptocurrency price forecasting using on-chain metrics

**Pełny tytuł:** On-chain analysis and cryptocurrency price forecasting using on-chain metrics  
**Autor:** [autor pracy magisterskiej, NTU]  
**URL:** https://dr.ntu.edu.sg/entities/publication/f393fcc8-1fdf-4fd6-8e4b-272a77097039

### Teza (intuicja)

On‑chain metrics, szczególnie miner revenue, opłaty i aktywne adresy, poprawiają krótkoterminowe prognozy cen kryptowalut (w tym BTC) w modelach GARCH+SARIMAX, Prophet i BiLSTM.

### Miary i mapowanie

- `RevUSD`, `IssTotUSD`, `FeeTotUSD`, `FeeRevPct`, `HashRate`, `PuellMultiple`.
- `AdrActCnt`.
- `PriceUSD`.

### Modele

1. **GARCH+SARIMAX** na log‑zwrotach z exogenous on‑chain:
\[
 r_t = \mu + \sum_j \beta_j X_{j,t} + \epsilon_t, \quad h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}
\]

2. **Prophet** z exogenous regressors (on‑chain metrics).

3. **BiLSTM/LSTM** z sekwencjami cech \(X_t = (PriceUSD_t, RevUSD_t, AdrActCnt_t, FeeTotUSD_t, HashRate_t, PuellMultiple_t, \dots)\).

### Co obserwować

- Poprawa RMSE/MAE w stosunku do modeli bez on‑chain metrics.
- Najważniejsze cechy wg importance / SHAP.

### Prompt dla programisty (podstrona 5)

1. Przygotuj dataset z dziennymi danymi `PriceUSD`, `RevUSD`, `FeeTotUSD`, `AdrActCnt`, `HashRate`, `PuellMultiple`, `FeeRevPct`.
2. Zbuduj modele:
   - GARCH+SARIMAX (z feature’ami jako exogenous).
   - Prophet (regressors = te same feature’y).
   - LSTM/BiLSTM (window 30–90 dni, horyzont predykcji 1–3 dni).
3. Porównaj RMSE/MAE/MAPE na train/validation/test.
4. UI:
   - Zakładka „Porównanie modeli” z tabelą metryk błędu.
   - Wykres rzeczywistej ceny vs predykcja dla każdego modelu.
   - Tabela „Feature importance” – podświetl w **zielonym** cechy najbardziej istotne we wszystkich modelach (np. RevUSD, AdrActCnt).

---

## 6. Cryptocurrency Volatility Forecasting Using Transformer-Based Deep Learning Models and On-Chain Metrics

**Pełny tytuł:** Cryptocurrency Volatility Forecasting Using Transformer-Based Deep Learning Models and On-Chain Metrics  
**Autorzy:** [lista autorów – finanse/data science]  
**URL:** https://www.al-kindipublisher.com/index.php/jefas/article/view/10158

### Teza (intuicja)

Architektury Transformer z bogatym zestawem on‑chain metrics lepiej modelują zmienność BTC/ETH niż LSTM/GRU, szczególnie w okresach wysokiej turbulencji.

### Miary i mapowanie

- Target: 1‑/7‑dniowa realized volatility \( RV_t \) z `PriceUSD`.
- Features: `AdrActCnt`, `TxCntSec`, `TxTfrValUSD`, `FeeTotUSD`, `FeeMeanUSD`, `FeeMedNtv`, `HashRate`, `RevUSD`, `NUPL`, `MVRV_Z`, `SOPR`, `CapMrktCurUSD`, `CapRealUSD`.

### Modele

1. LSTM/GRU.
2. Transformer encoder.

### Co obserwować

- RMSE/MAE w okresach high vs low vol.
- Attention weights Transformera – które feature’y są najważniejsze w reżimach stresowych.

### Prompt dla programisty (podstrona 6)

1. Oblicz \(RV_t\) jako 7‑dniową rolling std log‑zwrotów `PriceUSD`.
2. Zbuduj sekwencyjne okna (np. 60 dni) z wymienionych feature’ów.
3. Dopasuj LSTM i Transformer encoder; zastosuj tę samą procedurę train/val/test.
4. UI:
   - Dwa wykresy: rzeczywista vol vs predykowana vol dla LSTM i Transformera.
   - Tabela błędów z osobnymi wierszami dla okresów high vol (np. górny kwartyl RV) i low vol (dolny kwartyl).
   - Heatmapa attention weights (czas vs feature) dla przykładowego okna, z podświetleniem najbardziej ważnych cech w okresach kryzysowych.

---

## 7. Using Machine and Deep Learning Models, On-chain Data, and Technical Analysis for Predicting Bitcoin Price Direction and Magnitude

**Pełny tytuł:** Using Machine and Deep Learning Models, On-chain Data, and Technical Analysis for Predicting Bitcoin Price Direction and Magnitude  
**Autorzy:** [autorzy – system engineering / data science]  
**URL:** https://scholarsmine.mst.edu/engman_syseng_facwork/1602/

### Teza (intuicja)

Modele ML/DL korzystające z kombinacji ceny, TA i on‑chain metrics osiągają wysoką skuteczność w prognozowaniu kierunku i magnitudy ruchów BTC; on‑chain metrics istotnie poprawiają wyniki.

### Miary i mapowanie

- Ceny/wycena: `PriceUSD`, `CapMrktCurUSD`, `RealizedPrice`, `CapRealUSD`, `MVRV_Z`, `NUPL`.
- Aktywność: `AdrActCnt`, `TxCntSec`, `TxTfrCnt`, `TxTfrValUSD`.
- Górnicy: `HashRate`, `RevUSD`, `PuellMultiple`, `IssContPctAnn`.
- Przepływy: `FlowInExUSD`, `FlowOutExUSD`.
- TA (do wyliczenia w kodzie): SMA, EMA, RSI, itp.

### Modele

- SVM, Random Forest, Gradient Boosting, MLP, LSTM, CNN‑LSTM, TCN itd.
- Selekcja cech (Boruta, SHAP, L1).

### Co obserwować

- Wzrost accuracy/F1, gdy dodasz on‑chain metrics.
- Ranking cech – które on‑chain metrics dominują.

### Prompt dla programisty (podstrona 7)

1. Zbuduj etykiety: 
   - Kierunek D+1: `label_dir = sign(r_{t+1})` ∈ {down, up}.
   - Magnituda: kategorie typu small/medium/large move.
2. Oblicz podstawowe wskaźniki TA z `PriceUSD` i `TxTfrValUSD` (RSI, MACD, SMA, itp.).
3. Zbuduj feature set (cena+TA+on‑chain) oraz wersję bez on‑chain.
4. Dopasuj SVM, Random Forest, Gradient Boosting; zastosuj Boruta/SHAP.
5. UI:
   - Macierz konfuzji i ROC dla najlepszego modelu.
   - Tabela feature importance; on‑chain metrics, które są w top N, pokoloruj **ciemną zielenią**.
   - Dwa wykresy słupkowe: accuracy/F1 modelu z on‑chain vs bez on‑chain.

---

## 8. Deciphering Bitcoin Blockchain Data by Cohort Analysis

**Pełny tytuł:** Deciphering Bitcoin blockchain data by cohort analysis  
**Autorzy:** [lista autorów – Scientific Data]  
**URL:** https://www.nature.com/articles/s41597-022-01254-0

### Teza (intuicja)

Dane UTXO można przekształcić w kohortowe szeregi czasowe (wiek monet, lifespan spent outputs), które są ekonomicznie interpretowalne i nadają się do analizy cykli i przepływów.

### Miary i mapowanie

- `CDD` (Coin Days Destroyed).
- `SplyAct30d`, `SplyAct180d`, `SplyAct1yr`.
- `CapAct1yrUSD`.
- `PriceUSD`.

### Modele

- Korelacja i proste regresje: 
\[
 RV_t = \alpha + \beta_1 CDD_t + \beta_2 SplyAct1yr_t + \epsilon_t
\]
- Ewentualnie modele progowe / Markov switching, gdzie reżim zależy od poziomu CDD.

### Co obserwować

- Czy skoki CDD i wzrost aktywnej podaży starszych monet wyprzedzają okresy podwyższonej zmienności.

### Prompt dla programisty (podstrona 8)

1. Wyświetl time‑series `CDD`, `SplyAct30d`, `SplyAct180d`, `SplyAct1yr`, `PriceUSD`.
2. Zidentyfikuj skrajne wartości CDD (np. > 95‑percentyla) i zaznacz je na wykresie ceny.
3. Zbadaj korelacje \(CDD_t\) i \(CDD_{t-1}\) z \(RV_t\).
4. UI:
   - Wykres ceny z markerami dni o ekstremalnym CDD.
   - Wykres scatter CDD vs RV z linią regresji.
   - Tabela prostych współczynników regresji z podświetleniem istotnych \(\beta\).

---

## 9. Blockchain Analysis of the Bitcoin Market

**Pełny tytuł:** Blockchain Analysis of the Bitcoin Market  
**Autorzy:** Igor Makarov, Antoinette Schoar  
**URL:** https://www.nber.org/papers/w29396

### Teza (intuicja)

Rynek BTC jest zdominowany przez duże encje (giełdy, górnicy, usługi), a znaczna część wolumenu on‑chain nie jest ekonomicznie znacząca; analiza powinna odbywać się na poziomie encji i przepływów między nimi.

### Miary i mapowanie

- Przepływy z/do giełd: `FlowInExUSD`, `FlowOutExUSD`, `FlowInExNtv`, `FlowOutExNtv`.
- Wolumen on‑chain: `TxTfrValUSD`, `TxTfrCnt`.
- Struktura adresów: `AdrBalCnt`.

### Modele / analizy

- Udział wolumenu związanego z giełdami w całości wolumenu on‑chain.
- Dynamika `AdrBalCnt` jako proxy rozproszenia własności.

### Co obserwować

- Czy wzrost udziału wolumenu giełdowego koreluje z większą zmiennością / inną strukturą zwrotów.

### Prompt dla programisty (podstrona 9)

1. Oblicz udział wolumenu giełdowego: 
   - \( ShareExUSD_t = (FlowInExUSD_t + FlowOutExUSD_t) / TxTfrValUSD_t \).
2. Wyświetl time‑series `ShareExUSD` i `PriceUSD`.
3. Oblicz rolling korelacje pomiędzy `ShareExUSD` a 30‑dniową realized volatility.
4. UI:
   - Wykres `ShareExUSD` z kolorowaniem okresów najwyższych 20% wartości.
   - Wykres rolling korelacji vs czasu.
   - Prosta tabela statystyk (średnia, mediana, skrajne kwantyle `ShareExUSD`).

---

## 10. Inferring Short-Term Volatility Indicators from Bitcoin Blockchain

**Pełny tytuł:** Inferring short-term volatility indicators from Bitcoin blockchain  
**Autorzy:** Aaron Zweig, Gabriele O. Santini, inni (sprawdź arXiv)  
**URL:** https://arxiv.org/abs/1809.07856

### Teza (intuicja)

Da się zbudować wczesne wskaźniki ostrzegawcze (EWIs) dla okresów ekstremalnej zmienności, opierając się na cechach dziennych grafów transakcji i ich niskowymiarowych reprezentacjach (NMF, SVD).

### Miary i mapowanie

- W pracy: cechy grafu transakcji (flow, złożoność, decompositions).
- W Twoich danych brak bezpośrednich cech grafowych, ale część może być proxy:
  - `TxCnt`, `TxTfrCnt`, `TxTfrValUSD`, `UTXOCnt`, `AdrActCnt`, `FeeTotUSD`, `BlkSizeMeanByte`.

### Modele

- Idea: z cech grafu budować niskowymiarowy wektor \(z_t\) i używać go jako EWI dla zmienności.
- Przy braku pełnego grafu możesz zbudować analogiczny \(z_t\) z Twoich metryk, np. PCA lub NMF:
\[
 X_t \in \mathbb{R}^d \Rightarrow z_t = W^T X_t, \quad dim(z_t) = k \ll d
\]

### Co obserwować

- Czy \(z_t\) przewiduje przekroczenia progów zmienności (np. dni, w których \(RV_t > q_{0.95}\)).

### Prompt dla programisty (podstrona 10)

1. Zbuduj macierz feature’ów \(X_t\) z `TxCnt`, `TxTfrCnt`, `TxTfrValUSD`, `UTXOCnt`, `AdrActCnt`, `FeeTotUSD`, `BlkSizeMeanByte`.
2. Zastosuj PCA lub NMF, redukując wymiar do k=2–5.
3. Zdefiniuj high‑vol days jako te z RV_t > 95‑percentyla; dopasuj prosty klasyfikator (logit) \(z_t\) → high‑vol or not.
4. UI:
   - Wykres scatter głównych komponentów (PC1 vs PC2) z kolorowaniem poziomu vol.
   - Tabela jakości klasyfikacji (TP, FP, prec/recall) dla prognozy high‑vol.
   - Time‑series EWI_t vs RV_t, z wyraźnym podświetleniem dni, kiedy EWI_t przekracza próg alarmowy.

---

## Uwaga o brakujących wskaźnikach

Większość brakujących w literaturze wskaźników możesz skonstruować jako kombinacje istniejących szeregów:

- \( NVT_t = CapMrktCurUSD_t / TxTfrValUSD_t \).
- Supply in profit/loss na bazie `PriceUSD`, `RealizedPrice`, `SplyCur` (i ewentualnie zewnętrznych danych o rozkładzie UTXO).
- Proste HODL waves na bazie `SplyAct30d/180d/1yr`.
- Przybliżone CVDD z `CDD` i wartości emisji (`IssTotNtv`, `IssTotUSD`).
