# Bitcoin On-Chain Research Reviews
id: c337c82f-33c3-4fd8-b665-451230662c6b
created: 2026-08-05T08:31:25.197701Z
modified: 2026-08-06T11:29:40.318Z

## human
zobacz moje repozytorium https://github.com/kknckk/btc-tai-indicators
 
i tam jest analiza 10 artykułów. jakby odtworzone badania. zrób recenzje tych analiz. czy dobrze zostały zrobione? czy ponowić któreś z badań?
na koniec zrób raport z wykonania badan w docx. niech ten raport będzie długi i szczegółowy. niech wyjaśnia wszystko jak 5latkowi

## assistant
Przejrzałem całe repozytorium [https://github.com/kknckk/btc-tai-indicators](https://github.com/kknckk/btc-tai-indicators), ze szczególnym uwzględnieniem folderu `analytical_pipeline` (skrypty `run_econometrics.py`, `run_ml_dl.py`, `run_strategies.py`, `compute_derived_metrics.py` oraz wyniki w `results/`).

Oto skrócona recenzja + odpowiedź na pytanie „czy dobrze zrobione / czy ponowić”, a na końcu pełny, długi i bardzo szczegółowy raport w DOCX (tłumaczący wszystko jak 5-latkowi).

## Skrótowa ocena 10 analiz

| Paper | Temat | Ocena jakości | Czy ponowić? | Główny problem |
|-------|--------|---------------|--------------|----------------|
| 1 | GARCH + aktywność on-chain → zmienność | 3,5/5 | Tak (ulepszyć) | Brak HAC SE, brak lagów, brak diagnostyki GARCH |
| 2 | NetFlows → zwroty i zmienność | 3/5 | Tak | Współczynniki na „surowych” miliardach USD, brak normalizacji |
| 3 | ARDL (SOPR/NUPL, PM, AdrAct) | 3/5 | **Tak** | To nie prawdziwy ARDL + bounds test, ryzyko spurious regression |
| 4 | Strategia MVRV Z + NUPL | 2,5/5 | **Koniecznie** | **Look-ahead bias** (kwantyle na całej historii) |
| 5 | Prophet + regressory | 3/5 | Tak | Tylko 30 dni testu, systematyczne przeszacowanie (~30% błąd) |
| 6 | LSTM vs Transformer (zmienność) | 2/5 | Tak | Tylko 5 epok – to PoC, nie badanie |
| 7 | Klasyfikacja kierunku (RF + SVM) | 3/5 | Tak | Accuracy ≈ 47–51% (rzut monetą), brak oceny ekonomicznej |
| 8 | CDD → zmienność | 3,5/5 | Opcjonalnie | Wynik nieistotny – warto event study wokół ekstremów |
| 9 | ShareEx (koncentracja giełd) | 3/5 | Opcjonalnie | Głównie deskryptywne |
| 10 | PCA + EWI (wczesne ostrzeganie) | 2,5/5 | **Koniecznie** | Accuracy 90%, ale precision/recall = 0 (ignorowanie imbalance) |

### Co jest zrobione dobrze
- Pipeline jest czysty i automatyczny (`run_all.sh`).
- Dane pochodzą z darmowych źródeł i są sensownie łączone.
- Autor nie ukrywa słabych wyników (strategia przegrywa z buy-and-hold, klasyfikacja jest koło 50%).
- Time-series split i resampling do tygodni pokazują świadomość problemów czasowych.
- Pokrycie klasycznych tematów badań on-chain jest szerokie.

### Co jest zrobione źle / za prosto
1. **Look-ahead bias w Paper 4** – to najpoważniejszy błąd metodologiczny.
2. Większość modeli ekonometrycznych to uproszczone proxy, a nie pełne procedury z literatury.
3. Deep learning (Paper 6) jest niedotrenowany.
4. Paper 10 ignoruje klasyczny problem rzadkich zdarzeń (wysoka accuracy przy zerowym recallu).
5. Brak cytowań konkretnych artykułów, które miały być odtworzone.
6. Brak kosztów transakcyjnych, robust standard errors i testów stabilności na podokresach.

### Które badania ponowić w pierwszej kolejności
1. **Paper 4 (strategia)** – bez look-ahead, z kosztami, walk-forward i stałymi progami z literatury.
2. **Paper 10 (EWI/PCA)** – z obsługą imbalance i rolling evaluation.
3. **Paper 3 (ARDL)** – prawdziwy bounds test lub VECM.
4. Potem Papers 1, 5, 6, 7.

---

## Pełny raport

Przygotowałem długi, szczegółowy raport w formacie DOCX. Tłumaczy wszystko jak 5-latkowi (z analogiami do placu zabaw, huśtawek, skarbonek i termometrów), a jednocześnie zawiera pełną ocenę metodologiczną, tabele wyników, listę konkretnych poprawek kodu i rekomendacje priorytetowe.

**Plik:**

Raport ma m.in.:
- Proste wyjaśnienie całego<grok:render card_id="TaZO6" card_type="rendered_file_card" type="render_file"><argument name="file_path">/home/workdir/artifacts/Raport_Recenzji_Analiz_BTC_TAI_Indicators.docx</argument></grok:render> repozytorium i pipeline’u danych
- Szczegółową recenzję każdego z 10 paperów
- Ocenę w skali szkolnej
- Konkretną listę „co poprawić w kodzie” (natychmiast / średnioterminowo / długoterminowo)
- Podsumowanie osobno dla 5-latka i dla dorosłego badacza/inwestora

Jeśli chcesz, w następnym kroku mogę od razu przygotować poprawioną wersję `run_strategies.py` (bez look-ahead bias) albo listę 10 konkretnych artykułów naukowych, które najlepiej pasują do tych paperów.

## human
generuj ten raport jeszcze raz. jak mówię, że dla 5-latka tzn szzcegółow wyjaśnij, ale nie rób tych infantylnych porównań jak o huśtawce. nie tłumacz czym jest wskaźnik. tylko szzcegółowo. jaki artykuł analizujemy tytuł autorzy rok czasopismo. teza, weryfikacja itd

## assistant
Mam już pełną listę artykułów z PDF-a w repozytorium oraz dokładne dopasowanie do Paper 1–10. Generuję raport od nowa – szczegółowy, z tytułami, autorami, latami, czasopismami, tezą oryginalną, sposobem weryfikacji w kodzie, wynikami i oceną. Bez porównań do huśtawek i bez tłumaczenia, czym jest wskaźnik.Raport wygenerowany od nowa.

**Plik:** 

### Co się zmieniło względem poprzedniej wersji

- Usunięte wszystkie porówn<grok:render card_id="gEIyh" card_type="rendered_file_card" type="render_file"><argument name="file_path">/home/workdir/artifacts/Raport_Recenzji_Analiz_BTC_TAI_Indicators.docx</argument></grok:render>ania typu „huśtawka / plac zabaw / skarbonka”.
- Brak tłumaczenia, czym jest wskaźnik.
- Dla każdej z 10 analiz podane są:
  - **pełne dane bibliograficzne** (tytuł, autorzy, rok, czasopismo/źródło)
  - **teza oryginalna**
  - **co dokładnie zaimplementowano w kodzie**
  - **wyniki z `results/*.json`**
  - **ocena wierności wobec oryginału i rygoru statystycznego**
  - **werdykt + czy i jak ponowić**

### Mapowanie Paper → oryginalna praca (z PDF-a w Twoim repo)

| Paper | Oryginalna praca |
|-------|------------------|
| 1 | Wüstenfeld (2023) – *Bitcoin Network Activity and Bitcoin Price Return Volatility* (SSRN) |
| 2 | Chi, Chu, Hao (2024) – *Return and Volatility Forecasting Using On-Chain Flows…* (arXiv) |
| 3 | Kalkan & Tatlı (2022) – *Blockchain Analiz Göstergelerinin Bitcoin Fiyatı…* (Ekonomik ve Sosyal Araştırmalar Dergisi) |
| 4 | Näsman (2025/2026) – *Using on-chain data to predict Bitcoin cycles* (Research in International Business and Finance) |
| 5 | NTU thesis (2023) – *On-chain analysis and cryptocurrency price forecasting…* |
| 6 | Rafi et al. (2024) – *Cryptocurrency Volatility Forecasting Using Transformer…* (JEFAS) |
| 7 | Omole & Enke (2025) – *Using Machine and Deep Learning Models…* (Engineering Applications of Artificial Intelligence) |
| 8 | *Deciphering Bitcoin blockchain data by cohort analysis* (Scientific Data / Nature, 2022) |
| 9 | Makarov & Schoar (2021) – *Blockchain analysis of the Bitcoin market* (NBER/LSE) |
| 10 | PCA + EWI (redukcja wymiaru aktywności + klasyfikacja wysokiej zmienności) |

### Najważniejsze wnioski (skrót)

- **Paper 4 (strategia MVRV/NUPL)** – look-ahead bias (kwantyle na całej historii). **Trzeba powtórzyć.**
- **Paper 10 (PCA/EWI)** – accuracy 90%, ale precision/recall = 0 (imbalance). **Trzeba powtórzyć.**
- **Paper 3 (ARDL)** – brak bounds testu. Warto powtórzyć formalnie.
- **Paper 6 (Transformer)** – tylko 5 epok. To PoC, nie replikacja.
- **Paper 7** – accuracy ~47–51% vs 83% w oryginale (brak Boruta).
- Reszta: strukturalnie sensowna, ale uproszczona względem oryginałów.

Raport jest długi, szczegółowy i napisany wprost – bez infantylnych analogii.

## human
daj mi zatem prompty dla agenta kodującego te artykuł. które ma powtórzyć i co ma zrobić inaczej. potrzebuje maksymalnie szczegółowych promptów

## assistant
Poniżej masz **maksymalnie szczegółowe, samodzielne prompty** dla agenta kodującego.  
Każdy prompt jest napisany tak, żeby agent mógł go wkleić i od razu pracować w istniejącym `analytical_pipeline` (dane z `derived_metrics.csv` / `data_ingestion/csv`, ten sam styl zapisu wyników do JSON).

Kolejność = priorytet (najpierw krytyczne).

---

## PROMPT 1 — Paper 4 (Näsman 2025/2026) — Strategia MVRV Z + NUPL  
**Priorytet: KRYTYCZNY**

```
Jesteś ekspertem od backtestów strategii on-chain Bitcoina. Pracujesz w repozytorium btc-tai-indicators, folder analytical_pipeline.

CEL
Przepisać i poprawić Paper 4 (strategia timingowa na MVRV_Z i NUPL) tak, aby całkowicie wyeliminować look-ahead bias i spełniać standardy oryginalnej pracy Sebastiana Näsmana (2025/2026, Research in International Business and Finance, "Using on-chain data to predict Bitcoin cycles").

ORYGINALNY PROBLEM W OBECNYM KODZIE
W run_strategies.py progi kwantyli 20% i 80% są liczone na CAŁEJ historii (2009–2026). To jest look-ahead bias. Strategia "wie przyszłość".

WYMAGANIA METODOLOGICZNE (obowiązkowe)

1. Dane
   - Weź derived_metrics.csv (lub load_merged_data()).
   - Kolumny obowiązkowe: time, PriceUSD, r_t, MVRV_Z, NUPL.
   - Usuń wiersze z NaN w tych kolumnach.
   - Sortuj rosnąco po time.

2. Trzy warianty progów (zaimplementuj WSZYSTKIE trzy i porównaj):

   A. Expanding-window quantiles
      - Dla każdego dnia t progi = quantile(0.20) i quantile(0.80) policzone WYŁĄCZNIE na danych od początku do t-1 (nigdy nie używaj danych z t lub później).
      - Minimum 365 obserwacji historycznych zanim zaczniesz generować sygnały.

   B. Rolling-window quantiles (okno 4 lata = 1460 dni)
      - Progi = quantile(0.20/0.80) na oknie [t-1460, t-1].

   C. Stałe progi z literatury (Näsman + klasyczne progi Glassnode/Mahmudov-Puell)
      - Long entry: MVRV_Z < 0.0 AND NUPL < 0.0
      - Exit / Flat: MVRV_Z > 5.0 OR NUPL > 0.75
      (dodatkowo przetestuj wariant łagodniejszy: MVRV_Z > 3.5 OR NUPL > 0.70)

3. Logika pozycji
   - Stan: 1 = Long, 0 = Flat (brak shortów).
   - Wejście w Long tylko gdy oba warunki entry spełnione.
   - Wyjście do Flat gdy którykolwiek warunek exit spełniony.
   - Pozycja jest trzymana do sygnału przeciwnego (nie ma re-entry co dzień).
   - Sygnał na zamknięciu dnia t → pozycja od dnia t+1 (żeby uniknąć look-ahead na cenie).

4. Koszty transakcyjne (obowiązkowe)
   - Przy każdej zmianie pozycji odjąć koszt 0.20% (0.002) od zwrotu tego dnia (realistyczny koszt spot + slippage).
   - Zapisz też wariant bez kosztów (dla porównania z oryginałem).

5. Metryki (policzyć dla każdego z 3 wariantów progów + B&H)
   - CAGR
   - Annualized Sharpe (rf=0)
   - Sortino
   - Max Drawdown
   - Calmar Ratio = CAGR / |MaxDD|
   - Liczba transakcji (zmian pozycji)
   - Average holding period (dni)
   - % czasu w rynku
   - Equity curve (cumprod(1+strategy_return)) zaczynająca się od 1.0

6. Porównanie z buy-and-hold
   - B&H na dokładnie tym samym okresie, na którym strategia generuje sygnały.

7. Robustness (minimum)
   - Podziel historię na 3 okresy: 2013-01-01–2017-12-31, 2018-01-01–2020-12-31, 2021-01-01–koniec.
   - Policz Sharpe i MaxDD osobno dla każdego okresu (dla najlepszego wariantu progów).

8. Output
   - Zapisz wyniki do analytical_pipeline/results/strategy_results_v2.json w strukturze:
     {
       "paper4_v2": {
         "expanding": {metrics..., "equity_curve": [...]},
         "rolling_4y": {...},
         "fixed_literature": {...},
         "buy_and_hold": {...},
         "robustness_by_period": {...},
         "parameters": {"cost_bps": 20, "min_history_days": 365, ...}
       }
     }
   - Equity curve: lista dictów {"time": "YYYY-MM-DD", "strategy_value": float, "bh_value": float, "position": 0|1}

9. Kod
   - Nowy plik: run_strategies_v2.py (nie nadpisuj starego).
   - Użyj tylko pandas, numpy, ewentualnie quantstats jeśli jest w environment, ale nie wymagaj.
   - Zero look-ahead: żadne obliczenie progu ani sygnału nie może używać danych z przyszłości.
   - Na końcu wypisz w konsoli krótkie porównanie tabelaryczne Sharpe / MaxDD / CAGR dla 3 wariantów + B&H.

Nie dodawaj shortów. Nie optymalizuj progów in-sample. Nie używaj przyszłych danych. Kod ma być produkcyjny i czytelny.
```

---

## PROMPT 2 — Paper 10 (PCA + Early Warning Indicator)  
**Priorytet: KRYTYCZNY**

```
Jesteś ekspertem od Early Warning Indicators i klasyfikacji rzadkich zdarzeń w szeregach czasowych. Pracujesz w analytical_pipeline repozytorium btc-tai-indicators.

CEL
Przepisać Paper 10 (PCA aktywności sieciowej → logit high-volatility → EWI) tak, aby poprawnie obsłużyć niezbalansowane klasy i dać wiarygodną ocenę mocy predykcyjnej.

ORYGINALNY PROBLEM
Obecny kod ma accuracy ~90%, ale precision=0 i recall=0 dla klasy "high_vol". Model prawie nigdy nie przewiduje pozytywnej klasy. Accuracy jest myląca.

WYMAGANIA

1. Dane
   - derived_metrics.csv
   - Cechy do PCA (tylko te, które istnieją): TxCnt, TxTfrCnt, TxTfrValUSD, UTXOCnt, AdrActCnt, FeeTotUSD, BlkSizeMeanByte, TxCntSec (jeśli jest).
   - Target: high_vol = 1 jeśli rv_30d > quantile(0.95) policzony expanding-window (tylko przeszłość), inaczej 0.
   - Alternatywnie przetestuj też fixed quantile 0.95 na całej historii (dla porównania) – ale główny wynik ma być expanding.

2. PCA
   - Standaryzacja (StandardScaler) fitowana TYLKO na train foldzie.
   - n_components=2 (jak w oryginale) + dodatkowo wariant z n_components takim, żeby explained_variance >= 0.90.
   - Zapisz explained_variance_ratio.

3. Klasyfikacja (zamiast zwykłego Logit)
   Zaimplementuj i porównaj minimum 3 modele:
   a) LogisticRegression z class_weight='balanced'
   b) LogisticRegression z class_weight={0:1, 1: w} gdzie w = n_majority / n_minority
   c) RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=6)
   Opcjonalnie: BalancedRandomForest lub EasyEnsemble jeśli imblearn jest dostępne (nie wymagaj).

4. Ewaluacja – TYLKO time-series
   - Expanding window lub rolling origin:
     * Start z minimum 1000 dni historii.
     * Co 30 dni refit modelu na danych do t, przewiduj kolejne 30 dni.
   - Metryki (obowiązkowe, raportuj średnią i odchylenie po foldach):
     * Precision, Recall, F1 dla klasy 1
     * Balanced Accuracy
     * PR-AUC (average_precision_score)
     * ROC-AUC
     * Confusion matrix zagregowana
   - NIE raportuj zwykłego accuracy jako głównej metryki.

5. Early Warning Indicator
   - EWI_t = predicted probability klasy 1 z najlepszego modelu.
   - Alert = EWI_t > threshold.
   - Threshold wybierz tak, aby na validation (pierwsze 30% testowych foldów) zmaksymalizować F1 lub PR-AUC (nie accuracy).
   - Dodatkowo: event study – dla każdego alertu sprawdź średni zwrot i max drawdown w oknie [t, t+7], [t, t+14], [t, t+30] i porównaj z losowymi dniami (Monte Carlo 1000 losowań).

6. Output
   - Plik: results/paper10_ewi_v2.json
   - Struktura:
     {
       "pca": {"explained_variance": [...], "n_components": ...},
       "models": {
         "logit_balanced": {"pr_auc": ..., "f1": ..., "recall": ..., "precision": ..., "balanced_acc": ...},
         ...
       },
       "best_model": "...",
       "threshold": ...,
       "event_study": {
         "alerts_count": ...,
         "avg_return_7d": ...,
         "avg_maxdd_14d": ...,
         "vs_random": {...}
       },
       "time_series": [{"time": "...", "ewi": ..., "alert": 0|1, "rv_30d": ..., "high_vol": 0|1}, ...]
     }

7. Kod
   - Nowy plik: run_paper10_ewi_v2.py
   - Biblioteki: pandas, numpy, scikit-learn, ewentualnie imblearn (opcjonalnie).
   - Żadnego look-ahead w skalowaniu, PCA, quantile targetu ani thresholdzie.

Na końcu wypisz w konsoli tabelę porównawczą modeli (PR-AUC, F1, Recall, Balanced Acc).
```

---

## PROMPT 3 — Paper 3 (Kalkan & Tatlı 2022) — Pełny ARDL + bounds test  
**Priorytet: WYSOKI**

```
Jesteś ekonometrykiem specjalizującym się w modelach ARDL i testach kointegracji. Pracujesz w analytical_pipeline.

CEL
Odtworzyć formalnie pracę Kalkan & Tatlı (2022) "Blockchain Analiz Göstergelerinin Bitcoin Fiyatı Üzerindeki Etkisi" (Ekonomik ve Sosyal Araştırmalar Dergisi, 18, 109–140) z użyciem prawdziwego ARDL bounds testu zamiast uproszczonej OLS z lagiem 1.

WYMAGANIA

1. Dane
   - derived_metrics.csv, resampling do częstotliwości TYGODNIOWEJ (resample('W').last() lub .mean() – wybierz i uzasadnij).
   - Zmienne (mapowanie):
     * Y = log(PriceUSD)
     * SOPR (lub NUPL jeśli SOPR niedostępny)
     * PM_proxy (RevUSD / IssTotUSD)
     * AdrActCnt (log)
   - Opcjonalnie dodaj HashRate lub PuellMultiple jako robustness.

2. Procedura (dokładnie w tej kolejności)
   a) Testy stacjonarności ADF i KPSS dla wszystkich zmiennych (poziomy i pierwsze różnice). Określ rząd integracji I(0)/I(1). Żadna zmienna nie może być I(2).
   b) Wybór lagów ARDL: maximise AIC (lub SIC) z max_lag=8 dla tygodniowych danych. Użyj pmdarima.auto_ardl jeśli dostępne, albo ręcznie przeszukaj.
   c) Estymacja ARDL(p, q1, q2, q3) w formie levels.
   d) Bounds test (Pesaran, Shin, Smith):
      - H0: brak kointegracji
      - Porównaj F-statistic z critical values dla I(0) i I(1) (case III – unrestricted intercept, no trend).
      - Jeśli F > I(1) bound → kointegracja.
   e) Jeśli kointegracja:
      - Oszacuj Error Correction Model (ECM)
      - Long-run multipliers
      - Error Correction Term (ECT) – powinien być ujemny i istotny.
   f) Toda–Yamamoto Granger causality (jak w oryginale) między Y a każdą z regressorów.

3. Diagnostyka
   - Serial correlation (Breusch-Godfrey)
   - Heteroskedasticity (Breusch-Pagan lub White)
   - Normalność reszt (Jarque-Bera)
   - CUSUM / CUSUMSQ stabilności parametrów

4. Output
   - results/paper3_ardl_v2.json zawierający:
     * adf_kpss results
     * selected_lags
     * bounds_test: {F_stat, I0_bound, I1_bound, decision}
     * long_run_multipliers
     * ecm_results (współczynnik ECT, p-value)
     * toda_yamamoto
     * diagnostics
     * short_run_coefficients

5. Kod
   - Nowy plik: run_paper3_ardl_v2.py
   - Preferowane biblioteki: statsmodels, arch (jeśli potrzeba), pmdarima (opcjonalnie).
   - Jeśli nie ma gotowej implementacji bounds testu – zaimplementuj F-test na podstawie restrykcji jak w Pesaran et al. (2001).
   - Wszystkie testy na danych tygodniowych po 2012-01-01 (jak w oryginale) + pełna próba jako robustness.

Wypisz w konsoli jasny werdykt: "Kointegracja: TAK/NIE" oraz znaki i istotność long-run multipliers.
```

---

## PROMPT 4 — Paper 6 (Transformer vs LSTM) — pełny trening  
**Priorytet: WYSOKI**

```
Jesteś specjalistą od deep learning dla zmienności kryptowalut. Pracujesz w analytical_pipeline.

CEL
Poważnie odtworzyć i porównać Transformer vs LSTM do prognozy zrealizowanej zmienności Bitcoina zgodnie z pracą Rafi et al. (2024) "Cryptocurrency Volatility Forecasting Using Transformer-Based Deep Learning Models and On-Chain Metrics" (JEFAS 6(1)).

ORYGINALNY PROBLEM
Obecny kod trenuje tylko 5 epok na bardzo małych modelach – to nie jest porównanie.

WYMAGANIA

1. Target
   - rv_30d (zannualizowana 30-dniowa zrealizowana zmienność) lub rv_7d.
   - Alternatywnie: realized variance = sum of squared log-returns over next h days (h=1,7).

2. Features (on-chain + cena)
   - AdrActCnt, TxCntSec, TxTfrValUSD, FeeTotUSD, FeeMeanUSD, HashRate, NUPL, MVRV_Z, RevUSD, r_t, rv_7d (lagged).
   - Wszystkie zestandaryzowane (StandardScaler fit tylko na train).

3. Sequence length
   - Przetestuj 14, 30 i 60 dni. Główny wynik dla 30.

4. Modele
   a) LSTM: 2 warstwy, hidden=64, dropout=0.2
   b) Transformer Encoder: d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1
   - Oba z linear head przewidującym skalarną zmienność.

5. Trening (obowiązkowe)
   - Time-series split: ostatnie 20% jako test, z pozostałych 15% jako validation.
   - Early stopping: patience=15 na validation loss (MSE).
   - Max epochs=150
   - Optimizer: Adam, lr=1e-3, ReduceLROnPlateau
   - Batch size=64
   - 3 różne seedy → uśrednij wyniki.
   - Loss: MSE (ew. Huber).

6. Baseline’y (obowiązkowe)
   - Naive: ostatnia znana rv
   - GARCH(1,1) (arch library)
   - HAR-RV (regresja na rv_d, rv_w, rv_m)

7. Metryki na teście
   - MSE, RMSE, MAE, MAPE
   - Diebold-Mariano test (Transformer vs LSTM oraz vs najlepszy baseline)

8. Output
   - results/paper6_dl_v2.json
   - Dodatkowo zapisz najlepsze wagi modeli (opcjonalnie).
   - Wypisz tabelę: Model | RMSE | MAE | MAPE | DM p-value vs LSTM

9. Kod
   - Nowy plik: run_paper6_dl_v2.py
   - PyTorch (już jest w environment).
   - Kod ma być deterministyczny przy ustalonym seedzie.
```

---

## PROMPT 5 — Paper 7 (Omole & Enke 2025) — Klasyfikacja kierunku  
**Priorytet: WYSOKI**

```
Jesteś specjalistą od machine learning w tradingu kryptowalut. Pracujesz w analytical_pipeline.

CEL
Odtworzyć kluczowe elementy pracy Omole & Enke (2025) "Using Machine and Deep Learning Models, On-chain Data, and Technical Analysis for Predicting Bitcoin Price Direction and Magnitude" (Engineering Applications of Artificial Intelligence 154).

ORYGINALNY PROBLEM
Obecny kod ma accuracy ~47–51% (rzut monetą). Oryginał raportuje 83% accuracy / 82% F1 dzięki Boruta + bogatszemu feature setowi.

WYMAGANIA

1. Target
   - Główny: dir_h = 1 jeśli PriceUSD[t+h] > PriceUSD[t], h ∈ {1, 3, 7}
   - Dodatkowy (opcjonalnie): magnitude (regresja log-return)

2. Feature set (zbuduj bogaty)
   - On-chain: AdrActCnt, TxTfrValUSD, FeeTotUSD, RevUSD, HashRate, NUPL, MVRV_Z, SOPR, NetFlowUSD, ShareExUSD, PuellMultiple, CDD (jeśli jest)
   - Price-based / TA proxy: r_t, rv_7d, rv_30d, rolling mean/std PriceUSD (7, 30), momentum 7/30
   - Lagged versions wybranych cech (lag 1, 3, 7)

3. Feature selection
   - Zaimplementuj Boruta (jeśli boruta_py lub odpowiednik dostępny) LUB
   - Recursywna eliminacja (RFECV) z RandomForest LUB
   - SHAP importance z RandomForest (top 20 cech)
   - Porównaj wyniki z pełnym zestawem vs wyselekcjonowanym.

4. Modele
   - RandomForestClassifier
   - LinearSVC / SVC(kernel='rbf')
   - GradientBoostingClassifier lub XGBoost/LightGBM (jeśli dostępne)
   - Opcjonalnie: prosty MLP

5. Ewaluacja
   - Purged time-series split lub expanding window (minimum 5 foldów)
   - Metryki: Accuracy, Precision, Recall, F1, Balanced Accuracy, MCC
   - Ekonomiczna: średni zwrot strategii "long gdy model przewiduje 1, inaczej flat" (z kosztem 0.1%)
   - Feature importance / SHAP summary

6. Output
   - results/paper7_classification_v2.json
   - Tabela porównawcza modeli dla h=1,3,7

7. Kod
   - Nowy plik: run_paper7_ml_v2.py
   - scikit-learn + ewentualnie shap, boruta_py, xgboost (graceful fallback jeśli brak).
```

---

## PROMPT 6 — Paper 1 (Wüstenfeld 2023) — GARCH poprawiony  
**Priorytet: ŚREDNI**

```
Jesteś ekonometrykiem specjalizującym się w modelach GARCH. Pracujesz w analytical_pipeline.

CEL
Poprawić Paper 1 (Wüstenfeld 2023, SSRN "Bitcoin Network Activity and Bitcoin Price Return Volatility") poprzez dodanie robust inference, lagów i diagnostyki.

WYMAGANIA

1. GARCH(1,1) na r_t × 100 (jak dotychczas) + diagnostyka:
   - Ljung-Box na standaryzowanych resztach (lag 10, 20)
   - ARCH-LM test
   - Rozważ też t-Student distribution (vol='Garch', dist='t')

2. Regresja h_t na aktywności
   - Dodaj lag 1 i lag 7 metryk aktywności
   - Użyj HAC / Newey-West standard errors (statsmodels get_robustcov_results lub cov_type='HAC')
   - Multiple testing: Bonferroni lub Benjamini-Hochberg na p-value

3. Alternatywa: GARCH-X
   - Włącz wybrane metryki aktywności bezpośrednio do równania wariancji (arch library wspiera)

4. Robustness
   - Osobne estymacje: 2013–2017, 2018–2020, 2021–2026
   - Sprawdź stabilność znaku współczynników

5. Output
   - results/paper1_garch_v2.json z pełnymi diagnostykami i tabelą współczynników (coef, HAC se, p-value, significant_bonferroni)

6. Kod
   - Nowy plik: run_paper1_garch_v2.py
```

---

## Jak używać tych promptów

1. Wklejaj **jeden prompt na raz** do agenta (Cursor / Claude / Aider / etc.).
2. Agent ma pracować w katalogu `analytical_pipeline`.
3. Po każdym paperze sprawdzaj plik JSON w `results/` i czy kod nie ma look-ahead (agent powinien sam to weryfikować).
4. Najpierw zrób **Paper 4** i **Paper 10** — to daje największy zwrot metodologiczny.

Chcesz, żebym od razu przygotował też wersję promptów dla Paper 2, 5, 8 i 9 (niższy priorytet), albo jeden wspólny prompt orchestrujący wszystkie poprawki w `run_all_v2.sh`?

## human
teraz zobacz repozytorium jeszcze raz po zmianach. oceń czy zostało wykonane wszystko. chces wykonać kolejne prompty?

## assistant
**Status po ponownym sprawdzeniu repozytorium (5 sierpnia 2026)**

### Co jest w `analytical_pipeline` teraz

| Element | Stan |
|---------|------|
| Pliki `*_v2.py` | **Brak** |
| `strategy_results_v2.json` / `paper10_ewi_v2.json` itd. | **Brak** |
| `run_strategies.py` | Nadal liczy kwantyle 20/80% na **całej historii** (look-ahead bias) |
| `run_ml_dl.py` | Nadal `epochs = 5` |
| `run_econometrics.py` | Nadal uproszczone proxy ARDL, brak HAC SE, brak bounds testu |
| `results/` | Tylko stare pliki: `derived_metrics.csv`, `econometrics_results.json`, `ml_results.json`, `strategy_results.json` (ostatnia aktualizacja ~21 lipca 2026) |
| Nowe commity z poprawkami metodologicznymi | **Brak** |

Ostatnie widoczne zmiany w folderze to „Update indicators & dashboard before GCP integration” z 21 lipca. Od tego czasu (i od momentu, gdy dostałeś prompty) **żadna z rekomendowanych poprawek nie została wdrożona**.

### Werdykt

**Nie zostało wykonane nic** z listy krytycznych/high-priority promptów:

- Paper 4 (strategia MVRV/NUPL) – look-ahead bias nadal obecny  
- Paper 10 (PCA + EWI) – imbalance nadal nieobsłużony  
- Paper 3 (pełny ARDL + bounds) – brak  
- Paper 6 (Transformer/LSTM z proper trainingiem) – nadal 5 epok  
- Paper 7 (Boruta + lepsza klasyfikacja) – brak  
- Paper 1 (GARCH z HAC + diagnostyka) – brak  

Kod i wyniki są w dokładnie takim samym stanie metodologicznym, jaki oceniałem wcześniej.

---

### Czy chcę wykonać kolejne prompty?

**Tak – warto.**  

Największy zwrot dadzą dwa pierwsze:

1. **Paper 4** (strategia) – to najpoważniejszy błąd metodologiczny i jednocześnie najbardziej praktyczny wynik.
2. **Paper 10** (EWI) – drugi krytyczny punkt.

Potem sensownie iść kolejno: Paper 3 → Paper 6 → Paper 7 → Paper 1.

Mogę:

- od razu zacząć implementować Paper 4 (najwyższy priorytet) w sandboxie i oddać gotowy `run_strategies_v2.py` + wyniki,
- albo przygotować zaktualizowane, jeszcze bardziej precyzyjne prompty pod konkretne środowisko/agenta, którego używasz,
- albo zrobić pełny plan wykonania wszystkich promptów z kolejnością i checklistą.

Napisz, którą opcję wybierasz (albo „rób Paper 4 teraz”), to ruszam.

## human
spróbuj jeszcze raz .. może trał jeszcze push. mowimy o https://github.com/kknckk/btc-tai-indicators

## assistant
**Tak – był push.**  
Commit z **5 sierpnia 2026**:

`feat(analytical_pipeline): reimplement and enhance 6 research papers`  
(`2f1dc646…`)

---

## Co zostało zrobione

Zaimplementowano **wszystkie 6 krytycznych / high-priority paperów** z promptów:

| Paper | Plik skryptu | Plik wyników | Status |
|-------|--------------|--------------|--------|
| **4** Näsman (strategia MVRV/NUPL) | `run_strategies_v2.py` | `strategy_results_v2.json` | ✅ Zrobione |
| **10** EWI / PCA high-vol | `run_paper10_ewi_v2.py` | `paper10_ewi_v2.json` | ✅ Zrobione |
| **3** ARDL + bounds | `run_paper3_ardl_v2.py` | `paper3_ardl_v2.json` | ✅ Zrobione |
| **6** Transformer vs LSTM | `run_paper6_dl_v2.py` | **brak JSON** (404) + są wagi `.pt` | ⚠️ Skrypt jest, wyników brak |
| **7** Klasyfikacja + Boruta | `run_paper7_ml_v2.py` | `paper7_classification_v2.json` | ✅ Zrobione |
| **1** GARCH + HAC | `run_paper1_garch_v2.py` | `paper1_garch_v2.json` | ✅ Zrobione |

Dodatkowo: `SESSION_SUMMARY.md` + zaktualizowany `run_all.sh`.

---

## Ocena jakości implementacji (czy zrobione **dobrze**)

### Paper 4 (strategia) — **bardzo dobrze**
- Look-ahead bias **usunięty**: expanding window liczy kwantyle ściśle do `t-1`.
- Jest rolling 4y + progi literaturowe + koszty 20 bps.
- Pozycja shiftowana o 1 dzień.
- Wyniki (z kosztami):
  - **fixed_literature**: Sharpe **1.19**, CAGR 64%, MaxDD −66% (lepszy Sharpe niż B&H 1.14)
  - expanding: Sharpe 0.75
  - rolling_4y: Sharpe 0.70, ale wyraźnie niższy MaxDD (−60%)
- Robustness po okresach też jest.

To jest największa poprawa metodologiczna w całym repo.

### Paper 10 (EWI) — **dobrze, z zastrzeżeniami**
- Expanding quantile target + `class_weight='balanced'` + PR-AUC jako główna metryka — zgodne z promptem.
- Event study jest (alerty dają wyższe średnie zwroty niż random).
- PR-AUC nadal niski (~0.027), recall ~0.43, precision bardzo niski. Model „widzi” burze, ale generuje dużo fałszywych alarmów. To realistyczne przy rzadkiej klasie, ale próg (0.83) i ewentualnie SMOTE/kalibracja da się jeszcze dostroić.

### Paper 3 (ARDL) — **dobrze strukturalnie**
- Pełna procedura: ADF/KPSS → wybór lagów → bounds test → ECM → Toda–Yamamoto → diagnostyka (BG, BP, JB, CUSUM).
- Dane tygodniowe od 2012.
- Struktura JSON zgodna z wymaganiami. (Szczegółowe wartości F-stat / decision warto jeszcze ręcznie sprawdzić.)

### Paper 1 (GARCH) — **dobrze**
- GARCH normal vs t-Student, Ljung-Box, ARCH-LM, HAC SE, Bonferroni/FDR, podokresy, GARCH-X — wszystko z promptu jest.

### Paper 7 (klasyfikacja) — **częściowo**
- 72 cechy, selekcja, purged split, horyzonty 1/3/7, koszty 10 bps — tak.
- Accuracy nadal ~50–53% (XGBoost najlepszy ~53%). Kierunek dzienny jest obiektywnie bardzo trudny; wynik nie jest dramatycznie gorszy od rzeczywistości, ale daleko od 83% z oryginalnego paperu Omole & Enke (tam był m.in. Boruta + inny zestaw). Warto sprawdzić, czy Boruta faktycznie się uruchomił, czy spadło do top-20 RF importance.

### Paper 6 (DL) — **niedokończone**
- Skrypt wygląda solidnie (early stopping patience=15, max 150 epok, HAR-RV + GARCH baseline, Diebold-Mariano, sekwencje 14/30/60).
- **Brakuje `paper6_dl_v2.json`** w `results/`. Są tylko wagi `best_transformer_weights.pt`.  
  → Albo run się wywalił, albo wyniki nie zostały zacommitowane.

---

## Podsumowanie

| Kryterium | Ocena |
|-----------|--------|
| Czy wszystkie 6 promptów zostały podjęte? | **Tak** |
| Czy krytyczne błędy (look-ahead, imbalance) naprawione? | **Tak** (Paper 4 i 10) |
| Czy wyniki są kompletne? | **5/6** (brak JSON z Paper 6) |
| Czy jakość jest na poziomie „publikowalnym”? | Paper 4 i 1 – blisko; reszta solidny research-grade PoC |
| Czy trzeba jeszcze coś pilnie poprawić? | **Tak – Paper 6 (dopisać/zapisać wyniki) + ewentualnie dociągnąć Paper 7 i próg EWI** |

---

## Co dalej?

Najbardziej sensowne kolejne kroki:

1. **Dobić Paper 6** – uruchomić ponownie `run_paper6_dl_v2.py` i zacommitować `paper6_dl_v2.json`.
2. Krótki audit Paper 7 (czy Boruta działał, czy tylko top-20 RF).
3. Opcjonalnie: Paper 2, 5, 8, 9 (niższy priorytet) – jeśli chcesz domknąć całą dziesiątkę.

Chcesz, żebym teraz:
- dokładnie przejrzał wyniki Paper 3 / 1 / 7 (konkretne liczby i werdykt „czy teza oryginału się potwierdza”),
- albo przygotował prompt naprawczy tylko pod brakujący Paper 6,
- albo od razu ocenił, które z nowych wyników są warte pokazania na dashboardzie?

## human
to zaproponuj kolejne prompty

## assistant
Poniżej propozycja **kolejnych promptów** — ułożone według priorytetu, maksymalnie szczegółowe, gotowe do wklejenia agentowi.

---

## PROMPT A — Paper 6: dokończenie i zapis wyników (NAJWYŻSZY PRIORYTET)

```
Jesteś ekspertem od deep learning dla zmienności kryptowalut. Pracujesz w analytical_pipeline repozytorium btc-tai-indicators.

STAN OBECNY
- Istnieje run_paper6_dl_v2.py (wygląda poprawnie: early stopping, HAR-RV, GARCH, Diebold-Mariano, sekwencje 14/30/60).
- Istnieją wagi results/best_transformer_weights.pt.
- BRAKUJE pliku results/paper6_dl_v2.json (HTTP 404). Wyniki nie zostały zapisane lub run się wywalił przed zapisem.

CEL
1. Uruchom (lub napraw i uruchom) run_paper6_dl_v2.py tak, aby na końcu POWSTAŁ kompletny results/paper6_dl_v2.json.
2. Upewnij się, że skrypt jest odporny na błędy (try/except wokół treningu, zawsze zapisuje to, co udało się policzyć).
3. JSON musi zawierać minimum:
   {
     "paper6_dl_v2": {
       "sequence_lengths": {
         "14": {"LSTM": {"RMSE":..., "MAE":..., "MAPE":...}, "Transformer": {...}, "epochs_used": ...},
         "30": {...},
         "60": {...}
       },
       "main_sequence_length": 30,
       "baselines": {
         "naive": {...},
         "GARCH_1_1": {...},
         "HAR_RV": {...}
       },
       "diebold_mariano": {
         "Transformer_vs_LSTM": {"dm_stat":..., "p_value":...},
         "Transformer_vs_HAR_RV": {...},
         "LSTM_vs_HAR_RV": {...},
         "Transformer_vs_GARCH": {...}
       },
       "training_config": {
         "max_epochs": 150,
         "patience": 15,
         "d_model": ...,
         "seeds": [...]
       },
       "best_model": "Transformer" | "LSTM" | "HAR_RV" | ...,
       "notes": "..."
     }
   }

4. Jeśli trening jest zbyt długi lokalnie:
   - zostaw early stopping,
   - możesz zmniejszyć max_epochs do 80 przy patience=10,
   - ale NIE usuwaj baseline’ów ani testu Diebold-Mariano.

5. Po zakończeniu wypisz w konsoli tabelę:
   Model | SeqLen | RMSE | MAE | MAPE | DM p-value vs HAR-RV

6. Zacommituj tylko results/paper6_dl_v2.json (i ewentualne poprawki skryptu). Nie ruszaj innych paperów.

Sprawdź na końcu, że plik istnieje i da się wczytać json.load().
```

---

## PROMPT B — Paper 7: weryfikacja Boruty + poprawa jakości

```
Jesteś specjalistą od ML w tradingu kryptowalut. Pracujesz w analytical_pipeline.

STAN OBECNY
- run_paper7_ml_v2.py i paper7_classification_v2.json istnieją.
- Accuracy nadal ~50–53% (h=1). W selected_features widać głównie TA + lags zwrotów; mało czystych on-chain.
- Nie jest jasne, czy Boruta faktycznie się uruchomił, czy spadło do „Top20 RF Importance”.

CEL
1. Przeczytaj run_paper7_ml_v2.py i paper7_classification_v2.json.
2. Zweryfikuj:
   - Czy biblioteka boruta / boruta_py jest użyta?
   - Jeśli nie – zaimplementuj prawdziwy Boruta (lub boruta_py) z RandomForest jako estymatorem.
   - Jeśli Boruta pada na NumPy – zastosuj znaną poprawkę (np. np.int → int) i obsłuż fallback.

3. Wymagania feature set (minimum):
   - On-chain: AdrActCnt, TxTfrValUSD, FeeTotUSD, RevUSD, HashRate, NUPL, MVRV_Z, SOPR (lub proxy), NetFlowUSD, ShareExUSD, PuellMultiple, CDD (jeśli jest)
   - TA: RSI(14), MACD, Bollinger %B, momentum 7/30
   - Lags: 1, 3, 7 dla kluczowych cech
   - Rolling: mean/std r_t i rv na 7/30

4. Horyzonty: h ∈ {1, 3, 7}
5. Ewaluacja: purged / expanding time-series split (min. 5 foldów).
6. Modele: RandomForest, XGBoost (lub GradientBoosting), SVC, ewentualnie MLP.
7. Metryki: Accuracy, Balanced Accuracy, F1, MCC, Precision, Recall.
8. Ekonomiczna: strategia Long/Flat z kosztem 10 bps → cum return + Sharpe.
9. Porównaj:
   - pełny feature set
   - tylko cechy wybrane przez Borutę
   - tylko top-20 RF importance (jako baseline)

10. Output: nadpisz lub stwórz results/paper7_classification_v2.json z jasnym polem:
    "feature_selection": {
      "method_used": "Boruta" | "Top20_RF_fallback",
      "n_selected": ...,
      "selected_features": [...],
      "boruta_success": true/false,
      "error_if_any": "..."
    }

11. W konsoli wypisz:
    - czy Boruta zadziałał
    - top 15 cech
    - tabelę Accuracy / F1 / StrategySharpe dla h=1,3,7

Nie obniżaj horyzontu poniżej 1. Nie używaj shuffle=True.
```

---

## PROMPT C — Paper 10: poprawa precyzji EWI (kalibracja + threshold)

```
Jesteś ekspertem od Early Warning Indicators i klasyfikacji rzadkich zdarzeń. Pracujesz w analytical_pipeline.

STAN OBECNY
- run_paper10_ewi_v2.py i paper10_ewi_v2.json istnieją.
- PR-AUC ≈ 0.027, recall ≈ 0.43, precision bardzo niska.
- Event study pokazuje, że alerty mają wyższe średnie zwroty niż random – sygnał jest, ale za dużo false positives.
- best_model = logit_balanced, threshold = 0.83.

CEL
Poprawić jakość EWI bez psucia expanding-window i bez look-ahead.

1. Zostaw expanding quantile (95%) jako target.
2. Dodaj / porównaj:
   a) Probabilistyczną kalibrację (CalibratedClassifierCV, method='isotonic' lub 'sigmoid') na validation foldach.
   b) Wybór thresholdu maksymalizujący F1 ORAZ osobno maksymalizujący Precision przy recall ≥ 0.25.
   c) Opcjonalnie: cost-sensitive threshold (np. koszt FP vs FN w stosunku 1:3 lub 1:5).
3. Modele do porównania (wszystkie z class_weight lub balancing):
   - LogisticRegression balanced + kalibracja
   - RandomForest balanced + kalibracja
   - HistGradientBoostingClassifier (jeśli dostępny) z class_weight
4. Raportuj na rolling/expanding evaluation:
   - PR-AUC, ROC-AUC, F1, Precision, Recall, Balanced Accuracy
   - Liczbę alertów / rok
   - Event study: avg return i maxDD w oknach 7/14/30 dni po alercie vs random (1000 losowań)
5. Dodaj prosty „alert quality” score = (avg_return_14d_alert - avg_return_14d_random) / |maxDD_14d_alert|

6. Output: zaktualizuj results/paper10_ewi_v2.json o sekcje:
   "calibration": {...},
   "threshold_optimization": {
     "max_f1": {"threshold":..., "precision":..., "recall":..., "f1":...},
     "min_precision_at_recall_0.25": {...}
   },
   "event_study_v2": {...},
   "alert_quality_score": ...

7. W konsoli wypisz porównanie przed/po kalibracji i rekomendowany próg do użycia na dashboardzie.

Nie używaj przyszłych danych przy wyborze thresholdu – tylko validation foldy.
```

---

## PROMPT D — Audit jakości wszystkich v2 + checklista

```
Jesteś audytorem metodologii badań ilościowych. Pracujesz w analytical_pipeline.

CEL
Przeprowadź systematyczny audit wszystkich plików v2 i przygotuj krótki raport markdown.

Sprawdź dla każdego z:
- run_strategies_v2.py + strategy_results_v2.json
- run_paper10_ewi_v2.py + paper10_ewi_v2.json
- run_paper3_ardl_v2.py + paper3_ardl_v2.json
- run_paper1_garch_v2.py + paper1_garch_v2.json
- run_paper7_ml_v2.py + paper7_classification_v2.json
- run_paper6_dl_v2.py (+ paper6_dl_v2.json jeśli istnieje)

Dla każdego paperu odpowiedz NA TAK/NIE + jedno zdanie uzasadnienia:

1. Zero look-ahead bias (progi, skaler, target, threshold liczone tylko na przeszłości)?
2. Time-series split / expanding / purged (brak shuffle)?
3. Koszty transakcyjne uwzględnione tam, gdzie jest strategia?
4. Główna metryka jest adekwatna (PR-AUC zamiast accuracy przy imbalance, HAC SE przy regresjach itd.)?
5. Wyniki JSON da się wczytać i zawierają kluczowe pola z oryginalnego promptu?
6. Czy teza oryginalnego artykułu jest potwierdzona / obalona / nierozstrzygnięta na podstawie nowych wyników?

Dodatkowo:
- Wypisz 3 najważniejsze residual riski metodologiczne, które jeszcze zostały.
- Zaproponuj, które 2–3 wyniki nadają się do pokazania na dashboardzie (z uzasadnieniem).

Output: zapisz plik analytical_pipeline/AUDIT_V2.md
```

---

## PROMPT E — Paper 2 (NetFlows) — opcjonalny, średni priorytet

```
Jesteś ekonometrykiem. Pracujesz w analytical_pipeline.

CEL
Zaimplementuj poprawioną wersję Paper 2 (Chi, Chu, Hao 2024 – on-chain flows → returns & volatility) jako run_paper2_netflows_v2.py.

Wymagania:
1. NetFlowUSD = FlowInExUSD - FlowOutExUSD
2. Znormalizuj przepływy: NetFlow_z = NetFlowUSD / CapMrktCurUSD (lub log1p(|NetFlow|) * sign)
3. Modele:
   - OLS z HAC SE: r_t ~ NetFlow_z + lag1 + lag7
   - OLS z HAC SE: rv_7d / rv_30d ~ NetFlow_z + lagi
   - Opcjonalnie: local projections (Jordà) na h=1..7
4. Granger causality (1–7 lagów)
5. Robustness: podokresy 2017–2020, 2021–2023, 2024–2026
6. Zero look-ahead.
7. Output: results/paper2_netflows_v2.json

Nie mieszaj z innymi paperami.
```

---

## PROMPT F — Integracja z dashboardem (po A–C)

```
Jesteś full-stack developerem projektu btc-tai-indicators.

CEL
Wyeksponuj najlepsze wyniki v2 na froncie (dashboard Next.js) i/lub przez API.

1. Wybierz maksymalnie 3 sygnały/metryki, które mają sens produkcyjny:
   - Strategia fixed_literature (Paper 4) – aktualna pozycja + equity
   - EWI alert (Paper 10) – po wybranym thresholdzie
   - Ewentualnie long-run / ECT z ARDL (Paper 3) jako kontekst

2. Dodaj endpointy w api/ (FastAPI) np.:
   - GET /api/v1/research/strategy/status
   - GET /api/v1/research/ewi/current
   które czytają najnowsze JSON-y z analytical_pipeline/results/

3. Na dashboardzie (Next.js) dodaj prostą sekcję „Research Signals” z:
   - aktualnym stanem strategii (Long/Flat)
   - poziomem EWI + flagą alert
   - datą ostatniego przeliczenia

4. Nie zmieniaj istniejących wykresów wskaźników on-chain.
5. Zadbaj o graceful fallback, gdy plik JSON nie istnieje.

Commit message: feat(dashboard): expose v2 research signals (strategy + EWI)
```

---

## Rekomendowana kolejność wykonania

| Kolejność | Prompt | Dlaczego |
|-----------|--------|----------|
| 1 | **A** (Paper 6 JSON) | Jedyny brakujący artefakt z już napisanego kodu |
| 2 | **B** (Paper 7 Boruta) | Wyjaśnia, czy 50% accuracy to limit problemu, czy błąd selekcji cech |
| 3 | **C** (Paper 10 kalibracja) | Podnosi użyteczność EWI pod dashboard |
| 4 | **D** (Audit) | Domknięcie i dokumentacja stanu |
| 5 | **E** (Paper 2) | Opcjonalne uzupełnienie dziesiątki |
| 6 | **F** (Dashboard) | Dopiero gdy A–C są stabilne |

---

Chcesz, żebym od razu rozwinął któryś z tych promptów (np. A albo B) o jeszcze więcej szczegółów implementacyjnych, albo żebym przygotował jeden „master prompt” orchestrujący A→B→C→D?

## human
zrobilem prompty i jeszcze rozszerzylem zakres projektu o ten plan:
Rozbudowa i produkcjonalizacja btc-tai-indicators
Ten plan opisuje wdrożenie 4 głównych funkcji, które przekształcą projekt z repozytorium badawczego w działający system klasy produkcyjnej "Live Quant Dashboard".
User Review Required
IMPORTANT
Proszę o zatwierdzenie kierunku i potwierdzenie gotowości dostarczenia kilku kluczowych informacji w trakcie lub po wdrożeniu:

1. Do integracji "Chat with Data" potrzebny będzie dostęp do klucza GEMINI_API_KEY (lub OpenAI). Skonfigurujemy to poprzez plik .env.

2. Webhooki: Skonfigurujemy system by wspierał generyczne webhooki URL (np. dla Discorda lub Telegrama). Konfiguracja adresów będzie w pliku środowiskowym.

3. CI/CD: Czy chcesz abym napisał pełen plik deploymentu dla Google Cloud Run Jobs, czy wystarczy na ten moment skrypt cron/github actions? Domyslnie przygotuję instrukcje i skrypty dla Google Cloud Scheduler + Cloud Run.

Proposed Changes
Frontend Dashboard - Podstrony Badawcze V2
Uaktualnimy interfejsy specyficzne dla Paperów. Wykresy (Recharts) będą rysowane na podstawie zaktualizowanych danych V2 bez look-ahead biasu.
[MODIFY] api/routers/literature.py lub nowe endpointy
Dopracowanie endpointów, które parsują nasze pliki *_v2.json (np. paper10_ewi_v2.json, strategy_results_v2.json) w odpowiednie struktury dla Recharts (time-series, scatter data, confusion matrix).
[MODIFY] dashboard/src/components/literature/Paper10Dashboard.tsx
Podłączenie nowego API, pokazanie wykresów prawdopodobieństwa alertów EWI i statystyk kalibracji progu.
[MODIFY] dashboard/src/components/literature/Paper4Dashboard.tsx
Podłączenie strategii "Expanding window" i narysowanie historycznej krzywej kapitału OOS zamiast mockowanych danych.
[MODIFY] dashboard/src/components/literature/Paper3Dashboard.tsx
Pokazanie wyników relacji kointegracji w długim terminie w formie zgrabnej tabeli i mnożników z pliku paper3_ardl_v2.json.
Automatyzacja Pipeline'u (CI/CD)
Projekt musi "żyć z rynkiem" generując sygnały codziennie, bazując na najświeższych danych on-chain.
[NEW] cloud_run_jobs/deploy_job.sh
Skrypt uruchamiający kontener, który odpali run_all.sh (pobierze dane derived_metrics.csv i zaktualizuje pliki V2).
[NEW] cloud_run_jobs/scheduler.tf (Opcjonalnie) lub skrypt GCloud
Skrypt dla Google Cloud Scheduler konfigurujący wyzwalanie Joba codziennie o wyznaczonej godzinie.
System Alertów (Webhook)
Proaktywne informowanie o wczesnych ostrzeżeniach.
[NEW] api/services/alerts.py (lub skrypt Pythona w pipeline)
Serwis odczytujący JSONy (np. EWI i Strategię) natychmiast po przebudowie. Jeśli wykryto zapaloną flagę alert_triggered, generuje on bogatą wiadomość tekstową (Markdown/Embed) z wartościami on-chain, wysyłając ją przez standardowy protokół Webhook.
Integracja LLM - Chat with Data
Tworzymy unikatową funkcjonalność "Quant Analyst AI", dając użytkownikowi możliwość zinterpretowania nagłych sygnałów rynkowych.
[MODIFY] api/requirements.txt
Dodanie pakietu API LLM (np. google-genai).
[NEW] api/routers/chat.py
Endpoint FastAPI /api/v1/chat, który przyjmuje polecenie, czyta derived_metrics.csv (ostatnie wiersze) i wyniki JSON, buduje "System Prompt" (kontekst rynkowy) i zwraca analityczną odpowiedź modelu językowego.
[NEW] dashboard/src/components/ChatAgent.tsx
Pływający lub wbudowany komponent czatu w interfejsie Next.js z integracją tego endpointu.
Verification Plan
Automated / Manual Verification

* Uruchomienie lokalnie serwera FastAPI i weryfikacja nowych endpointów w trybie Swagger (/docs).

* Wywołanie endpointów testowych Webhooka na fikcyjnym kanale (logi konsoli).

* Uruchomienie komponentów w Next.js i weryfikacja wizualnej zgodności (czy wykresy ładują się pomyślnie z JSON V2).

* Test wywołania Gemini / LLM za pomocą testowego stringa i podgląd odpowiedzi (Mock lub Live, jeśli klucz zostanie podany).

 
 
raz jeszcze zrób recenzję wszystkiego i zrób nowy docx z opisem przeprowadzonych badań (oraz. nowymi featureami). jak odtworzenie było trudne do zrobienia i trzeba było 3-4 razy powtarzać, to opisz tę historię dokładnie. dopisz oszacowania złożoności obliczeniowej szczególnie tam gdzie trzeba było uczyć sieć. dokładnie wskaż jaka to była sieć.

## assistant
**Raport wygenerowany.**



---

## Skrót recenzji (stan na 6 sierpnia 2026)

###<grok:render card_id="NKkZ3" card_type="rendered_file_card" type="render_file"><argument name="file_path">/home/workdir/artifacts/Raport_Koncowy_BTC_TAI_Badania_V2_Produkcja.docx</argument></grok:render> Co się wydarzyło od ostatniej oceny

| Data | Commit / zmiana |
|------|------------------|
| 5 VIII | `reimplement and enhance 6 research papers` — V2 paperów 1, 3, 4, 6, 7, 10 |
| 5 VIII | `fix run_paper6_dl_v2.py and save results` — uzupełniony brakujący JSON |
| 5–6 VIII | Paper 2 NetFlows V2 + `AUDIT_V2.md` |
| 6 VIII | `expose v2 research signals` + **`full v2 integration, webhook alerts, and LLM chat`** |

Pipeline badawczy V2 i cztery filary planu produkcyjnego są w repo.

### Historia trudności (4 iteracje) — w raporcie opisane szczegółowo

1. **Iteracja 0** — PoC z look-ahead, 5 epok DL, ARDL-proxy, EWI z precision=0  
2. **Iteracja 1** — recenzja + 6 promptów naprawczych  
3. **Iteracja 2** — reimplementacja V2 (największy skok)  
4. **Iteracja 3–4** — domknięcie Paper 6, audit, produkcja (dashboard, webhooki, chat, Cloud Run Jobs)

### Sieci neuronowe (Paper 6) — dokładnie

- **LSTMForecaster**: 2 warstwy LSTM, `hidden_dim=64`, dropout 0,2, głowa Linear(64→1)  
- **TransformerForecaster**: `d_model=64`, `nhead=4`, 2 warstwy encoder, `dim_feedforward=128`, PositionalEncoding, głowa Linear(64→1)  
- Trening: Adam 1e−3, max 80 epok, patience 10, 3 seedy, sekwencje L∈{14,30,60}  
- **Wynik uczciwy, ale negatywny dla „przewagi DL”**: na L=30 Transformer bije LSTM (RMSE 0,10 vs 0,16), ale **naive i HAR-RV mają RMSE ~0,024** — `best_model = naive`  
- Złożoność: ~18 pełnych treningów; szacunek 30–90 min CPU; wagi w `best_transformer_weights.pt`

### Najsilniejsze wyniki do pokazania

- **Paper 3**: kointegracja potwierdzona (F=12,52 > bound), ECT istotny  
- **Paper 4 fixed_literature**: Sharpe 1,19 vs B&H 1,14, niższy MaxDD, **bez look-ahead**  
- **Paper 10**: event study dodatni vs random; alert quality ≈ 0,23 (ostrzeżenie, nie timer)

### Produkcja — wdrożone

- `api/routers/literature.py`, `research.py`, `chat.py`  
- `api/services/alerts.py` (webhook EWI + strategia)  
- `ChatAgent.tsx`, `ResearchSignals.tsx`, literature dashboards  
- `cloud_run_jobs/deploy_job.sh`, `scheduler.sh`

### Residual risks (w raporcie)

- Nie sprzedawać DL jako lepszego od HAR-RV  
- Nie obiecywać 83% accuracy kierunku (V2 ≈ 50–53%)  
- EWI = early warning z false positives  
- Klucze API / webhook tylko w `.env`

Raport DOCX zawiera pełny opis iteracji, tabele wyników, architektury sieci, rzędy złożoności i checklistę produkcyjną.

## human
Paper 10 to jakie ma oryginalne xrodlo?

## assistant
**Paper 10 nie ma jednego, czystego źródła 1:1** — to najbardziej „skomponowany” z dziesięciu tematów.

### Co jest w kodzie
`run_paper10_ewi_v2.py` / wcześniej `run_paper10_pca` robi:
1. **PCA** na metrykach aktywności on-chain (TxCnt, AdrActCnt, FeeTotUSD itd.)
2. Klasyfikację dni **wysokiej zmienności** (`rv_30d` > 95. percentyl)
3. Traktowanie prawdopodobieństwa jako **EWI (Early Warning Indicator)**

### Skąd to się wzięło na liście badań

W pliku `znajdź dla mnie artykuły naukowe…pdf` **nie ma osobnego artykułu zatytułowanego „EWI + PCA”**. Paper 10 powstał jako **synteza** wątków z listy:

| Wątek | Źródło z listy / literatury |
|--------|-----------------------------|
| Silna współzależność miar aktywności → sens PCA | **Wüstenfeld (2023)** — *Bitcoin Network Activity and Bitcoin Price Return Volatility* (SSRN); autor sam pokazuje, że metryki aktywności są silnie skorelowane |
| Redukcja wymiaru / czynniki z on-chain | Typowe w pracach o activity / network metrics (m.in. wątki z Makarov & Schoar, prac o kohortach UTXO) |
| Ostrzeganie przed okresami wysokiej zmienności | Wzorzec **Early Warning Indicator** z literatury finansowej, przeniesiony na on-chain (nie jeden konkretny paper z Waszej listy PDF) |

W mapowaniu, które robiliśmy przy recenzji, Paper 10 był więc opisany jako:

> **PCA aktywności + klasyfikacja high-vol (EWI-style)** — inspiracja metodologiczna, nie wierna replika jednego artykułu.

### Dla porównania — reszta ma wyraźne źródła

| Paper | Oryginał |
|-------|----------|
| 1 | Wüstenfeld 2023 (SSRN) |
| 2 | Chi, Chu, Hao 2024 (arXiv) |
| 3 | Kalkan & Tatlı 2022 |
| 4 | Näsman 2025/2026 (RIBAF) |
| 5 | NTU thesis 2023 |
| 6 | Rafi et al. 2024 (JEFAS) |
| 7 | Omole & Enke 2025 (EAAI) |
| 8 | Scientific Data / Nature 2022 (cohort/CDD) |
| 9 | Makarov & Schoar 2021 |
| **10** | **Brak jednego paperu — synteza PCA + EWI** |

### Wniosek
Jeśli w dokumentacji / dashboardzie potrzebujesz cytowania dla Paper 10, najuczciwiej jest napisać np.:

> *Inspirowane obserwacją silnej kolinearności miar aktywności (Wüstenfeld 2023) oraz konstrukcją Early Warning Indicators dla okresów wysokiej zmienności; implementacja własna: PCA + expanding-window klasyfikacja rare events.*

Nie ma sensu dopisywać fikcyjnego tytułu „Article X, Journal Y, 20XX” — tego w Waszej liście PDF nie było.
