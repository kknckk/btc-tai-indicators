# Podsumowanie Sesji: Implementacja i Poprawa Modułów Badawczych (Bitcoin On-Chain)

Podczas tej sesji przeprowadziliśmy gruntowną refaktoryzację i reimplementację **sześciu kluczowych prac naukowych** wchodzących w skład repozytorium `btc-tai-indicators` (`analytical_pipeline`). Celem było wyeliminowanie błędów metodologicznych (np. *look-ahead bias*, problem niezbalansowanych klas), wprowadzenie rygorystycznych testów ekonometrycznych oraz wdrożenie zaawansowanych modeli uczenia maszynowego i głębokiego.

Wszystkie nowe skrypty zostały zintegrowane w głównym pliku uruchomieniowym: `run_all.sh`.

---

## 1. Paper 4: Näsman (2025/2026) – Strategia Timingowa (MVRV_Z, NUPL)
**Plik:** `run_strategies_v2.py` | **Wynik:** `strategy_results_v2.json`
* **Problem:** Wcześniejsza wersja obliczała progi kwantyli (20% i 80%) na podstawie całej historii, co prowadziło do *look-ahead bias* (model "znał" przyszłość).
* **Rozwiązanie:** 
  - Wdrożono rygorystyczne **rozszerzające się okno (expanding-window)**, gdzie progi dla dnia $t$ są liczone wyłącznie na danych od początku do $t-1$.
  - Dodano wariant z **przesuwnym oknem 4-letnim (rolling 1460 days)** dostosowującym się do zmieniającej się dynamiki cykli halvingowych.
  - Zaimplementowano sztywne progi literaturowe dla porównania (np. wejście: MVRV < 0, wyjście: MVRV > 5).
  - Obliczono metryki ekonomiczne (CAGR, Sharpe, Max Drawdown) z uwzględnieniem kosztów transakcyjnych.

## 2. Paper 10: Early Warning Indicators (EWI) dla Wysokiej Zmienności
**Plik:** `run_paper10_ewi_v2.py` | **Wynik:** `paper10_ewi_v2.json`
* **Problem:** Oryginalny model regresji logistycznej osiągał 90% Accuracy, ale miał 0% Precision i Recall dla rzadkiej klasy "wysokiej zmienności" (niezbalansowane dane).
* **Rozwiązanie:**
  - Dynamiczny target: zmienność $rv\_30d > 95\%$ kwantyl liczony w *expanding-window*.
  - Zastosowano **SMOTE (Synthetic Minority Over-sampling Technique)** oraz wagi klas (`class_weight='balanced'`) w celu poprawnej nauki klasy mniejszościowej.
  - Zmieniono metryki ewaluacji na odpowiednie dla niezbalansowanych danych: **PR-AUC (Precision-Recall Area Under Curve)**, ROC-AUC, Brier Score oraz F1-Score.

## 3. Paper 3: Kalkan & Tatlı (2022) – Test Kointegracji ARDL
**Plik:** `run_paper3_ardl_v2.py` | **Wynik:** `paper3_ardl_v2.json`
* **Problem:** Pierwotny kod używał uproszczonej regresji OLS zamiast formalnego modelu ARDL i testu kointegracji.
* **Rozwiązanie:**
  - Pełna procedura ekonometryczna na danych tygodniowych: testy stacjonarności **ADF i KPSS** (potwierdzenie, że zmienne to mieszanka I(0) i I(1)).
  - Optymalny dobór opóźnień (Lags) z użyciem kryterium informacyjnego AIC dla modelu ARDL.
  - Wykonanie **ARDL Bounds Test** (F-test i t-test) na obecność długoterminowej relacji kointegrującej między ceną, PM_proxy, AdrActCnt i SOPR.
  - Estymacja modelu korekty błędem (ECM) w celu zbadania szybkości powrotu do równowagi długookresowej.

## 4. Paper 6: Rafi et al. (2024) – Prognozowanie Zmienności (Deep Learning)
**Plik:** `run_paper6_dl_v2.py` | **Wynik:** `paper6_dl_v2.json`
* **Problem:** Zbyt małe modele trenowane tylko przez 5 epok, co nie pozwalało na miarodajne porównanie Transformer vs LSTM.
* **Rozwiązanie:**
  - Budowa architektur w PyTorch: **Transformer Encoder** (z Positional Encoding) oraz **LSTM** na zannualizowanej zmienności $rv\_7d$.
  - Porównanie różnych długości sekwencji wejściowych (14, 30, 60 dni).
  - Testowanie przeciwko mocnym benchmarkom: model **HAR-RV** oraz **GARCH(1,1)**.
  - Wdrożenie rygorystycznego **testu Diebolda-Mariano (DM)** do statystycznej weryfikacji, czy głębokie sieci neuronowe faktycznie prognozują lepiej niż klasyczna ekonometria.

## 5. Paper 7: Omole & Enke (2025) – Klasyfikacja Kierunku Ceny z Boruta
**Plik:** `run_paper7_ml_v2.py` | **Wynik:** `paper7_classification_v2.json`
* **Problem:** Poprzednie modele miały skuteczność rzędu rzutu monetą (~50%).
* **Rozwiązanie:**
  - Stworzenie bogatego zestawu 72 cech (*Feature Engineering*): wskaźniki on-chain, AT (RSI, MACD, Bollinger), statystyki kroczące oraz opóźnienia (lags 1, 3, 7).
  - Wdrożenie algorytmu **Boruta** (wspieranego przez Random Forest Importance) do zaawansowanej selekcji najważniejszych cech.
  - Trening i ewaluacja (RandomForest, XGBoost, SVC, MLP, GradientBoosting) za pomocą *Purged Time-Series Split* (horyzonty predykcji $h \in \{1, 3, 7\}$ dni).
  - Analiza wyników strategii inwestycyjnej (Long/Flat) z uwzględnieniem kosztów transakcyjnych 10 bps.

## 6. Paper 1: Wüstenfeld (2023) – Zmienność GARCH i Aktywność Sieci
**Plik:** `run_paper1_garch_v2.py` | **Wynik:** `paper1_garch_v2.json`
* **Problem:** Brak robust standard errors, testów diagnostycznych i testowania w podokresach.
* **Rozwiązanie:**
  - Porównanie modeli **GARCH(1,1)** dla rozkładu Normalnego oraz **t-Studenta** (który wykazał wielokrotnie lepsze dopasowanie AIC).
  - Pełna diagnostyka wystandaryzowanych reszt (testy Ljung-Box i ARCH-LM).
  - Regresja wariancji warunkowej na aktywność sieciową przy użyciu odpornych błędów standardowych **HAC (Newey-West)**.
  - Korekta wartości p na wielokrotne testowanie (**Bonferroni** oraz **FDR Benjamini-Hochberg**).
  - Estymacja GARCH-X oraz testowanie stabilności znaków współczynników w trzech podokresach (2013-17, 2018-20, 2021-26).

---

## Podsumowanie Działań Integracyjnych
- **Zależności:** Skrypty wykorzystują biblioteki `statsmodels`, `arch`, `boruta`, `xgboost`, `pytorch`, `scikit-learn`. Poprawiono kompatybilność pakietu `boruta` z najnowszym NumPy (`np.int`).
- **`run_all.sh`:** Zaktualizowano sekcje ekonometryczne, symulacji oraz uczenia maszynowego w głównym pliku uruchomieniowym, zapewniając płynne przejście całego pipeline'u analitycznego od pobierania metryk po generację wszystkich modeli i rezultatów.
