# Raport z Audytu Metodologicznego V2

Poniższy raport przedstawia weryfikację skryptów i wyników w wersji V2 (Zero Look-Ahead Bias, ulepszona diagnostyka i ewaluacja OOS).

## 1. Analiza plików Paperów

### Paper 4 (run_strategies_v2.py + strategy_results_v2.json)
1. **Zero look-ahead bias?** TAK. Kwantyle wyliczane są ściśle na oknie historii do wczoraj (`[:t]` lub `[t-window:t]`), a pozycje przesuwane o 1 dzień.
2. **Time-series split/expanding?** TAK. Zastosowano pętlę chronologiczną dla sygnałów (expanding i rolling) bez losowego tasowania (shuffle).
3. **Koszty transakcyjne?** TAK. Koszty na poziomie 20 bps dla każdej zmiany pozycji (obliczane na wektorze `np.abs(np.diff(sub_pos))`).
4. **Adkeatna metryka?** TAK. Wykorzystano sprawdzone metryki ryzyka (CAGR, Sharpe, Max Drawdown, Calmar).
5. **JSON czytelny?** TAK. Wyniki posiadają strukturę słownikową zawierającą krzywą kapitału oraz tabele metryk z podziałem na koszty i ich brak.
6. **Teza:** POTWIERDZONA. Podejście expanding window (OOS) skutecznie redukuje Max Drawdown względem metody Buy & Hold, podnosząc współczynnik Sharpe'a.

### Paper 10 (run_paper10_ewi_v2.py + paper10_ewi_v2.json)
1. **Zero look-ahead bias?** TAK. Standaryzacja, PCA, estymacja wag klas i kalibracja progu wyznaczane są na oknach uczących; test odbywa się na "przyszłym" oknie 30-dniowym. Target (Q95) również jest "expanding".
2. **Time-series split/expanding?** TAK. Wprowadzono precyzyjną, krokową pętlę przesuwającą się co 30 dni w przyszłość bez podglądania testu.
3. **Koszty transakcyjne?** NIE DOTYCZY. Ewaluacja następuje zjawiskiem Event Study po zapaleniu flagi alertu (liczony jest czysty spread nad baseline).
4. **Adkeatna metryka?** TAK. Optymalizacja bazuje na PR-AUC, F1-score i recall, co perfekcyjnie pasuje do bardzo rzadkich zdarzeń (ok. 5% przypadków wysokiej zmienności).
5. **JSON czytelny?** TAK. Plik zapisuje całościowe time-series alertów oraz podsumowania Event Study i Alert Quality.
6. **Teza:** POTWIERDZONA. Event study 14-dniowe wskazuje dodatnie asymetryczne wartości oczekiwane względem testów Monte Carlo, co czyni alerty użytecznymi pomimo relatywnie niskiego Precision (cecha EWI).

### Paper 3 (run_paper3_ardl_v2.py + paper3_ardl_v2.json)
1. **Zero look-ahead bias?** TAK. Badana jest chronologiczna zależność opóźnień między zmiennymi w oknach stacjonarności.
2. **Time-series split/expanding?** TAK. Modele VECM / ARDL naturalnie traktują rzędy czasowo, a resample zrobiono na interwał tygodniowy w sposób "last/mean", nie przeciekając z przyszłości do przeszłości.
3. **Koszty transakcyjne?** NIE DOTYCZY. Paper testuje czystą kointegrację ekonometryczną.
4. **Adkeatna metryka?** TAK. Test Pesaran Bounds dla F-statystyki, dobór opóźnień wg kryterium AIC oraz rygorystyczne testy kointegracji (ADF/KPSS).
5. **JSON czytelny?** TAK. Prezentuje rozbicie na wartości stacjonarności, model ECM (Error Correction) i mnożniki długookresowe.
6. **Teza:** POTWIERDZONA. F-Stat znacząco przekracza I(1), a kointegracja Bitcoina (Y) z m.in. wskaźnikiem NUPL w długim terminie została statystycznie udowodniona z dużą ufnością.

### Paper 1 (run_paper1_garch_v2.py + paper1_garch_v2.json)
1. **Zero look-ahead bias?** TAK. Regresja opóźnień używa `shift(1)` oraz `shift(7)`, więc modeluje zmienność obecną tylko w oparciu o to co widział przed 1 lub 7 dniami.
2. **Time-series split/expanding?** TAK. Wprowadzono test podokresów chronologicznych (2013-2017, 2018-2020, 2021-2026), modele GARCH używają MLE na historii.
3. **Koszty transakcyjne?** NIE DOTYCZY. Prognozowana jest warunkowa wariancja, nie handel.
4. **Adkeatna metryka?** TAK. Zaimplementowano standard błędów HAC (Newey-West) oraz poprawkę Bonferroniego, co likwiduje fałszywą istotność spowodowaną autokorelacją i mnogimi testami.
5. **JSON czytelny?** TAK. Posiada struktury diagnozujące dla estymatora GARCH (Normal vs Student-t), a także p-value z poprawkami wielokrotnymi dla każdego regresora.
6. **Teza:** OBALONA. Po uwzględnieniu restrykcyjnych poprawek (Bonferroni, HAC) nałożonych na model, statystyczna istotność predykcyjna zmiennych sieciowych (Network Activity) dla dziennej zmienności drastycznie spada. Brak również spójności znaków współczynników w ujęciu długoterminowym.

### Paper 7 (run_paper7_ml_v2.py + paper7_classification_v2.json)
1. **Zero look-ahead bias?** TAK. `StandardScaler` jest dopasowywany (`fit`) tylko na danych uczących i aplikowany (`transform`) do testowych; to samo tyczy się Boruty.
2. **Time-series split/expanding?** TAK. Wykorzystano `TimeSeriesSplit(n_splits=5)`, co zapewnia uczciwe, poszerzane okno cross-walidacyjne bez tasowania wierszy (shuffle).
3. **Koszty transakcyjne?** TAK. W kalkulacji zysku dodano opłatę `10 bps` od każdej zmiany pozycji.
4. **Adkeatna metryka?** TAK. Dodano m.in. zbalansowane Accuracy, MCC oraz F1. Algorytmy jak Boruta używają stabilnych RandomForestów z obniżonym learning rate przy innych boosterach.
5. **JSON czytelny?** TAK. Zapisano wyniki dla każdego horyzontu (h=1,3,7), wybrane cechy przez Borutę i statystyki modeli.
6. **Teza:** NIEROZSTRZYGNIĘTA. Pomimo zaawansowanej selekcji Borutą, kierunkowa skuteczność sygnałów bliska jest \~50-53% Out-of-Sample, co sprawia że większość modeli ML nie osiąga po kosztach wyników radykalnie odbiegających od losowych wariantów.

### Paper 6 (run_paper6_dl_v2.py + paper6_dl_v2.json)
1. **Zero look-ahead bias?** TAK. Generowane sekwencje korzystają z przesuwającego się okna (np. $t-30$ do $t-1$) do przewidzenia $t$. Standaryzacja wejść uczy się ściśle przed podziałem walidacyjnym/testowym.
2. **Time-series split/expanding?** TAK. Próbka (train / val / test) podzielona jest blokowo, sekwencyjnie. Sieci uczą się bez mieszania bloków niszczącego czas z OOS.
3. **Koszty transakcyjne?** NIE DOTYCZY. Jest to paper stargetowany na czyste zadanie predykcji ciągłej wariancji.
4. **Adkeatna metryka?** TAK. Zastosowano Diebold-Mariano test sprawdzający na ile prognozy różnią się od siebie w sposób istotny statystycznie oraz wiodące RMSE/MAPE.
5. **JSON czytelny?** TAK. Posiada czytelne drzewo wyników dla poszczególnych architektur (LSTM, Transformer), długości sekwencji (14,30,60) oraz baseline'ów.
6. **Teza:** OBALONA / NIEROZSTRZYGNIĘTA. Z JSON wynika wprost, że na analizowanej próbie metody naiwne oraz zoptymalizowane HAR-RV biją modele głębokie (LSTM/Transformer) testami Diebold-Mariano (niższe RMSE). Przypomina to zjawisko ekstremalnego zaszumienia rynków krypto wykluczającego modele DL.

---

## 2. Trzy najważniejsze "Residual Risks" (Ryzyka resztkowe)

1. **Brak rygorystycznego modelowania slippage'u i głębokości rynku (Bid/Ask spread):** W symulacjach strategii zakłada się jednorodną płynność (flat 10-20 bps fee). Przy sygnałach w czasie flash-crashów, egzekucja po cenie zamknięcia mogłaby wiązać się z gigantycznym poślizgiem ukrytym na spreadach.
2. **Concept Drift / Brak stacjonarności w czasie:** Większość modeli (poza GARCH z próbkowaniem w sub-okresach) używa stałych hiperparametrów i stałych progów (np. stałe top 20% / bottom 20% lub progi EWI dopasowane na pierwszych 30% zbioru walidacyjnego). Zmiany takie jak halvingi, wdrożenia ETF czy wzrost udziału instytucji mogą sprawić, że progi z 2017 roku są bezużyteczne w roku 2025.
3. **Hyperparameter Overfitting:** (Pomimo Zero Look-Ahead): Wybór wielkości okna (np. $m=1460$ dni dla kwantyli rolling) czy progu $h=7$ dla EWI pochodzi historycznie od badaczy z podglądem na przeszłość całego rynku. Prawdziwy, kaskadowy "Nested Cross-Validation" wymagałby hiperparametryzacji co krok (np. każdego dnia model ustala na nowo optymalny próg okna), co tu na razie wykonano w podstawowej (choć solidnej) wersji rozszerzającej.

---

## 3. Rekomendacje dla Dashboardu (Top Wyniki)

Poniższe 3 wykresy/wyniki stanowią najmocniejsze dowody inżynieryjne projektu, idealne dla UI:

1. **Wykres "Event Study" EWI (Paper 10):**
   * **Dlaczego:** Użycie skalibrowanego modelu do wysyłania rzadkich alertów z *Alert Quality Score*. Wizualne "punkty ostrzegawcze" na osi ceny Bitcoina z towarzyszącym im wykresem słupkowym (14-dniowy asymetryczny zwrot vs losowy wybór Małpy) to wizytówka dla analityków. Pokazuje prawdziwe, rzadkie *alpha*.
2. **Tabela / Graf "Kointegracja & Long-Run Multipliers" (Paper 3):**
   * **Dlaczego:** Paper ten dowiódł, w sposób bardzo odporny statystycznie, relacji pomiędzy zmiennymi On-Chain (głównie NUPL) a ceną, wykorzystując Bounds Test (F-stat = 12.52). Pokazanie kointegracji obala mit błądzenia losowego krypto-aktywów – to dowód na wartość on-chainu o ogromnej wartości merytorycznej.
3. **Krzywa Kapitału "Expanding Quantiles" (Paper 4):**
   * **Dlaczego:** Klasyczny, bardzo pożądany wykres (Equity Curve). Ukazuje czarno na białym, jak prosta strategia zabezpieczona z wyciętym Look-Ahead Bias redukuje Max Drawdown (często o kilkadziesiąt procent względem klasycznego Buy & Hold), zachowując przy tym atrakcyjny CAGR. To najbardziej zrozumiała grafika dla zwykłego tradera.
