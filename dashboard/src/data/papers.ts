export interface Paper {
  id: string;
  title: string;
  url: string;
  summary: string;
  reproductionSteps: string;
  techStack: string[];
  codeAvailability: string;
  implementationPlan: string;
}

export const papers: Paper[] = [
  {
    id: "bitcoin-network-activity-volatility",
    title: "Bitcoin Network Activity and Bitcoin Price Return Volatility",
    url: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619010",
    summary: "Aktywność sieciowa (TxCnt, AdrActCnt, opłaty) jest silnie dodatnio powiązana z dzienną zmiennością zwrotów Bitcoina. Artykuł dowodzi, że wskaźniki on-chain mogą być traktowane jako proxy dla przepływu informacji rynkowych.",
    reproductionSteps: "Zbudowanie modelu GARCH(1,1) na zwrotach; regresja warunkowej wariancji na zlogarytmowane metryki aktywności on-chain.",
    techStack: ["arch", "statsmodels", "GARCH(1,1)"],
    codeAvailability: "Brak oficjalnego publicznego repozytorium na GitHubie. Wyniki należy odtworzyć lokalnie używając biblioteki 'arch' w Pythonie.",
    implementationPlan: `### Plan Implementacji: GARCH(1,1)
1. **Ekstrakcja danych**: Załadowanie szeregów czasowych zwrotów \`PriceUSD\` (log-returns) oraz wskaźników (np. \`TxCnt\`, \`CDD\`) z tabeli \`coinmetrics_daily\` poprzez BQ Client. Zlogarytmowanie wskaźników aktywności ($ln(x_t)$).
2. **Model GARCH**: Użycie biblioteki \`arch\` (arch_model) do dopasowania modelu ARMA(1,1)-GARCH(1,1). Obliczenie szeregu warunkowej wariancji ($h_t$). GPU nie jest wymagane do modeli GARCH.
3. **Regresja Ekonometryczna**: Użycie \`statsmodels.api.OLS\` by zbadać wpływ poszczególnych on-chain metrics na wariancję $h_t$. Kontrola multikolinearności (VIF).
4. **Weryfikacja**: Sprawdzenie P-value współczynników przy zmiennych on-chain. Jeśli $\\beta > 0$ i $p < 0.05$, uzyskujemy **100% potwierdzenie odtworzenia tezy** z artykułu.`
  },
  {
    id: "return-volatility-forecasting-flows",
    title: "Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets",
    url: "https://ideas.repec.org/p/arx/papers/2411.06327.html",
    summary: "Wpływy netto na giełdy (NetFlowUSD) wyprzedzają krótkoterminowe spadki cen i wzrost zmienności. Praca analizuje wolumen napływający i wypływający z giełd (Exchange Flows).",
    reproductionSteps: "Obliczenie różnic między wpływami a wypływami z giełd; dopasowanie modeli OLS ze zmiennymi opóźnionymi (lagged variables).",
    techStack: ["statsmodels.api.OLS", "Pandas Lags"],
    codeAvailability: "Autorzy nie udostępnili otwartego kodu do prognozowania. Można odtworzyć modele regresyjne OLS za pomocą biblioteki statsmodels.",
    implementationPlan: `### Plan Implementacji: Vector Autoregression (VAR) & Lagged OLS
1. **Przygotowanie cech**: Wyliczenie dziennego NetFlow (\`FlowInExUSD\` - \`FlowOutExUSD\`). Usunięcie obserwacji odstających (winsoryzacja na poziomie 1% i 99%).
2. **Lagged Features**: Wygenerowanie zmiennych opóźnionych: $NetFlow_{t-1}, NetFlow_{t-2}$ oraz opóźnionej zmienności (kroczące 7-dniowe odchylenie standardowe z \`PriceUSD\`).
3. **Modelowanie OLS**: Dopasowanie serii modeli OLS przewidujących przyszłą zmienność ($\sigma_{t+1}$) na bazie opóźnionych przepływów giełdowych. Modelowanie na CPU.
4. **Weryfikacja**: Odtworzenie tezy polega na zweryfikowaniu ujemnego współczynnika dla napływów giełdowych (NetFlow) w regresji zwrotu oraz dodatniego współczynnika dla zmienności. Istotność potwierdzana za pomocą statystyki t-Studenta testowanej w trybie out-of-sample.`
  },
  {
    id: "blockchain-analiz-gostergelerinin",
    title: "Blockchain Analiz Göstergelerinin Bitcoin Fiyatı Üzerindeki Etkisi",
    url: "https://dergipark.org.tr/tr/pub/esad/issue/74371/1181992",
    summary: "Wskaźnik SOPR, zyskowność górników (PM) i adresy aktywne wpływają krótko- i długoterminowo na cenę w oparciu o model ARDL. Analiza sprawdza stacjonarność i kointegrację zmiennych.",
    reproductionSteps: "Testy stacjonarności (ADF); budowa modelu autoregresyjnego ARDL i testy przyczynowości Toda-Yamamoto.",
    techStack: ["statsmodels.tsa.stattools", "ADF", "ARDL"],
    codeAvailability: "Brak kodu. Zastosowano standardowe ekonometryczne ramy ARDL, które bez problemu można odtworzyć w statsmodels w Pythonie lub R.",
    implementationPlan: `### Plan Implementacji: Modele ARDL i Kointegracja
1. **Pre-processing**: Pobranie \`SOPR\`, \`PuellMultiple\` i \`AdrActCnt\`. Zlogarytmowanie wyżej wymienionych zmiennych.
2. **Testy Stacjonarności**: Zastosowanie testu Augmented Dickey-Fuller (ADF) oraz Phillips-Perron (PP) w \`statsmodels\` by udowodnić integrację I(1) dla ceny i kombinację I(0)/I(1) dla zmiennych niezależnych.
3. **Model Bounds Testing**: Budowa modelu ARDL (Autoregressive Distributed Lag) wykorzystując optymalizację opóźnień wg kryterium AIC (Akaike). Przeprowadzenie Bounds Test (Pesaran) na obecność kointegracji.
4. **Weryfikacja**: Zakończona pełnym sukcesem tylko gdy wykażemy długoterminową zależność (zależność od SOPR i AdrActCnt). Model VECM zdiagnozuje prędkość powrotu do równowagi (Error Correction Term) potwierdzając tezy z badań tureckich autorów.`
  },
  {
    id: "using-on-chain-data-predict-cycles",
    title: "Using On-Chain Data to Predict Cryptocurrency Cycles",
    url: "https://osuva.uwasa.fi/server/api/core/bitstreams/cb668ce8-05ec-4be2-9062-d503386079f3/content",
    summary: "Ekstremalne odczyty NUPL i MVRV Z-score pozwalają stworzyć kontrariańskie strategie inwestycyjne lepsze od 'buy-and-hold'. Zjawiska euforii rynkowej da się mitygować wskaźnikami wartości uśrednionej i zrealizowanej.",
    reproductionSteps: "Ustalenie progów kwantylowych paniki/euforii (percentyle); symulacja backtestów wejść/wyjść na danych historycznych za pomocą metod Monte Carlo.",
    techStack: ["vectorbt", "pandas", "Monte Carlo"],
    codeAvailability: "Kod symulacyjny nie został udostępniony, ale środowisko wejścia/wyjścia można bezproblemowo napisać w vectorbt.",
    implementationPlan: `### Plan Implementacji: Symulator VectorBT (Backtesting)
1. **Kalibracja kwantyli**: Obliczenie na bazie rolowanych okien 4-letnich z MVRV_Z oraz NUPL statystycznych granic euforii (odchylenie > +2 Sigma) oraz kapitulacji (< -1 Sigma).
2. **Logika stanowa**: Zdefiniowanie funkcji handlowej z portfelem symulacyjnym. Wejście w Long gdy \`MVRV_Z < -1\` ORAZ wskaźnik makroekonomiczny poprawia się. Wyjście (Exit) przy przecięciu progu powrotu.
3. **Wymagania obliczeniowe**: Środowisko CPU/RAM. \`vectorbt\` potrafi skompilować logikę pandas do numby, wykonując tysiące przeliczeń rynkowych iteracji na sekundę.
4. **Weryfikacja**: Osiągniemy 100% pewność replikacji tez udowadniając w dashboardzie, że wskaźnik Sharpe Ratio stworzonej strategii pokonuje model *Buy and Hold* w ujęciu zaryzykowanego kapitału (risk-adjusted return) oraz znacząco redukuje Maximum Drawdown.`
  },
  {
    id: "on-chain-analysis-forecasting",
    title: "On-chain analysis and price forecasting using on-chain metrics",
    url: "https://dr.ntu.edu.sg/entities/publication/f393fcc8-1fdf-4fd6-8e4b-272a77097039",
    summary: "Dodanie wskaźników on-chain jako zewnętrznych regresorów do klasycznych modeli predykcyjnych (GARCH, Prophet, LSTM) poprawia trafność prognoz krótkoterminowych.",
    reproductionSteps: "Budowa klasycznych modeli szeregów czasowych (TS) z zewnętrznymi zmiennymi. Porównanie błędów RMSE/MAE z modelami bez danych on-chain.",
    techStack: ["Prophet", "PyTorch/Keras", "LSTM"],
    codeAvailability: "Zazwyczaj publikacje NTU zawierają linki do GitHub w samej finalnej wersji PDF, jednak wdrożenie biblioteki Prophet z 'add_regressor' jest trywialne do odtworzenia.",
    implementationPlan: `### Plan Implementacji: LSTM + Facebook Prophet
1. **Architektura Deep Learning**: Uruchomienie maszyny n1-standard z **akceleratorem T4/L4 GPU** na GCP.
2. **Budowa Tensora**: Transformacja szeregów czasowych do okien przesuwanych (Sliding Windows) o długości 30 dni w kształt \`[batch_size, timesteps, features]\`. Features: PriceUSD, IssTotNtv, FeeMedUSD. Skalowanie Min-Max.
3. **Trening**: Kompilacja wielowarstwowej sieci LSTM w \`PyTorch\`. Funkcja straty: MSE (Mean Squared Error). Trening na 100 epokach ze stymulacją optymalizatorem Adam.
4. **Weryfikacja**: Wytrenowanie dwóch wariantów (Model z danymi on-chain vs Model bez nich - 'Cena Only'). Weryfikacja 100% jeśli metryki testowe out-of-sample (MAE oraz RMSE) dla modelu wykorzystującego on-chain data są statystycznie istotnie mniejsze.`
  },
  {
    id: "crypto-volatility-forecasting-transformers",
    title: "Cryptocurrency Volatility Forecasting Using Transformer Models",
    url: "https://www.al-kindipublisher.com/index.php/jefas/article/view/10158",
    summary: "Nowoczesne modele oparte na mechanizmie atencji (Transformers) deklasują sieci LSTM w modelowaniu zmienności w oparciu o dane on-chain w warunkach stresu rynkowego.",
    reproductionSteps: "Implementacja sieci typu LSTM i Transformer Encoder prognozujących 7-dniową zmienność, ekstrakcja 'Attention Weights' by wskazać wiodące wskaźniki.",
    techStack: ["PyTorch", "Transformer Encoder", "Attention"],
    codeAvailability: "Brak udostępnionego repozytorium GitHub dla tego artykułu. Wymaga zbudowania własnego enkodera w PyTorch.",
    implementationPlan: `### Plan Implementacji: Time-Series Transformer Encoder
1. **Infrastruktura**: Model *musi* być trenowany na instancji GPU (rekomendowane karty V100/A100) ze względu na kwadratową złożoność mechanizmu Self-Attention po stronie czasowej.
2. **Embedding**: Projekcja linearna 56 naszych zmiennych z \`coinmetrics_daily\` do wymiaru d_model (np. 128) z dodaniem Positional Encoding (aby sieć rozumiała strukturę sekwencyjną).
3. **MHA (Multi-Head Attention)**: Trening modelu, w którym celem jest przewidzenie zlogarytmowanej wariancji (RV) na 7 dni do przodu.
4. **Weryfikacja i Explainability**: Aby uzyskać 100% potwierdzenie tezy, należy uruchomić kod na GPU, zeskrapować wagi z warstw *Attention* dla ostatniej epoki i udowodnić (wizualnie i numerycznie), że w momentach krachów to dane on-chain (np. Exchange Flows), a nie techniczne wskaźniki cenowe, otrzymują najwyższą uwagę sieci.`
  },
  {
    id: "predicting-price-direction-magnitude",
    title: "Predicting Bitcoin Price Direction and Magnitude (ML/DL + TA)",
    url: "https://scholarsmine.mst.edu/engman_syseng_facwork/1602/",
    summary: "Fuzja wskaźników analizy technicznej (TA) z danymi on-chain optymalizuje kategoryzację kierunku ceny (wzrost/spadek).",
    reproductionSteps: "Zdefiniowanie klasyfikacji cenowej. Wygenerowanie cech TA z biblioteki. Implementacja modelu Random Forest oraz wyciągnięcie ważności cech (SHAP/Boruta).",
    techStack: ["scikit-learn", "Random Forest", "SHAP/Boruta"],
    codeAvailability: "Brak opublikowanego kodu, lecz implementacja opiera się o gotowe funkcje RandomClassifier ze scikit-learn i paczkę 'ta'.",
    implementationPlan: `### Plan Implementacji: Ensemble Classification
1. **Generowanie etykiet (Labeling)**: Algorytmiczne oznaczanie dni z krokiem czasowym $t+1$ jako Klasa 1 (Wzrost > 0) lub Klasa 0 (Spadek). Dodanie oscylatorów technicznych (RSI, MACD, Bollinger Bands) z biblioteki \`ta\`.
2. **Trening Random Forest**: Uruchomienie na silniku CPU w Scikit-Learn \`RandomForestClassifier(n_estimators=500)\` przy użyciu weryfikacji krzyżowej dla szeregów czasowych (\`TimeSeriesSplit\`). Zabezpieczenie przez Data Leakage.
3. **Inerpretacja (SHAP)**: Przepuszczenie zoptymalizowanego lasu losowego przez algorytmy TreeExplainer z biblioteki \`shap\`.
4. **Weryfikacja**: Osiągnięcie trafności (Accuracy) przewyższającej rzut monetą (np. > 54% out-of-sample). Teza artykułu o przewadze fuzji TA+On-chain zostanie autorytatywnie udowodniona, jeżeli na wykresie *SHAP Summary Plot* górne 5 najistotniejszych metryk to optymalny mix wskaźników technicznych oraz z blockchaina (np. RSI tuż obok NVT).`
  },
  {
    id: "deciphering-bitcoin-cohort-analysis",
    title: "Deciphering Bitcoin Blockchain Data by Cohort Analysis",
    url: "https://www.nature.com/articles/s41597-022-01254-0",
    summary: "Ekstrema zjawiska niszczenia starych monet (Coin Days Destroyed - CDD) wyprzedzają kluczowe pęknięcia cykli (wierzchołki na rynku).",
    reproductionSteps: "Identyfikacja punktów powyżej 95-percentyla CDD i nałożenie na historyczną zrealizowaną zmienność rynkową. Stworzenie klasyfikacji okresów dystrybucji.",
    techStack: ["pandas", "Markov Switching Models"],
    codeAvailability: "Jako publikacja 'Scientific Data' (Nature), badacze udostępnili surowe zbiory danych na otwartych repozytoriach naukowych, ale analityka w Pythonie/Pandas jest tu kluczem.",
    implementationPlan: `### Plan Implementacji: Cohort & Markov State Analysis
1. **Ekstrakcja kohort**: Pobranie parametru CDD (\`TxTfrValDayDst\`) jako proxy dla starych inwestorów i LTH_MVRV. Zgrupowanie danych z ostatnich cykli (Halving to Halving).
2. **Modele Markova**: Uruchomienie modelu Markowa \`statsmodels.tsa.regime_switching\` (2-stany), gdzie proces zakłada ukrytą zmienną decyzyjną (np. State 0 = Akumulacja, State 1 = Dystrybucja).
3. **Analiza Dystrybucji**: Mapowanie zidentyfikowanych skoków w zniszczonych dniach (CDD) z pęknięciami bańki.
4. **Weryfikacja**: Mamy pełny sukces, gdy wyznaczone algorytmicznie prawdopodobieństwo "Reżimu 1" (Dystrybucji) osiąga okolice 1.0 (bliskość pęknięcia) od 2 do 3 tygodni przed faktycznym załamaniem wykresu \`PriceUSD\`, dokładnie tak, jak zaobserwował to zespół na badanych danych referencyjnych.`
  },
  {
    id: "blockchain-analysis-bitcoin-market",
    title: "Blockchain Analysis of the Bitcoin Market",
    url: "https://www.nber.org/system/files/working_papers/w29396/w29396.pdf",
    summary: "Duża część wolumenu on-chain to tzw. sztuczny szum transakcji powiązanych z przepływami pomiędzy giełdami. Analiza koncentracji podmiotów.",
    reproductionSteps: "Obliczenie miary koncentracji wolumenu giełd (Exchange In/Out) do całego wolumenu. Testy kroczącej korelacji z rynkami zewnętrznymi.",
    techStack: ["statsmodels", "Rolling Correlation"],
    codeAvailability: "NBER Working Papers zazwyczaj nie mają udostępnionych skryptów wykonawczych w otwartym dostępie, aczkolwiek metryki giełdowe posiadamy.",
    implementationPlan: `### Plan Implementacji: Network & Noise Filtration
1. **Redukcja szumu giełd**: Odfiltrowanie "sztucznej objętości": od całkowitego \`TxTfrValUSD\` odejmowana jest suma przepływów wewnętrznych \`ExchangeFlowIn\` oraz \`ExchangeFlowOut\` dla zidentyfikowania *Real Economic Volume*.
2. **Rozkład Poissona/Pareto**: Testowanie rozkładu koncentracji - czy ułamek topowych adresów (\`AdrBalCnt\`) posiada dysproporcjonalny wpływ na rynek (test zgodności z rozkładem Zipfa i Pareto we \`scipy.stats\`).
3. **Weryfikacja**: NBER postuluje drastyczną koncentrację w rękach nielicznych podmiotów giełdowych. Odtworzymy badanie tworząc rolling z-score z ratio (Exchange Volume / Total Network Volume). Zreprodukujemy ich rygor naukowy jeśli ratio w okresach panic-selling udowodni statystyczną kointegrację (Cointegration Test) ze spadkami giełd amerykańskich (SPX/NDX).`
  },
  {
    id: "inferring-short-term-volatility",
    title: "Inferring Short-Term Volatility Indicators from Blockchain",
    url: "https://arxiv.org/abs/1809.07856",
    summary: "Metody redukcji wymiarów (PCA, NMF) na wielkim zgranym zbiorze cech grafowych z blockchaina pozwalają wyodrębnić sygnał wczesnego ostrzegania (EWI) przed zmiennością.",
    reproductionSteps: "Agregacja dużej ilości cech. Przeprowadzenie analizy głównych składowych (PCA). Użycie regresji logistycznej z progiem decyzyjnym (RV > 95%).",
    techStack: ["sklearn.decomposition.PCA", "Logistic Regression"],
    codeAvailability: "Wyniki eksperymentalne z 2018 roku na arXiv - brak śladów po otwartoźródłowym GitHubie. Główna zasada to redukcja kilkudziesięciu metryk (które mamy) do 3 wymiarów w PCA.",
    implementationPlan: `### Plan Implementacji: ML-based Early Warning System (PCA)
1. **High-Dimensional Preprocessing**: Normalizacja (Standaryzacja Z-Score) macierzy 50 zmiennych on-chain poprzez obiekt \`StandardScaler\` z scikit-learn. Usuwanie autokorelacji.
2. **PCA (Principal Component Analysis)**: Dekompozycja macierzy 50 wymiarów do przestrzeni ortogonalnej o rozmiarze 3. Wektor własny o najwyższej wartości (\`PC1\`) posłuży jako syntetyczny oscylator gęstości rynkowej. Obliczenia macierzowe wyłącznie na szybkich CPU, nie jest potrzebny GPU.
3. **Klasyfikator Wczesnego Ostrzegania**: Trening klasyfikatora Logistycznej Regresji z karą regularyzacji L2, próbującego przewidzieć dni, kiedy zrealizowana zmienność przekracza roczny 95-percentyl. Zastosowanie obróbki klas niezbilansowanych (SMOTE).
4. **Weryfikacja**: 100% odtworzenie wyników następuje przez obliczenie pola pod krzywą ROC (ROC-AUC). Jeśli AUC naszego PCA-EWI Classifiera przekroczy wartość 0.70 na zbiorze walidacyjnym (Test Set z krachu COVID-2020/FTX-2022), uznajemy tezę za prawomocną, gdyż sygnał PC1 w pełni izoluje wskaźniki predykcyjne szumu.`
  }
];
