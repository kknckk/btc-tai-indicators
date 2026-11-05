# Recenzja raportu z realizacji zadania 1.4 — projekt FENG PKO BP–SGH

**Dokument oceniany:** „Raport z realizacji zadania 1.4. Potok modeli uczenia maszynowego i sztucznej inteligencji”, wersja 1.0, 30.10.2026, projekt FENG.01.01-IP.01-A059/23.  
**Perspektywa:** przydatność wyników dla współpracy z PKO BP oraz dla projektu BTC-TAI.  
**Podstawa odniesień:** numeracja rozdziałów i tabel dokumentu; rozdziały 7.4–7.6 nie były w całości dostępne w treści przekazanej do analizy.

## 1. Konkluzja dla zespołu

Raport dokumentuje **wykonanie szerokiego, odtwarzalnego etapu badawczo-integracyjnego**, a nie wybór ani wdrożenie modeli produkcyjnych. Oceniono **190 kombinacji** modelu, wariantu danych i horyzontu — dla obligacji również tenoru — z czego **188 dało pełny zestaw prognoz**. Dwa niepełne przypadki dotyczą modelu dekompozycyjnego **VMD + XGBoost dla ropy**. Osiem grup wskaźników to EUR/PLN, USD/PLN, stopa referencyjna NBP, obligacyjna krzywa DNS, WIG20, ropa Brent, złoto i BTC/USD (§1, §3, tab. 1, 23).

Najbardziej przekonujący **sygnał prognostyczny względem właściwych benchmarków** dotyczy modeli zmienności GARCH: według zestawienia raportu przewyższają benchmark w **19 z 20** konfiguracji, a najlepsze wyniki na długim horyzoncie są rzędu 14–15% poprawy MAE (§7.9, tab. 23). Dla poziomu cen przewaga jest znacznie rzadsza i mniejsza. W szczególności wynik **BTC: SVR +1,8%, TFT +0,7% dla 30 dni** jest interesującym punktem odniesienia, lecz **nie dowodzi jeszcze trwałej przewagi inwestycyjnej**: dane BTC kończą się 31.03.2025, a raport nie przedstawia testu istotności tych różnic (§4.1–4.2, §7.8, tab. 22).

> „Status zaliczenia oznacza [...] poprawne przejście testu integracyjnego w zakresie dostępnych komponentów infrastruktury, a nie akceptację modelu do zastosowania produkcyjnego” — sens zastrzeżenia z §1 i §6.6.

Dla naszej współpracy najważniejsze są zatem **wspólny protokół oceny, kontrakt danych point-in-time i kontrakt wdrożeniowy ONNX**, a nie przejmowanie rankingu modeli jako gotowej decyzji biznesowej.

## 2. Fakty: zakres, protokół i infrastruktura

### 2.1. Co rzeczywiście wykonano

| Obszar | Fakt z raportu | Znaczenie praktyczne |
|---|---|---|
| Zakres | Osiem grup wskaźników wymienionych wyżej; modele poziomu, zmienności, kierunku decyzji NBP oraz czynniki obligacyjne DNS (§3–4, tab. 1–2). | Nie należy porównywać jednym rankingiem modeli rozwiązujących różne zadania. |
| Skala | 190 ocenionych kombinacji; 188 kompletnych prognoz (§1, §7.9, tab. 23). | Wysoka kompletność potoku, ale nie „190 modeli produkcyjnych”. |
| Podział | Chronologicznie **70% trening / 15% walidacja / 15% test**, osobno na osi czasu każdego targetu (§4.2, tab. 3; §5.1). | Prawidłowy punkt wyjścia dla porównań poza próbą. |
| Backtest | Walk-forward z rozszerzanym oknem; model ponownie budowany na danych dostępnych do chwili prognozy (§5.1). | Trzeba odtworzyć również chwile prognozy, krok, kalendarz i dostępność cech, nie tylko sam podział procentowy. |
| Horyzonty | Dla FX, WIG20, ropy, złota i BTC zasadniczo **1 i 30**, lecz w odpowiednich jednostkach: dni notowań lub dni kalendarzowe BTC. **Wyjątki:** obligacje **60 i 250 dni notowań**; stopa miesięczna **1 i 12 miesięcy** (§5.1, tab. 4). | Sformułowanie „wszędzie horyzonty 1 i 30 dni” byłoby nieprawdziwe. |
| Benchmark poziomu | Ostatnia znana wartość: prognoza naiwna `ŷ(t+h)=y(t)` (§5.3). | Silny, konieczny punkt odniesienia dla cen. |
| Metryki | Dla poziomów głównie **MAPE**, pomocniczo MAE/RMSE i trafność kierunku; dla poziomu stopy NBP głównie **MAE**; dla zmienności **MAE** względem kwadratu dziennego log-zwrotu i benchmark pięciu ostatnich obserwacji; dla decyzji NBP Balanced Accuracy i Macro-F1 (§5.3). | Nie wolno interpretować „poprawy” bez wskazania zadania, miary i benchmarku. |
| Kalibracja | Niewielkie, ustalone siatki konfiguracji, wybór na walidacji, **trzy ziarna** modeli losowych; brak systematycznej optymalizacji i rozbudowanej selekcji predyktorów (§5.2, §5.4). | Wyniki są baseline’em etapu, nie końcowym oszacowaniem potencjału metod. |

Prognozy wielokrokowe są według §5.1 dynamiczne: nie korzystają z faktycznych przyszłych wartości targetu, a nieprognozowane predyktory pozostają w horyzoncie na stanie znanym w chwili `t`. To istotny szczegół przy odtwarzaniu eksperymentu BTC-TAI. Trzeba jednak oddzielnie sprawdzić, czy każdy konkretny model i eksport zachowują tę samą semantykę.

### 2.2. Co oznacza „przekazanie do PKO BP”

Infrastruktura banku to **PKO Feature Registry (`pko-fr`)**, korzystający z danych BigQuery w warstwach **Silver/Gold projektu `lab-biz-tai-gcp-pr`**. Bankowa biblioteka prowadzi pobranie i wyrównanie danych, walidację, walk-forward, metryki oraz serializację; dostawca modelu przekazuje klasę `Model` w `src/model.py`, `model_config.yaml`, powiązania z wersjami cech i targetu oraz wynik referencyjny (§6–6.1). Wyrównanie cech o różnej częstotliwości ma używać złączenia **as-of/point-in-time** (§6.1, §6.3).

Kontrakt dla **modeli poziomu** brzmi: **poziom na wejściu i wyjściu, log-zwrot obliczany wewnątrz modelu, bieżący poziom jako `anchor`**. Rekonstrukcja, np. `P̂(t+h)=P(t)·exp(r̂)`, musi być częścią grafu **ONNX**, a inferencja ma działać przez **onnxruntime** (§6.1–6.2). Wzorzec ten nie dotyczy bezpośrednio wyjść modeli zmienności, klasyfikacji decyzji ani wielowymiarowych czynników DNS (§6.5).

Przejście backendu GCP na **Vertex Pipelines** uniemożliwiało zakończenie pełnej publikacji artefaktów przez `model_publish`; testowano dostępny zakres do treningu i walidacji (§6.3, §6.6, tab. 6). Ponadto raport jawnie wskazuje, że natywne artefakty **GARCH** oraz niewyeksportowane równania **ARIMA/VAR/BVAR** nie stanowią jeszcze ścieżki produkcyjnej zgodnej z wymogiem ONNX (§6.5). Zaliczenie tab. 6 jest więc **wynikiem testu działania/integracji**, nie certyfikatem gotowości operacyjnej czy akceptacją ryzyka modelowego.

## 3. Wyniki według wskaźnika: gdzie jest przewaga nad benchmarkiem

Liczby poniżej oznaczają **względną poprawę błędu**, chyba że podano Balanced Accuracy. Dodatnia wartość sama w sobie **nie oznacza istotności statystycznej**.

| Wskaźnik / zadanie | Wynik dodatni | Wynik negatywny lub ograniczenie | Źródło |
|---|---|---|---|
| **EUR/PLN — poziom** | VAR i BVAR: po **+6,1% MAPE** dla 30 dni notowań. Przy 1 dniu: +0,7% i +0,6%. | Pozostałe metody przy 1 dniu są zasadniczo bliskie naiwnej lub gorsze; LSTM z predyktorami dla 30 dni: **−378,1%**, z dużą niestabilnością między ziarnami. | §7.1, tab. 8 |
| **USD/PLN — poziom** | Dla 30 dni: VAR **+5,6%**, BVAR **+5,5%**, TFT tylko target **+5,1%**, XGBoost **+4,5%** bez predyktorów / **+3,2%** z nimi. | Przy 1 dniu przewagi są niewielkie; LSTM z predyktorami dla 30 dni **−38,8%**. | §7.2, tab. 9 |
| **Stopa NBP — poziom** | **Brak przewagi** modeli nad naiwną w tabelach dla serii dziennej i miesięcznej. | Dla dziennego h=1 błąd naiwnej wynosi **0,000 pkt proc.**; ARIMA dla h=30 jest gorsza o 46,6%. Dla miesięcznego h=12 najlepszy z podanych VAR nadal jest gorszy o 13,6%. | §7.3, tab. 10 |
| **Stopa NBP — kierunek decyzji** | Ordered logit i ordered probit z predyktorami: **BA 96,9%** wobec 50,0% benchmarku. | Test to tylko **17 momentów, w tym jedna zmiana stopy**; predyktory panelu miesięcznego nie uwzględniają opóźnienia publikacji. Wynik **nie jest wiarygodnym dowodem jakości przyszłych decyzji**. | §4.2, tab. 2–3; §7.3, tab. 11 |
| **Obligacje — DNS** | Według zestawienia: najsilniejsza poprawa **β1 (nachylenie) +36,3%**, pełna krzywa **+9,7%**, ceny obligacji **+6,7%**; horyzont istotnego wyniku β1: **250 dni notowań**. | Targetem są zerokuponowe rentowności **odtworzone z DNS**, nie bezpośrednio notowania benchmarkowych obligacji. Szczegółowych testów i tenorów nie dało się niezależnie sprawdzić w uciętym fragmencie §7.4. | §1–2, §4.2 tab. 2, §5.1 tab. 4, §7.9 tab. 23 |
| **WIG20 — poziom** | W zestawieniu tylko jedna konfiguracja lepsza: ARIMA h=1 **+0,8%**. | Nie widać mocnej przewagi w prognozie poziomu. | §7.9, tab. 23; szczegóły §7.5 do weryfikacji |
| **WIG20 — zmienność** | Wszystkie sześć konfiguracji lepszych; najlepsza EGARCH h=30 **+14,6%**. | To inny target i benchmark niż dla poziomu indeksu. | §7.9, tab. 23 |
| **Ropa Brent — poziom** | Zestawienie podaje ARIMA h=30 **+0,1%** — różnica praktycznie zerowa. | **16/18** kompletnych konfiguracji poziomu; VMD + XGBoost wymaga dopracowania. Raport wskazuje kumulację błędu LSTM w 30-krokowych ścieżkach, szczególnie w marcu/kwietniu 2026. | §1, §7.9 tab. 23; szczegóły §7.6 do weryfikacji |
| **Ropa Brent — zmienność** | Siedem z ośmiu konfiguracji lepszych; EGARCH h=30 **+15,2%**. | Nie należy przenosić tego wyniku na prognozę poziomu ceny. | §7.9, tab. 23 |
| **Złoto — poziom** | LSTM tylko target h=30 **+3,8%**; h=1 **+0,1%**. | XGBoost h=30: około **−16%**; drzewa modelujące poziom nie ekstrapolują dobrze przy nowych rekordach cen. | §7.7, tab. 20, rys. 12 |
| **Złoto — zmienność** | Wszystkie sześć konfiguracji lepszych; EGARCH **+5,2%** dla h=1 i **+13,8%** dla h=30. | Zmienność jest mierzona przybliżeniem opartym na kwadracie pojedynczego zwrotu, nie bezpośrednio obserwowaną wariancją warunkową. | §5.3, §7.7, tab. 21 |
| **BTC/USD — poziom** | Tylko target, h=30: **SVR +1,8%**, **TFT +0,7%**. | **Wszystkie** badane konfiguracje h=1 są słabsze od naiwnej. Dla h=30 XGBoost, PatchTST i LSTM z predyktorami są wyraźnie gorsze; ten ostatni **−62,9%**. | §7.8, tab. 22 |

**Istotna korekta interpretacyjna:** stwierdzenie „modele biją naiwną dla BTC” jest prawdziwe wyłącznie dla **dwóch wskazanych konfiguracji na 30 dni**, na historycznym teście kończącym się w marcu 2025. Nie dotyczy prognoz jednodniowych, całej rodziny transformerów ani dowolnego zestawu cech (§7.8, tab. 22).

## 4. Najważniejsze pułapki metodologiczne i danych

### 4.1. Nieporównywalna aktualność danych BTC

Eksport BTC datowany 13.09.2026 kończy się **31.03.2025**; zbiór testowy obejmuje **01.09.2023–31.03.2025** (§4.1–4.2, tab. 3). To nie jest test aktualnego zachowania modelu na dzień raportu, 30.10.2026. Przed decyzją o przydatności w PKO trzeba uzupełnić historię, sprawdzić spójność źródła i wykonać nowy, chronologiczny test po marcu 2025 **bez dobierania modeli na jego podstawie**.

### 4.2. Przeciek dostępności danych w panelu miesięcznym NBP

Raport zasadniczo opisuje opóźnienia publikacji i stan wiedzy na chwilę prognozy (§4.3, §5.1), ale **wprost wyłącza z tej zasady miesięczny panel stopy**: CPI i pozostałe predyktory wykorzystano bez właściwych opóźnień, zgodnie z implementacją referencyjną (§4.4, §5.1, §7.9). Złączenie as-of na samej dacie przypisanej obserwacji **nie naprawi** problemu, jeżeli wartość okresowa była faktycznie publikowana później. Wynik BA 96,9% należy traktować przede wszystkim jako test implementacji klasyfikatora, dopóki cechy nie zostaną odtworzone według rzeczywistych dat publikacji i, tam gdzie istotne, rewizji danych.

Przy 17 prognozach i jednej zmianie stopy wartość BA około 96,9% może wynikać z poprawnego rozpoznania tej jednej zmiany przy niemal bezbłędnej klasie dominującej; **pojedyncze zdarzenie nie wystarcza do oszacowania czułości na przyszłe zmiany reżimu**. W tab. 11 Macro-F1 wynosi 65,6%, co dodatkowo przemawia za analizą macierzy pomyłek, a nie reklamowaniem samego BA.

### 4.3. Predyktory nie zawsze dochodzą do końca testu

Serie rynkowe z USA i WTI kończą się w **maju 2026**. Raport podaje wykorzystanie ostatniej dostępnej wartości m.in. w **2 z 31 momentów FX** i **57 z 599 momentów WIG20** (§4.2, tab. 2; §4.4). To nie jest przeciek z przyszłości, ale jest **degradacją świeżości cechy**. W produkcji powinna mieć limit dopuszczalnej „starości”, alarm i zdefiniowane zachowanie modelu po przekroczeniu limitu.

### 4.4. Ograniczenia konstrukcji modeli wielokrokowych

- **Drzewa na poziomach:** przypadek złota pokazuje niezdolność XGBoost modelującego bezpośrednio cenę do sensownej ekstrapolacji ponad zakres treningowy (§7.7, rys. 12). Naturalnym kandydatem jest model zwrotu z rekonstrukcją poziomu względem kotwicy, ale nie gwarantuje to poprawy jakości.
- **Rekurencyjna kumulacja błędu:** raport sygnalizuje problem LSTM dla ropy przy h=30, zwłaszcza w marcu/kwietniu 2026; szczegółowe wartości należy sprawdzić w pełnym §7.6. Trzeba rozważyć predykcję bezpośrednio dla horyzontu, ograniczenia skrajnych zwrotów i testy stabilności całej ścieżki, nie tylko końcowego punktu.
- **Niepełny VMD + XGBoost:** brak dwóch pełnych prognoz uniemożliwia traktowanie tej metody jako zweryfikowanego odpowiednika kompletnych modeli (§1, §7.9, tab. 23).
- **Nachodzące cele h=30:** przy kroku jednej obserwacji dla BTC, WIG20 i surowców wiele kolejnych prognoz długiego horyzontu dotyczy nakładających się okresów (§5.1, tab. 4). Liczby błędów nie są wtedy niezależnymi replikacjami; przedziały ufności wymagają metod uwzględniających zależność czasową.

## 5. Krytyczna ocena raportu

### Mocne strony

1. **Uczciwie ograniczony zakres wniosków.** Autorzy rozróżniają lokalną kalibrację, techniczną integrację i brak akceptacji produkcyjnej (§1–2, §5.4, §6.6).
2. **Wspólny, chronologiczny protokół** oraz benchmark oceniany w tych samych momentach co modele (§5.1–5.3).
3. **Jawne reguły dostępności danych** i udokumentowany wyjątek dla CPI, zamiast przemilczenia ryzyka (§4.3–4.4).
4. **Praktyczny kontrakt wdrożeniowy**: wersjonowane cechy, manifest eksportu, porównanie wyniku lokalnego z bankowym, anchor i rekonstrukcja w ONNX (§6.1–6.5).
5. **Adekwatne rozdzielenie zadań:** poziom, zmienność, kierunek decyzji i obligacyjna rekonstrukcja krzywej nie zostały sztucznie sprowadzone do jednej miary (§5.3, §7.9).
6. **Rozbudowany strumień obligacyjny**, obejmujący budowę targetu DNS i wycenę, a nie wyłącznie regresję na gotowej serii (§1–2).

### Słabości i brakujące dowody

1. **Brak testów istotności większości przewag.** Z opisu wynika, że szerszą walidację statystyczną przeprowadzono dla obligacji (§7.4.7); dla niewielkich przewag BTC czy FX sam znak różnicy MAPE nie wystarcza. Potrzebne są parowane różnice strat w tych samych chwilach prognozy, przedziały ufności i testy odporne na autokorelację/nakładanie horyzontów.
2. **Wiele porównań, wstępny wybór konfiguracji.** Przy 190 kombinacjach część niewielkich dodatnich wyników może pojawić się przypadkiem. Należy raportować niepewność i uwzględniać wielokrotne testowanie, zwłaszcza gdy komunikowany jest „najlepszy model”.
3. **Ograniczona kalibracja:** małe siatki, trzy ziarna i brak systematycznej selekcji cech (§5.2). Trzy ziarna ujawniają część niestabilności, lecz nie są mocnym oszacowaniem ryzyka ogonowego treningu.
4. **Problem aktualności i żywotności cech**, szczególnie BTC i kończących się wcześniej szeregów USA (§4.2–4.4).
5. **Niedostateczne dowody przeniesienia wyników lokalnych na artefakt docelowy.** Tab. 5 opisuje niekiedy inne cechy niż lokalne eksperymenty — np. BTC lokalnie korzysta z predyktorów rynkowych wymienionych w tab. 2, a przekazane modele BTC używają m.in. `RISK_SCORE`; dla FX lokalny VAR/BVAR opisano z WTI (§7.1–7.2), podczas gdy tab. 5 wskazuje `OIL_BRENT_PRICE`. Nie musi to oznaczać błędu, lecz **wymaga mapy wersji danych i potwierdzenia, które dokładnie wyniki są porównywane 1:1**.
6. **Różne horyzonty obligacyjne w dwóch kontekstach:** eksperymenty lokalne DNS używają 60/250 dni notowań (§5.1, tab. 4), a przykłady przekazanych modeli obligacji — horyzontów **91/365** (§6.5, tab. 5–6). Przed wspólnym raportowaniem jakości należy wyjaśnić relację tych zadań, jednostek czasu i zbiorów testowych.
7. **Brak pełnej oceny biznesowej:** raport nie dowodzi użyteczności po kosztach transakcyjnych, jakości kalibracji ryzyka, odporności na zmianę reżimu, ani gotowości nadzoru, monitoringu i wycofania modelu. To zrozumiałe wobec zakresu zadania 1.4, lecz konieczne przed zastosowaniem bankowym.

**Proponowany standard dowodu w kolejnym etapie:** dla każdego targetu zamrozić moment prognozy, snapshot i wersje cech, baseline oraz okna testu; opublikować parowane błędy i niepewność przewagi z blokowym bootstrapem lub odpowiednio skorygowanym testem różnic strat; wykonać oddzielne testy na reżimach rynkowych i danych najnowszych. Małe przewagi, takie jak BTC +0,7%, powinny być komunikowane jako **hipotezy**, dopóki nie przejdą takich prób.

## 6. Konsekwencje dla BTC-TAI

**Granica tej części:** nie uzyskałem dostępu do `/opt/data/grok_push/btc-tai-indicators`, dlatego nie stwierdzam, że poniższe komponenty już istnieją lub że ich wyniki są lepsze. „Nasze podejście” rozumiem tu zgodnie z opisem w zleceniu jako **TAI — wskaźniki analizy technicznej oraz dane on-chain**. Rzeczywiste przewagi implementacji trzeba potwierdzić audytem repozytorium i porównaniem na identycznym teście.

### Co przejąć od PKO BP–SGH

1. **Protokół:** chronologiczny train/validation/test i walk-forward; identyczne momenty prognozy i kalendarz dla modelu oraz benchmarku. Dla BTC h=1 i h=30 muszą oznaczać **dni kalendarzowe**, z jednoznacznym czasem zamknięcia (§4.3, §5.1, tab. 4).
2. **Raportowanie:** MAPE i MAE poziomu, naiwna `P(t)` jako baseline, a dla innych zadań — osobny baseline i miary. Dodać rozkład błędów, przedziały ufności i diagnostykę ogonów, nie tylko średnią poprawę (§5.3).
3. **Kontrakt point-in-time:** każda cecha TAI/on-chain powinna mieć nie tylko datę obserwacji, ale **czas faktycznej dostępności dla prognozy**, politykę rewizji, opóźnień i braków (§4.3, §6.1).
4. **Wzorzec integracji:** poziom BTC jako `anchor`, model zwrotu wewnątrz, rekonstrukcja poziomu **w grafie ONNX**, test parytetu lokalnego i `onnxruntime` pozycja po pozycji (§6.2–6.4).
5. **Aktualność:** przed porównaniem z testem raportu PKO odtworzyć historyczny okres do 31.03.2025; następnie wykonać **osobną, z góry zamrożoną ocenę na okresie późniejszym**, bez dostrajania na nim (§4.1, §7.8).
6. **Bezpieczeństwo h=30:** kontrolować ekstremalne prognozowane zwroty, dryf rekonstrukcji, braki predyktorów i działanie przy zmianie reżimu; porównać wariant bezpośredni z rekurencyjnym (§5.1, §7.9).

### Gdzie TAI + on-chain *może* być mocniejsze

W lokalnym teście PKO BTC wykorzystano głównie historię ceny oraz szerokie zmienne rynkowe, takie jak S&P 500, VIX, DXY, UST 10Y i złoto (§4.2, tab. 2; §7.8). Raport sam rekomenduje dodanie **on-chain i sentymentu** (§7.8.3). Jeżeli nasze repo faktycznie zawiera poprawnie datowane, historycznie dostępne wskaźniki techniczne i on-chain, ma **bogatszy zestaw hipotez specyficznych dla BTC** niż opisany eksperyment lokalny. To przewaga **zakresu cech**, nie dowód przewagi prognoz.

Należy ją wykazać ablacyjnie na wspólnym teście: `(a) historia ceny`, `(b) + TAI`, `(c) + on-chain`, `(d) + TAI + on-chain`, przy tych samych chwilach prognozy, modelach bazowych i zasadach dostępności informacji. W szczególności wskaźniki techniczne wyliczone z ceny nie są automatycznie nową informacją, a dane on-chain mogą przychodzić z opóźnieniem lub podlegać rewizjom. Bez kontroli tych kwestii deklarowana przewaga byłaby metodologicznie słabsza od benchmarku PKO.

## 7. Konkretne następne kroki dla współpracy

| Priorytet | Działanie i kryterium zakończenia | Podstawa w raporcie |
|---|---|---|
| **P0 — uzgodnić porównanie** | Z PKO uzgodnić wersję targetu BTC/USD, strefę czasową zamknięcia, jednostkę horyzontu, chwile testowe, baseline i definicję MAPE. Przekazać manifest pozwalający odtworzyć *identyczne* prognozy obu zespołów. | §4.3, §5.1–5.3, §6.3 |
| **P0 — odświeżyć BTC** | Uzupełnić dane od 01.04.2025, oznaczyć pochodzenie i luki; zablokować późniejszy test przed strojeniem. Zweryfikować zgodność starej części szeregu z użytym w raporcie eksportem. | §4.1–4.2, §7.8 |
| **P0 — audyt dostępności cech** | Sporządzić dla TAI, on-chain, sentymentu i `RISK_SCORE` tabelę: źródło, timestamp zdarzenia, timestamp publikacji/dostępności, rewizje, wersja SQL, polityka braków. Odrzucić eksperyment, którego cech nie da się historycznie odtworzyć point-in-time. | §4.3–4.4, §6.1 |
| **P1 — eksperymenty BTC** | Wykonać ablacje cech i porównać modele zwrotowe z naiwną dla h=1 i h=30; raportować parowane różnice błędów, niepewność i wynik na okresie po marcu 2025. Weryfikować stabilność między większą liczbą ziaren. | §5.2, §7.8, tab. 22 |
| **P1 — ścieżka integracyjna** | Dostarczyć prototyp klasy `Model` z anchorem i grafem ONNX; testować zgodność `onnxruntime` z implementacją treningową na tych samych rekordach, wraz z zachowaniem przy nieaktualnej cesze. | §6.2–6.5 |
| **P1 — zabezpieczenia wielokrokowe** | Ustalić limity zwrotów i alarmy niestabilności, sprawdzić ścieżki h=30 w okresach gwałtownych ruchów; nie maskować skrajnego błędu samą średnią MAPE. | §7.1, §7.6 — do weryfikacji, §7.8 |
| **P2 — bramka produkcyjna** | Dopiero po ukończeniu `model_publish` na backendzie Vertex Pipelines, walidacji artefaktu ONNX, monitoringu danych/modelu oraz decyzji właściciela ryzyka rozważać status produkcyjny. | §6.1–6.6, §8 |

Raport jako plan kolejnego etapu wskazuje **systematyczne strojenie hiperparametrów, selekcję i rozszerzenie predyktorów, modelowanie zwrotów zamiast poziomów, stabilizację prognoz wielokrokowych oraz aktualizację danych, szczególnie BTC** (§5.4, §7.8–7.9). Dla współpracy oznacza to najlepszy moment, aby **wspólnie ustalić protokół i definicje cech BTC-TAI przed rozbudową modeli**. Pozwoli to ocenić rzeczywistą wartość on-chain i TAI, a później bez zmiany znaczenia eksperymentu przenieść wybrane rozwiązanie do `pko-fr`.

## 8. Mapowanie: lista modeli/assetów z zlecenia PKO vs to, co faktycznie zrealizowano w zad. 1.4

Poniżej zestawienie listy „modeli do przetestowania i przegrania do Banku” (mail Działu Wewnętrznego z 05.11.2026) z faktycznym zakresem raportu zad. 1.4 (30.10.2026). **Kredyty konsumpcyjne i hipoteczne nie zostały w raporcie zrealizowane wcale** — raport obejmuje 8 grup wskaźników (EUR/PLN, USD/PLN, stopa NBP, krzywa obligacji, WIG20, ropa, złoto, BTC), a w §3/tab. 1 i §7 nie ma ani słowa o kredytach.

**Stopa NBP**
- I: Reguła Taylora / funkcja reakcji — **NIE w raporcie** (brak; jest ARIMA/VAR/BVAR/logit/probit/ordered). Implementacja istnieje po naszej stronie: `experiments/sgh/tier1_nbp.py::TaylorRuleForecaster` (TRD-32), ale bez wyników liczbowych w repo.
- I: Modele liniowe forward-looking — **CZĘŚCIOWO**: raport ma ARIMA/VAR/BVAR na poziomie stopy; nasz `ForwardLinearForecaster` (Ridge na lagach) — kod, brak wyników.
- I: Ordered logit / ordered probit — **TAK** (§7.3.1, tab. 11: BA 96,9% z predyktorami, Macro-F1 65,6%).
- II: VAR, BVAR, SVAR — **VAR/BVAR TAK** (§7.3.1, tab. 10, oba gorsze od naiwnej); **SVAR NIE** w raporcie (mamy `SVARForecaster` w tier2_nbp.py, brak wyników).
- II: Modele z oczekiwaniami i komunikacją — **NIE** (kod: `ExpectationsCommForecaster`, brak wyników).
- II: RPP Random forest, Gradient boosting — **NIE w raporcie dla stopy** (kod: `RandomForestForecaster`, `GradientBoostingForecaster` w tier2_nbp.py, brak wyników).
- II: Modele tekstowe / NLP — **NIE** (kod: `RppNlpForecaster` TF-IDF na komunikatach RPP, brak wyników).

**Obligacje 2/5/7/10Y**
- I: Nelson-Siegel, Svensson, Diebold-Li — **TAK jako estymacja krzywej DNS** (§7.4.2, λ=0,602), ale prognozowanie czynników robi screening AR/VAR/Ridge/PLS/drzewa/PCA, nie klasyczne NS/Svensson/Diebold-Li jako modele prognostyczne. Nasze implementacje (`NelsonSiegelForecaster`, `SvenssonForecaster`, `DieboldLiForecaster`, TRD-34) — kod, brak wyników.
- II: PCA dla czynników, VAR/BVAR/VARX czynników, makrofinansowe, AFNS, PCA-LSTM, LSTM — **CZĘŚCIOWO w raporcie**: PCA i VAR czynników DNS są w §7.4.3; AFNS/PCA-LSTM/LSTM **NIE** w raporcie (kod: `AFNSForecaster`, `PCALSTMForecaster`, `MacroFinanceForecaster`, `VARXFactorsForecaster` w tier2_bonds.py, brak wyników).

**Kredyty konsumpcyjne** — **CAŁY BLOK NIEZREALIZOWANY w raporcie** (brak sekcji). Po naszej stronie pełna implementacja kodu: tier I ARDL/ECM/dynamic regression/logit scorecard (TRD-36), tier II VAR/BVAR/factor/FCI/LASSO/ElasticNet/RF/GBM/PKO-internal (TRD-37) — wszystko bez wyników liczbowych (`results/sgh_credit_consumer_*.json` nie istnieją).

**Kredyty hipoteczne** — **CAŁY BLOK NIEZREALIZOWANY w raporcie**. Kod: tier I ARDL/VECM/dynamic/logit (TRD-38), tier II VAR/panel/contract/RF/GBM/LSTM/PKO-internal (TRD-39) — brak wyników.

**Wniosek dla PKO:** z listy 4 assetów (stopa, obligacje, kredyty konsumpcyjne, kredyty hipoteczne) raport SGH zrealizował w pełni tylko stopę (częściowo — bez SVAR/NLP/RF) i obligacje (częściowo — bez AFNS/LSTM na czynnikach). **Kredyty konsumpcyjne i hipoteczne nie pojawiają się w raporcie wcale**, mimo że są na liście. Modele z tych bloków istnieją jako kod w naszym repo (BTC-TAI, TRD-32..41), ale bez uruchomionych wyników — do „przegrania do Banku” potrzebne są najpierw pełne wyniki walk-forward.

## 9. Werdykt recenzencki

**Zadanie 1.4 można uznać za dobrze udokumentowany etap budowy i technicznego sprawdzenia potoku.** Raport jest szczególnie wartościowy jako opis protokołu, ograniczeń danych i bankowego interfejsu modeli. **Nie uzasadnia natomiast twierdzenia, że wyłoniono produkcyjny model BTC ani że mała poprawa BTC na 30 dni utrzyma się na aktualnych danych.**

Naszą najbardziej wartościową propozycją wobec PKO nie powinien być kolejny nieporównywalny ranking architektur, lecz **aktualny, point-in-time odtwarzalny zestaw cech TAI/on-chain, uczciwy wspólny backtest oraz model możliwy do sprawdzenia w kontrakcie ONNX/anchor**. Wtedy ewentualna przewaga będzie zarówno mierzalna, jak i użyteczna w bankowej ścieżce integracyjnej.