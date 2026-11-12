## D1 — koryguję

Raport SGH podaje test BTC 01.09.2023–31.03.2025: 578 dni kalendarzowych i **549 momentów prognozy** dla h=30 (raport, linie 366 i 2097–2098). Prognozy są dzienne, więc sąsiednie cele 30-dniowe dzielą 29 dziennych przyrostów. Iloraz 549/30 = **18,3** (albo 578/30 = 19,27) to jedynie orientacyjna liczba bloków bez nakładania, a nie wyznaczona z danych efektywna liczebność dla testu Diebolda–Mariano (DM). Ta zależy od autokowariancji *parowanych różnic strat*.

Dla MAPE właściwa baza RW w raporcie wynosi **11,246%** (raport, linia 2108), a SVR **11,047%** (linia 2109): (11,246 − 11,047)/11,246 = **1,7695%**, czyli zaokrąglone 1,8%. W qmd (linie 157–164) podano inną bazę, 11,73%, oraz odchylenie 0,0863 w jednostkach log-zwrotu. Raport nie udostępnia szeregu parowanych strat ani uzasadnienia wartości 0,0863 jako ich odchylenia; jednostka log-zwrotu nie jest jednostką różnicy strat MAPE. Nie da się więc z dostarczonych danych obliczyć rzeczywistej wariancji długookresowej ani statystyki DM dla tej pary prognoz.

**Kontrola rachunku warunkowego, nie wynik testu DM:** nawet przy dodatkowym założeniu, że 0,0863 jest odchyleniem różnic strat w skali ułamka MAPE, i przy n_eff = 549/30 = 18,3, błąd standardowy wynosi 0,0863/√18,3 = **0,02017**. Dwustronny próg 95% z normalnym mnożnikiem 1,96 wyniósłby 1,96 × 0,02017 / 0,11246 = **±35,16% błędu RW**; z orientacyjnym t₁₇ ≈ 2,11: **±37,85%**. Qmd otrzymuje około 33% przez podstawienie 19 bloków i 11,73% do tego samego wzoru (wynik 33,08%), ale to nie jest przeliczony test DM. Cel odpowiada różnicy 0,018 × 0,11246 = **0,0020243** w skali ułamka MAPE. Przy tych samych, bardzo mocnych założeniach i progu z = 1,96 potrzeba (1,96 × 0,0863 / 0,0020243)² = **6982 niezależnych bloków**, czyli **573 lata** po 30/365,25 roku na blok; przy orientacyjnym t = 2,11 wychodzi 8092 bloków i 665 lat. To rachunek dla samego przekroczenia progu, bez założonej mocy testu, niestacjonarności rynku i doboru wielu modeli. **„530 lat” nie jest empirycznym wnioskiem ani wiarygodną prognozą zapotrzebowania na dane.**

**Propozycja zmiany qmd.** W tabeli podać MAPE RW 11,246% i 549 par prognoz oraz usunąć nieudokumentowane 0,0863 jako „rozrzut parowanych różnic”. Próg i lata opisać co najwyżej jako przykład zależny od jawnie przyjętej wariancji, bez nazywania go wynikiem DM. Właściwy wniosek wymaga szeregu dziennych strat SVR i RW, estymacji wariancji długookresowej różnic (co najmniej z uwzględnieniem nakładania h=30) i przedziału ufności dla przewagi; uwzględnić też wybór zwycięzcy spośród modeli.

## D2 — koryguję

Odczytałem strukturę JSON przed obliczeniem: `skill_vs_rw` jest polem rekordu modelu, a składniki MSE są w `metrics.vs_benchmarks` (nie w `metrics.point`). Dla każdego pola przeliczyłem **skill [%] = 100 × (1 − mse_model/mse_rw)**. Źródła: `results/sgh_nbp_rate_tier1.json`, ścieżki `horizons.{1,3,12}.{ordered_logit,ordered_probit,taylor_rule}.metrics.vs_benchmarks`; VAR i BVAR znajdują się w analogicznych ścieżkach `results/sgh_nbp_rate_tier2.json`, nie w pliku tier I.

| Model | h=1 mies. | h=3 mies. | h=12 mies. |
|---|---:|---:|---:|
| ordered logit | +9,8391% | +29,3161% | +1,3267% |
| ordered probit | +14,3614% | +28,4952% | +1,0492% |
| reguła Taylora | −9,3838% | −14,7179% | −108,0153% |
| VAR (tier II) | −97,7824% | −283,8710% | −901,8044% |
| BVAR (tier II) | −1172,6341% | −1919,5406% | −3790,8210% |

Kontrola przykładowa: dla ordered probit, h=1, 100 × (1 − 0,03511002/0,04099790) = **+14,3614%**; dla BVAR, h=1, 100 × (1 − 0,52175327/0,04099790) = **−1172,6341%**. Qmd (linie 248–253) poprawnie oddaje kierunek i większość wielkości po zaokrągleniu. Wpisy BVAR −1172/−1919/−3790% są ucięte, a przy zaokrągleniu do pełnych punktów procentowych powinny brzmieć **−1173/−1920/−3791%**. Nagłówek „tier I” mylnie obejmuje tu także VAR i BVAR z tier II.

**Propozycja zmiany qmd.** Nazwać tabelę „Stopa NBP: modele tier I i II”, albo rozdzielić wiersze według tier. Wartości BVAR zaokrąglić konsekwentnie według jednej reguły; pod tabelą dodać definicję skill opartą na MSE i wskazać oba pliki JSON.

## D3 — koryguję

**Liczby są potwierdzone, interpretacja „RW clone” jako zachowania wyuczonego przez sieci jest obalona.** W `results/sgh_bitcoin_tier1.json` (`horizons.{1,7,30}.bilstm`) i `results/sgh_bitcoin_tier2.json` (`horizons.{1,7,30}.{lstm,tft,hellformer}`) każde z 12 pól `skill_vs_rw` wynosi dokładnie 0,0. Niezależna kontrola: dla każdego z czterech modeli MSE modelu równa się MSE RW: **1 134 584,242518502** przy h=1, **7 577 091,99990828** przy h=7 i **35 569 243,750558645** przy h=30; stąd 1 − MSE_model/MSE_RW = **0,0000** na każdym horyzoncie.

Kod ujawnia mechanizm artefaktu. `evaluation/walk_forward.py`, linie 151–169, podaje do `predict()` wyłącznie bieżący blok testowy długości najwyżej 30 wierszy (`step: 30` w konfiguracjach BTC). `models/_common.py`, linie 36–55, zwraca pusty zbiór okien, gdy blok jest krótszy od okna sieci. Funkcje `predict()` zastępują wtedy brakującą prognozę (`NaN`) bieżącą ceną: `experiments/sgh/tier1_bitcoin.py`, linie 152–156; `experiments/sgh/tier1_wheat.py`, linie 164–168 (LSTM użyty dla BTC); `experiments/sgh/tier2_eurpln.py`, linie 356–361 (TFT użyty dla BTC); `experiments/sgh/tier2_bitcoin.py`, linie 207–212 (HellFormer). W konfiguracji GPU LSTM i HellFormer mają okno 60, TFT 63 (`experiments/configs/sgh_bitcoin_tier2_gpu.yaml`, linie 17–34), zatem przy blokach najwyżej 30 **każda ich prognoza testowa przechodzi przez zastąpienie ceną bieżącą**. Dla BiLSTM siatka dopuszcza okno 30/60/90 (`experiments/configs/sgh_bitcoin_tier1.yaml`, linie 13–16); bez zapisu okna wybranego w każdym foldzie nie można ustalić udziału zastąpień, choć identyczne MSE wskazują, że taki mechanizm wymaga kontroli. Wyniki nie dowodzą, że sieci „nauczyły się” RW ani że poziom ceny nie ma przewidywalnego sygnału.

**Propozycja zmiany qmd.** Zastąpić tezę o zbieżności czterech sieci do optimum RW informacją, że raportowane wyniki są identyczne z RW i że w kodzie występuje ścieżka zastępcza przy braku pełnego okna wejściowego. Przed wnioskiem ekonomicznym ponowić ocenę z oknem zbudowanym z historii dostępnej w chwili każdej prognozy i podać udział rzeczywistych predykcji sieci oraz zastąpień RW. Dopiero wyniki bez tych zastąpień mogą służyć do oceny architektur.

## D4 — potwierdzam

Źródło: `results/sgh_nbp_rate_tier2.json`, rekordy `horizons.{1,3,12}.{svar,rpp_nlp}.metrics.vs_benchmarks`. Przeliczyłem każde pole według **skill [%] = 100 × (1 − mse_model/mse_rw)**, zamiast polegać na zapisanym `skill_vs_rw`:

| Model | h [mies.] | MSE modelu | MSE RW | Przeliczony skill |
|---|---:|---:|---:|---:|
| SVAR | 1 | 0,08569095 | 0,04099790 | −109,0130% |
| SVAR | 3 | 0,99657071 | 0,27969017 | −256,3124% |
| SVAR | 12 | 20,89595296 | 2,61980324 | −697,6154% |
| RPP-NLP | 1 | 13,71279213 | 0,04099790 | −33 347,5483% |
| RPP-NLP | 3 | 12,83002696 | 0,27969017 | −4 487,2284% |
| RPP-NLP | 12 | 9,42955264 | 2,61980324 | −259,9336% |

Zaokrąglenia w qmd (linia 270) są zgodne z tymi wynikami. Ujemny skill jest tu względnym pogorszeniem **MSE**, więc −260% oznacza MSE około 3,60 razy większy niż dla RW, a nie wynik bliski jakościowo RW.

## D5 — koryguję

„**4 z 11**” w qmd (linia 60) dotyczy **rodzin modeli dla stopy NBP**, nie obszarów zlecenia. Z listy NBP wypisanej w qmd (linia 28) sam liczę 4 pozycje tier I + 7 tier II = **11**: reguła Taylora, model liniowy wyprzedzający, ordered logit, ordered probit, VAR, BVAR, SVAR, oczekiwania i komunikacja, las losowy, gradient boosting, NLP komunikatów RPP. Raport SGH (linie 197–198; szczegóły modeli kierunku w liniach 1329–1369) zawiera z tej listy **4**: ordered logit, ordered probit, VAR, BVAR. **Niepokrytych jest 11 − 4 = 7**. Logit, probit i ARIMA z raportu nie są dodatkowymi pozycjami tej listy.

Jeżeli „obszar” oznacza *wskaźnik/zadanie predykcji*, qmd wylicza **14** obszarów (linie 28–54). W tabeli 1 raportu SGH (linie 195–207) znajduję **8 z tych 14**: stopa NBP, obligacje, EUR/PLN, USD/PLN, WIG20, ropa, złoto, BTC. **Niepokrytych jest 14 − 8 = 6**: trzy rodzaje kredytów, depozyty osób prywatnych, aktywa TFI i pszenica. Sam tekst raportu SGH **nie zawiera listy TRD ani rozpiski tier I/II** (kontrola wyszukania „TRD” i „tier” w całym pliku), więc z niego można sprawdzić zakres *wykonany*, ale nie potwierdzić autentyczności i kompletności listy *zleconej* przytoczonej w qmd. Dla tej ostatniej potrzebny jest pierwotny dokument zlecenia.

**Propozycja zmiany qmd.** Pisać „dla stopy NBP: 4 z 11 wymienionych rodzin modeli, 7 brakujących”, a odrębnie „8 z 14 wymienionych obszarów ma reprezentację w raporcie SGH, 6 nie występuje”. Do tabeli dodać źródło pierwotnej listy TRD; bez niego określić pokrycie jako porównanie z listą przytoczoną w qmd, nie jako zweryfikowane wykonanie całego zlecenia. Nie utożsamiać samej obecności obszaru z pełną realizacją jego modeli i horyzontów.

## D6 — potwierdzam

Raport SGH wprost podaje **190** ocenionych kombinacji oraz **188** z pełnymi prognozami (linie 109–112 i 2219–2227); własne odejmowanie daje **190 − 188 = 2** niepełne. W tabeli 23 (linie 2248–2252) wyłącznie wiersz „Ropa – poziom” ma kompletność **16/18**, więc **18 − 16 = 2** braki; tabela 18 wskazuje gwiazdką oba horyzonty wariantu **VMD + XGBoost, tylko target** jako prognozy o niepełnej kompletności (linie 1830–1836). Nie oznacza to braku wszystkich prognoz tych wariantów: raport liczy ich MAPE na dostępnych prognozach (przypis w liniach 1833–1836).

Liczbę GARCH sprawdziłem na trzech wierszach zmienności tabeli 23 (linie 2247–2252): WIG20 **6/6**, ropa **7/8**, złoto **6/6** lepszych od odpowiedniego benchmarku, łącznie **(6 + 7 + 6)/(6 + 8 + 6) = 19/20 = 95%** konfiguracji. Jest to licznik przewag w zastosowanej mierze MAE, nie licznik istotnych statystycznie przewag; sam raport także podaje 19/20 w linii 2268.

## D7 — koryguję

**Poprawa punktowa jest w raporcie, twierdzenie o przekroczeniu granicy wykrywalności jest nieudowodnione.** Raport SGH, tabela 8 (linie 1089–1100), podaje dla EUR/PLN na h=30 MAPE RW 1,228%, VAR 1,154%, BVAR 1,153%; z *zaokrąglonych* MAPE sam liczę odpowiednio **6,0261%** i **6,1075%** poprawy, podczas gdy raport podaje dla obu 6,1% na danych przed zaokrągleniem. Tabela 9 (linie 1215–1226) podaje USD/PLN: RW 2,357%, VAR 2,225%, BVAR 2,227%; własne przeliczenie daje **5,6003%** i **5,5155%** (w raporcie 5,6% i 5,5%). Brak dostępnych plików JSON z prognozami FX i brak szeregu strat, więc ocena istotności poniżej jest **szacunkiem teoretycznym**, nie empirycznym testem DM.

Dla obu kursów raport podaje **937** notowań w teście, ale tylko **31 momentów prognozy** (tabela 3, linia 362); §7.1.1 mówi wprost „co 30 notowań” (linia 1055). Na h=30 prognozy są więc pobierane co 30 sesji: ich 30-sesyjne okresy celu **nie nakładają się mechanicznie**. Liczba par błędów użytecznych do testu wynosi **31**, nie 937, a efektywna liczebność przy zależności między blokami może się od niej różnić. Nie należy w tym układzie dodatkowo dzielić 31 przez 30 ani automatycznie stosować korekty na 29 nakładających się celów; nadal trzeba sprawdzić autokorelację różnic strat między kolejnymi blokami.

**Warunkowy rachunek wykrywalności:** niech `d_i = APE_RW,i − APE_model,i`, a `c = sd(d_i)/średnie_APE_RW`. Dla 31 niezależnych par i dwustronnego progu t₃₀ ≈ 2,042 warunek istotności punktowej poprawy `g` wynosi `c < g√31/2,042`. Dla `g = 5,5–6,1%` graniczny `c` to **0,150–0,166**. Bez obserwowanego `sd(d_i)` nie wiadomo, czy warunek jest spełniony: przykładowo przy `c = 0,10` próg wynosi **3,67%**, a przy `c = 0,20` **7,34%**. To szacunek pomijający dobór modelu spośród 18 konfiguracji na kurs w tabeli 23 (raport, linie 2238–2239), co może podnieść wymagany próg. EUR/PLN i USD/PLN oceniono na tych samych datach, więc ich przewagi nie stanowią dwóch niezależnych replikacji. Dla dwóch ostatnich prognoz VAR/BVAR użyto ostatniej dostępnej ceny WTI (raport, linie 1082–1083), co wymaga analizy wrażliwości. Również teza „jedyny obszar powyżej granicy” nie ma podstaw: dla pozostałych obszarów nie wyznaczono porównywalnych granic na podstawie ich własnych par strat.

**Propozycja zmiany qmd.** Pozostawić liczby 5,5–6,1% jako wyniki punktowe i napisać, że pochodzą z 31 prognoz na kurs bez mechanicznego nakładania 30-sesyjnych celów. Usunąć zdanie o przekroczeniu granicy wykrywalności i o „jedynym obszarze”, dopóki nie ma przedziałów ufności dla parowanych różnic MAPE, kontroli autokorelacji oraz korekty na wybór modeli. Dodać analizę wrażliwości po wyłączeniu dwóch prognoz z nieaktualną ceną WTI.

**Sprostowanie odsyłaczy w audycie.** W D1 wiersze tabeli 22 z MAPE RW i SVR to linie **2110–2111** raportu SGH (linie 2108–2109 są nagłówkami). W D7 informacja o 31 momentach co 30 notowań jest w liniach **1059–1062**, informacja o cenie WTI w liniach **1083–1085**, a liczby 18 konfiguracji FX w liniach **2234–2235**. W D2 określenie tabeli jako „tier I” pochodzi ze specyfikacji tego audytu; sam nagłówek tabeli qmd (linie 244–246) nie przypisuje jej formalnie do tier I.

## Brakujące analizy najważniejsze dla Banku

1. **Parowane prognozy i niepewność przewag:** udostępnić błędy modelu i RW w każdym momencie prognozy, przedziały ufności dla różnic strat, testy uwzględniające zależność w czasie oraz korektę na przeszukanie wielu konfiguracji.
2. **Audyt rzeczywistych prognoz sieci:** dla każdego modelu i horyzontu wykazać odsetek wyjść sieci i zastąpień benchmarkiem, naprawić budowę okna wejściowego z dostępnej historii i ponownie ocenić wyniki BTC.
3. **Walidacja informacji dostępnej w chwili prognozy:** odtworzyć historyczne daty publikacji i rewizje danych makro, osobno przeliczyć modele NBP po usunięciu podglądu CPI oraz FX po wyłączeniu prognoz z nieaktualną ceną ropy.
4. **Odporność w różnych okresach rynku:** pokazać stabilność przewag i ich przedziałów ufności w kolejnych, z góry określonych okresach testowych, w tym poza próbą używaną do wyboru modeli.
5. **Wartość dla decyzji Banku i zakres zlecenia:** przypisać każdej prognozie konkretną decyzję, koszt błędu i próg użyteczności; obok tego przedstawić macierz wszystkich pozycji pierwotnej listy TRD z dowodem realizacji, braku lub świadomego odstąpienia.
