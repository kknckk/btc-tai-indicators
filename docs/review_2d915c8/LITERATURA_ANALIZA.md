# Analiza literatury do raportu review_bank — przegląd po klasach zadania

Stan: 2026-11-12. Dokument roboczy do `review_bank.qmd`, sekcja „Publikacje wybrane do odtworzenia”.
Nie zmienia `references.bib` ani `review_bank.qmd`.

**Statusy:**

- **[bib]** — wpis jest już w `references.bib`;
- **[dodatki]** — nowy wpis w `references_dodatki.bib`; DOI rozwiązuje się na `https://doi.org/` (HEAD → 302), a metadane zgadzają się z Crossref albo DataCite (sprawdzone 2026-11-12);
- **[do potw.]** — wpis w sekcji „DO PÓŹNIEJSZEGO POTWIERDZENIA” w `references_dodatki.bib`. Nie cytować, dopóki DOI nie zostanie potwierdzony.

**Konwencja kolumn.** „Co odtwarzamy” to konkretny wynik, tabela albo figura, z którą porównujemy własną implementację. „Po co w protokole” mówi, którą część naszego protokołu (benchmark, test, metryka, kontrola negatywna) ta pozycja uzasadnia.

Numerów tabel i figur z oryginałów nie podaję z pamięci. Przed odtworzeniem trzeba je odczytać z PDF i wpisać do configu eksperymentu. Opisy wyników to ustalenia główne prac, które są dobrze znane w dziedzinie.

---

## 1. Zmienność — rodzina GARCH (WIG20, ropa, złoto, BTC)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Engle (1982) `engle1982arch`; Bollerslev (1986) `bollerslev1986garch` | GARCH(1,1) z rozkładem normalnym i t: oszacowania α+β bliskie 1 (wysoka persystencja) na dziennych zwrotach każdej serii | obowiązkowy benchmark zmienności (R2); kontrola negatywna implementacji: persystencja musi wyjść w typowym zakresie | [bib] |
| Nelson (1991) `nelson1991egarch` | EGARCH: znak i istotność członu asymetrii na indeksie akcyjnym | tier I WIG20 z listy zlecenia; porównanie z jedynym modelem zmienności, który raport SGH faktycznie liczy | [bib] |
| Glosten–Jagannathan–Runkle (1993) `gjr1993` | GJR-GARCH: dodatni człon γ (efekt dźwigni) na WIG20; brak lub odwrócony znak na BTC i złocie | tier I WIG20 (w SGH nieobecny); test, czy asymetria jest cechą akcji, a nie artefaktem implementacji | [bib] |
| Hansen & Lunde (2005) `hansenlunde2005` | główny wynik pracy: wśród 330 modeli ARCH żaden nie bije istotnie GARCH(1,1) na kursie DM/USD, natomiast na akcjach IBM wygrywają modele z dźwignią; odtwarzamy układ porównania (SPA + różne straty) na EUR/PLN i WIG20 | uzasadnia tezę, że GARCH(1,1) jest trudnym benchmarkiem, i dlaczego wynik SGH 19/20 wobec średniej z 5 obserwacji nic nie mówi | [bib] |
| Katsiampa (2017) `katsiampa2017` | ranking wariantów GARCH dla BTC według kryteriów informacyjnych (najlepiej wypada AR-CGARCH, czyli komponent długookresowy) | kotwica literaturowa dla zmienności BTC; nasz GJR-GARCH na BTC powinien dać podobne uporządkowanie dopasowania | [dodatki] |

## 2. Zmienność zrealizowana, HAR i rozszerzenia

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Andersen & Bollerslev (1998) `andersenbollerslev1998` | R² regresji Mincera–Zarnowitza prognozy GARCH na proxy zmienności: bardzo niski przy kwadracie dziennego zwrotu, wielokrotnie wyższy przy RV z danych śróddziennych | **centralny argument przeciw ocenie SGH na r²**. Niski R² nie musi oznaczać złego modelu, tylko szum proxy. Odtwarzamy to na BTC (mamy dane śróddzienne) | [dodatki] |
| Andersen–Bollerslev–Diebold–Labys (2003) `abdl2003rv` | prognoza log-RV prostym modelem autoregresyjnym jest lepsza od GARCH na danych dziennych (FX) | uzasadnia RV jako cel i benchmark tier II WIG20 („realized volatility” z listy) | [dodatki] |
| Corsi (2009) `corsi2009har` | HAR-RV: oszacowania składników dziennego, tygodniowego i miesięcznego (wszystkie istotne, malejące wagi) oraz przewaga out-of-sample nad AR(1) na horyzontach 1/5/22 dni | obowiązkowy benchmark (R2); kontrola negatywna: znaki i rząd wielkości współczynników | [bib] |
| Bollerslev–Patton–Quaedvlieg (2016) `bollerslevpattonquaedvlieg2016harq` | HARQ: współczynnik przy RV zależy od realized quarticity; poprawa QLIKE względem HAR | współczesny, mocniejszy benchmark zmienności. Jeśli nasz model nie bije HAR, sprawdzamy, czy HARQ go bije | [dodatki] |
| Patton & Sheppard (2015) `pattonsheppard2015` | semiwariancje: ujemna semiwariancja ma większą moc prognostyczną niż dodatnia (SHAR) | asymetria w ramach HAR, odpowiednik GJR po stronie RV | [dodatki] |
| Kim–Shephard–Chib (1998) `kimshephardchib1998sv` | model SV w estymacji MCMC i porównanie dopasowania z GARCH | tier II WIG20 („stochastic volatility”), niepokryty ani przez SGH, ani przez nas | [dodatki] |

Wspierające (już w bib): Patton (2011) `patton2011qlike` dostarcza QLIKE jako stratę odporną na szum proxy, a Hansen & Lunde (2006) `hansenlunde2006jbes` pokazują, że przy złym proxy ranking modeli może się odwrócić.

## 3. Stopa NBP — reguły polityki i modele decyzji dyskretnych (tier I)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Taylor (1993) `taylor1993` | reguła statyczna i = r* + π + 0,5(π−π*) + 0,5·luka; dopasowanie do ścieżki stopy w historii | kotwica tier I (u nas −9% / −15% / −108% vs RW, co wymaga wyjaśnienia, patrz niżej) | [bib] |
| Clarida–Galí–Gertler (2000) `claridagaligertler2000` | reguła **forward-looking z wygładzaniem stopy** (ρ·i_{t−1}); oszacowania ρ są wysokie | wyjaśnia, dlaczego statyczna reguła Taylora przegrywa z RW: brak członu wygładzania. Wariant z ρ to właściwy model tier I | [dodatki] |
| Orphanides (2001) `orphanides2001realtime` | oszacowania reguły Taylora na danych real-time różnią się istotnie od oszacowań na danych zrewidowanych | **bezpośrednia podstawa zarzutu wobec panelu NBP w SGH** (cechy nieznane w dniu decyzji) i rekomendacji R4 | [dodatki] |
| Croushore & Stark (2001) `croushorestark2001` | konstrukcja zbioru vintage'y i pokaz, że wnioski prognostyczne zależą od rocznika danych | wzorzec dla panelu vintages HICP/ESI w R4 | [dodatki] |
| McCullagh (1980) `mccullagh1980ordered` | ordered logit/probit na kategoriach obniżka / bez zmian / podwyżka | szkielet modelu decyzji RPP (u nas +9,8% / +14,4% vs RW na h=1) | [bib] |
| Hamilton & Jordà (2002) `hamiltonjorda2002` | model ACH-ordered probit: osobno moment zmiany i kierunek/wielkość zmiany docelowej stopy Fed | lepszy benchmark dyskretny niż statyczny ordered: modeluje rzadkość zmian, czyli dokładnie problem „17 momentów, 1 zmiana” | [dodatki] |

## 4. Stopa NBP — VAR/BVAR/SVAR i komunikacja (tier II)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Sims (1980) `sims1980var` | VAR w zredukowanej postaci jako punkt wyjścia | tier II; u nas VAR silnie przegrywa z RW (−98% na h=1) | [bib] |
| Litterman (1986) `litterman1986bvar` | BVAR z priorem Minnesota bije nieograniczony VAR w prognozach | uzasadnienie BVAR. **Nasz BVAR (−1172%) jest gorszy od VAR**, co jest sprzeczne z tą literaturą i wskazuje na błąd hiperparametrów priora albo implementacji | [bib] |
| Bańbura–Giannone–Reichlin (2010) `banburagiannonereichlin2010` | duży BVAR: optymalny shrinkage rośnie z wymiarem, a BVAR na wielu zmiennych bije mały VAR | procedura doboru λ (dopasowanie dopasowania in-sample do małego VAR). Kontrola negatywna dla BVAR | [dodatki] |
| Kilian & Lütkepohl (2017) `kilianlutkepohl2017svar` | identyfikacja SVAR (rekurencyjna, znakowa) | referencja dla SVAR tier II; dokumentacja wybranego schematu identyfikacji | [dodatki] |
| Hansen & McMahon (2016) `hansenmcmahon2016` | miary treści komunikatów FOMC (LDA) i ich wpływ w FAVAR | wzorzec dla „NLP komunikatów RPP”. U nas NLP daje −33 348% na h=1, co oznacza błąd skali celu albo przeuczenie i wymaga diagnozy przed raportowaniem | [dodatki] |
| Gentzkow–Kelly–Taddy (2019) `gentzkowkellytaddy2019` | przegląd metod tekst→cechy oraz ostrzeżenia o przeuczeniu przy p ≫ n | uzasadnienie regularyzacji przy ~300 obs. miesięcznych | [dodatki] |

## 5. Krzywa rentowności (obligacje 2/5/7/10 lat)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Nelson & Siegel (1987) `nelsonsiegel1987` | trzyczynnikowa forma krzywej | fundament; w SGH tylko wygładzanie danych | [bib] |
| Svensson (1994) `svensson1994` | rozszerzenie o drugi czynnik krzywizny | tier I z listy; w SGH nieobecny jako model prognostyczny | [dodatki] |
| Diebold & Li (2006) `dieboldli2006` | DNS z λ ustalonym i AR(1) na czynnikach; prognozy na 1/6/12 mies. vs RW. Na 1 mies. RW jest trudny do pobicia, na 6–12 mies. DNS wygrywa | **główny cel replikacji dla obligacji**: tabela RMSE per tenor i horyzont, liczona na rentownościach (bp), a nie na czynnikach (R6) | [bib] |
| Diebold–Rudebusch–Aruoba (2006) `dieboldrudebuscharuoba2006` | DNS w postaci stanowej z makro (aktywność, inflacja, stopa) | wariant „makro-finansowy” z tier II listy | [dodatki] |
| Christensen–Diebold–Rudebusch (2011) `christensendieboldrudebusch2011afns` | AFNS: wersja bez arbitrażu; porównanie prognoz z DNS | tier II (AFNS), brak w SGH | [dodatki] |
| Litterman & Scheinkman (1991) `littermanscheinkman1991` | trzy główne składowe (poziom, nachylenie, krzywizna) wyjaśniają niemal całą zmienność | tier II „PCA krzywej”; kontrola: udział wariancji pierwszych 3 PC na krzywej PL | [dodatki] |

Uzupełniające [dodatki]: Duffee (2002) `duffee2002` pokazuje, że standardowe modele afiniczne prognozują gorzej niż RW, więc RW jest obowiązkowy także dla krzywej. Ang & Piazzesi (2003) `angpiazzesi2003` to VAR bez arbitrażu z makro, tier II „VARX/makro-finansowe”.

## 6. Kursy walutowe (EUR/PLN, USD/PLN)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Meese & Rogoff (1983) `meeserogoff1983` | RMSE modeli strukturalnych vs RW na 1/6/12 mies.: RW nie zostaje pobity | **prior dziedziny**: każda przewaga nad RW w FX wymaga testu. Dotyczy wyniku SGH VAR/BVAR +5–6% | [dodatki] |
| Cheung–Chinn–Garcia Pascual (2005) `cheungchinnpascual2005` | ponowna ocena modeli lat 90.: żaden model nie bije RW konsekwentnie w MSE; pewne sukcesy w kierunku na długich horyzontach | uzasadnia równoległą ocenę kierunku (Pesaran–Timmermann) dla FX | [dodatki] |
| Rossi (2013) `rossi2013fx` | przegląd: przewidywalność zależy od predyktora, horyzontu, próby i **metody ewaluacji**; dla modeli zagnieżdżonych właściwy jest test Clarka–Westa | uzasadnia wybór testu (RW jest zagnieżdżony w VAR/ARIMA), patrz klasa 12 | [dodatki] |
| Nie i in. (2023) `nie2023patchtst` | PatchTST: wyniki na standardowych benchmarkach LTSF (w tym Exchange) | tier I FX z listy; w SGH nieobecny dla FX | [bib] |
| Lim i in. (2021) `lim2021tft` | TFT: straty kwantylowe P50/P90 na zbiorach referencyjnych (w tym zbiór zmienności) | jedyna pozycja listy FX pokryta w SGH. Odtwarzamy straty kwantylowe, żeby sprawdzić, czy implementacja jest poprawna | [dodatki] |
| Wolpert (1992) `wolpert1992stacking` | stacking z meta-uczniem trenowanym na prognozach out-of-fold | tier I FX („stacking XGBoost+ARIMA+BiLSTM”). Warunek konieczny: prognozy bazowe out-of-fold w każdym oknie walk-forward | [dodatki] |

## 7. WIG20 i rynek akcji — przewidywalność zwrotów, event study, sentyment

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Welch & Goyal (2008) `welchgoyal2008` | R² out-of-sample predyktorów premii za ryzyko jest w większości ujemny; figura skumulowanej różnicy SSE (model vs średnia historyczna) | wzorzec raportowania: figura skumulowanej różnicy strat w czasie dla WIG20 i BTC zamiast jednej liczby | [dodatki] |
| Campbell & Thompson (2008) `campbellthompson2008` | ograniczenia na znaki współczynników poprawiają OOS R²; nawet mały dodatni R² daje korzyść ekonomiczną | zakotwiczenie skali: dla zwrotów dziennych OOS R² rzędu dziesiątych procenta to już wynik | [dodatki] |
| Brown & Warner (1985) `brownwarner1985` | symulacje mocy testów nadzwyczajnych zwrotów na danych dziennych | tier I WIG20 („event study”), np. posiedzenia RPP i publikacje CPI | [dodatki] |
| MacKinlay (1997) | przegląd metodyki event study | jak wyżej; DOI niepotwierdzony | [do potw.] |
| Tetlock (2007) `tetlock2007` | pesymizm w mediach przewiduje spadki zwrotów i powrót w kolejnych dniach | tier II „modele sentymentu” | [dodatki] |
| Gu–Kelly–Xiu (2020) `gukellyxiu2020` | miesięczny OOS R² modeli ML (drzewa, sieci) jest mały, ale dodatni; ranking metod | tier II „las losowy, gradient boosting” dla akcji. Daje realistyczną skalę oczekiwań | [dodatki] |

## 8. Surowce (ropa, złoto, pszenica)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Alquist–Kilian–Vigfusson (2013) `alquistkilianvigfusson2013` | prognoza „bez zmian” jest trudna do pobicia; kontrakty futures nie biją jej systematycznie | benchmark RW i futures dla ropy; kotwica dla wyników ARIMA/EGARCH w SGH | [dodatki] |
| Baumeister & Kilian (2012) `baumeisterkilian2012` | prognozy real-time ceny ropy: VAR bije „bez zmian” na krótkich horyzontach | wzorzec ewaluacji real-time (vintage) dla ropy | [dodatki] |
| Engle–Ghysels–Sohn (2013) `englegyselssohn2013garchmidas` | GARCH-MIDAS: długookresowy składnik zmienności sterowany zmiennymi makro o niższej częstotliwości | tier I ropy z listy; brak w SGH i u nas | [dodatki] |
| Huang i in. (1998) `huang1998emd`; Dragomiretskiy & Zosso (2014) `dragomiretskiyzosso2014vmd` | definicje EMD i VMD | tier II ropy (EMD) i wariant SGH VMD+XGBoost. **Ryzyko przecieku:** dekompozycja na całej próbie używa przyszłości. Trzeba ją liczyć w każdym oknie, tylko na danych do T | [dodatki] |
| Primiceri (2005) `primiceri2005tvpvar` | TVP-VAR ze stochastyczną zmiennością | tier II złota z listy („TVP-VAR”) | [dodatki] |
| Baur & Lucey (2010) `baurlucey2010` | złoto jako hedge (średnio) i safe haven (w krachach) wobec akcji i obligacji | uzasadnienie cech zewnętrznych (indeksy akcji, VIX) w modelu złota | [dodatki] |

## 9. Bitcoin

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Urquhart (2016) `urquhart2016` | testy efektywności: nieefektywność w pełnej próbie, ewolucja ku efektywności w późniejszej podpróbie | prior: przewidywalność BTC maleje w czasie. Argument za R3 (test na świeżym oknie) | [dodatki] |
| Liu & Tsyvinski (2021) `liutsyvinski2021` | zwroty krypto przewidywane przez momentum i uwagę inwestorów, nie przez czynniki rynku akcji | wybór cech: momentum i uwaga zamiast S&P/VIX/DXY (lokalny zestaw cech SGH) | [dodatki] |
| Jaquart–Dann–Weinhardt (2021) `jaquartdannweinhardt2021` | trafność klasyfikacji kierunku na krótkich horyzontach nieco powyżej 50%; ocena strategii po kosztach | skala realistycznych oczekiwań i wzorzec oceny po kosztach. Zgodne z naszym wynikiem „nikt nie bije buy&hold po kosztach” | [dodatki] |
| Chen & Guestrin (2016) `chen2016xgboost` | XGBoost na log-zwrotach z cechami on-chain | baseline nieliniowy (u nas wyeliminowany przez MCS na h=30) | [bib] |
| Kehinde i in. (2025) `kehinde2025helformer` | Helformer (Holt-Winters + Transformer): deklarowane bardzo dobre metryki na BTC | „HellFormer” z listy. **Cel replikacji krytycznej:** odtworzyć wynik na protokole autorów, potem na naszym walk-forward z purge. U nas na poziomie ceny model zbiega do RW (skill 0,0000) | [dodatki] |
| Katsiampa (2017) | patrz klasa 1 | zmienność BTC | [dodatki] |

## 10. Uczenie maszynowe tablicowe: drzewa, regularyzacja, czynniki

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Breiman (2001) `breiman2001rf` | las losowy; błąd OOB | tier II NBP i kredytów; cytowanie metody, którą obaj liczymy | [dodatki] |
| Friedman (2001) `friedman2001gbm` | gradient boosting | jak wyżej (gradient boosting w NBP: −637% na h=1) | [dodatki] |
| Tibshirani (1996) `tibshirani1996lasso`; Zou & Hastie (2005) `zouhastie2005enet` | LASSO, ElasticNet | tier II kredytów konsumpcyjnych i firmowych z listy; naturalny baseline przy p ≫ n | [dodatki] |
| Stock & Watson (2002) `stockwatson2002di` | prognozy z indeksów dyfuzji (czynniki z dużego panelu) biją AR i małe VAR | tier II kredytów („modele czynnikowe”); alternatywa dla 19-cechowego panelu NBP | [dodatki] |
| Makridakis–Spiliotis–Assimakopoulos (2018) `makridakisspiliotisassimakopoulos2018` | na miesięcznych seriach M3 metody statystyczne (ETS, Theta, ARIMA) biją metody ML | uzasadnia obowiązkowe benchmarki statystyczne obok RW | [dodatki] |
| Chen & Guestrin (2016) | patrz klasa 9 | — | [bib] |

## 11. Sieci neuronowe i modele fundamentowe (TSFM)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Hochreiter & Schmidhuber (1997) `hochreiterschmidhuber1997lstm` | architektura LSTM | cytowanie rodziny LSTM/BiLSTM z listy (BTC, pszenica, złoto) | [dodatki] |
| Fischer & Krauss (2018) `fischerkrauss2018` | LSTM na spółkach S&P 500: dodatnie zwroty strategii przed kosztami, zanikające w ostatnich latach próby | najbardziej cytowany dowód „LSTM działa na rynku”. Odtwarzamy też jego słabnięcie w czasie | [dodatki] |
| Zeng i in. (2023) `zeng2023dlinear` | DLinear (jedna warstwa liniowa) bije Transformery LTSF na standardowych zbiorach | **brakujący benchmark liniowy**: każda sieć (PatchTST, TFT, Helformer) musi bić DLinear, nie tylko RW | [dodatki] |
| Nie i in. (2023) `nie2023patchtst`; Luo & Wang (2024) `luo2024moderntcn` | wyniki PatchTST i ModernTCN na ETT/Weather/Exchange | kontrola negatywna implementacji: odtworzyć MSE z tabel autorów na ich zbiorach, zanim uruchomimy na BTC | [bib] |
| Ansari i in. (2024) `ansari2024chronos`; Das i in. (2024) `das2024timesfm`; Woo i in. (2024) `woo2024moirai` | zero-shot na benchmarkach autorów | TSFM: u nas −3,5% do −15,1% vs RW na BTC. Moirai jako trzeci, niezależny TSFM | [bib] / [dodatki] |
| Makridakis i in. (2020) `makridakis2020m4`; (2022) `makridakis2022m5` | M4: najlepsze są kombinacje i hybrydy, czyste ML wypadają słabo; M5: wygrywa LightGBM | kalibracja oczekiwań wobec ML; kombinacje prognoz jako tani i mocny benchmark | [dodatki] |

## 12. Testy porównania prognoz i kontrola przesiewania

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Diebold & Mariano (1995) `dieboldmariano1995`; Harvey–Leybourne–Newbold (1997) `harveyleybournenewbold1997` | test DM z wariancją HAC do rzędu h−1 i poprawka HLN; odtwarzamy rozmiar testu w symulacji przy nakładających się celach | podstawa R1 | [bib] |
| Diebold (2015) `diebold2015dm20` | autor wyjaśnia, że DM porównuje **prognozy**, a nie modele, i że stosowanie go do modeli zagnieżdżonych to nadużycie | granice stosowania DM, które trzeba opisać w R1 | [dodatki] |
| Clark & West (2007) `clarkwest2007` | skorygowany MSPE dla modeli zagnieżdżonych; DM jest w tym przypadku niedowymiarowany | **RW jest zagnieżdżony w ARIMA/VAR/HAR**. Dla par „model vs RW” właściwy jest CW, nie DM | [dodatki] |
| Giacomini & White (2006) `giacominiwhite2006` | test warunkowej zdolności predykcyjnej (metoda = model + okno estymacji) | właściwy test przy rolling window i zmiennych reżimach (BTC, stopa NBP 2020–21) | [dodatki] |
| White (2000) `white2000rc`; Hansen (2005) `hansen2005spa` | Reality Check i SPA: czy najlepszy z K modeli bije benchmark po korekcie na przesiewanie | korekta dla ~190 wariantów SGH i dla wyboru „najsilniejszego wyniku” obligacji | [dodatki] |
| Romano & Wolf (2005) `romanowolf2005`; Hansen–Lunde–Nason (2011) `hansen2011mcs` | stepwise RW i MCS z bootstrapem blokowym | kontrola wielokrotności, zbiorczy MCS po 6 seriach | [bib] |

## 13. Metryki, ocena kierunku, ocena probabilistyczna i przedziałowa

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Hyndman & Koehler (2006) `hyndmankoehler2006mase` | MASE: definicja i własności, wady MAPE przy wartościach bliskich zera | R7 | [bib] |
| Gneiting (2011) `gneiting2011point` | funkcja straty musi być spójna z funkcjonałem, który prognozujemy (średnia → MSE, mediana → MAE); MAPE premiuje prognozy obciążone w dół | **teoretyczne uzasadnienie R7**: ranking pod MAPE może promować systematycznie zaniżone prognozy | [dodatki] |
| Patton (2011) `patton2011qlike` | QLIKE i MSE jako jedyne straty odporne na szum proxy w swojej klasie | R2 | [bib] |
| Pesaran & Timmermann (1992) `pesarantimmermann1992` | test trafności kierunku | ocena kierunku dla FX, BTC i decyzji RPP; uzupełnia samą trafność | [dodatki] |
| Brier (1950) `brier1950`; Epstein (1969) `epstein1969rps`; Gneiting & Raftery (2007) `gneitingraftery2007` | Brier score i **Ranked Probability Score** dla kategorii uporządkowanych; ściśle właściwe reguły oceny | **właściwa metryka dla decyzji RPP** (obniżka / bez zmian / podwyżka): RPS ocenia prognozę probabilistyczną, a nie trafność, która przy 1 zmianie na 17 jest bezwartościowa | [dodatki] |
| Christoffersen (1998) `christoffersen1998`; Kupiec (1995) `kupiec1995`; Romano i in. (2019) `romano2019cqr`; Gibbs & Candès (2021) `gibbscandes2021aci` | testy pokrycia bezwarunkowego (Kupiec) i warunkowego (Christoffersen); CQR; adaptacyjny conformal pod zmianą rozkładu | formalny test naszego twierdzenia „pokrycie 49–87% przy nominalnych 80%”; ACI jako kalibracja online dla BTC | [dodatki] / [bib] |

## 14. Protokół walk-forward, przeuczenie backtestu, ocena ekonomiczna

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Tashman (2000) `tashman2000` | ewaluacja z przesuwanym początkiem (rolling origin) vs pojedynczy podział | literaturowe uzasadnienie walk-forward zamiast podziału SGH 70/15/15 | [dodatki] |
| Bergmeir & Benítez (2012) `bergmeirbenitez2012` | blokowa CV dla szeregów czasowych vs ewaluacja na ostatnim bloku | uzasadnienie purge/embargo i wewnętrznej walidacji w oknie | [dodatki] |
| Bailey–Borwein–López de Prado–Zhu (2014) `baileyborweinlopezdepradozhu2014` | minimalna długość backtestu jako funkcja liczby przetestowanych konfiguracji | liczbowe wsparcie argumentu o ~190 wariantach SGH | [dodatki] |
| Bailey & López de Prado (2014) `baileylopezdeprado2014dsr` | Deflated Sharpe Ratio | korekta naszych Sharpe 0,74–1,03 o liczbę testowanych strategii | [dodatki] |
| Ledoit & Wolf (2008) `ledoitwolf2008sharpe` | test różnicy Sharpe dwóch strategii (HAC / bootstrap) | **brakuje go u nas**: „buy&hold 1,04–1,07 vs 0,74–1,03” to porównanie bez testu | [dodatki] |
| Harvey–Liu–Zhu (2016) `harveyliuzhu2016`; Hou–Xue–Zhang (2020) `houxuezhang2020` | próg t > 3 po korekcie na wielokrotność; większość opublikowanych anomalii nie przechodzi replikacji | cytowanie dla zdania w qmd „literatura sama ma niską replikowalność” (obecnie bez źródła) | [dodatki] |

## 15. Kredyty, depozyty, TFI (obszary 3–7; nieobecne w raporcie SGH)

| Pozycja | Co odtwarzamy | Po co w protokole | Status |
|---|---|---|---|
| Pesaran–Shin–Smith (2001) `pesaranshinsmith2001ardl` | ARDL i test granic (bounds test) na relację długookresową | tier I kredytów i depozytów (ARDL) | [dodatki] |
| Engle & Granger (1987) `englegranger1987`; Johansen (1991) `johansen1991` | ECM, kointegracja, VECM | tier I (ECM, VECM dla hipotek) | [dodatki] |
| Lown & Morgan (2006) `lownmorgan2006` | ankieta standardów kredytowych wyprzedza akcję kredytową | tier II kredytów firmowych („ankiety standardów kredytowych”); polski odpowiednik: ankieta NBP | [dodatki] |
| Hatzius i in. (2010) `hatzius2010fci` | konstrukcja FCI i jego moc prognostyczna dla aktywności | tier II kredytów („FCI”) | [dodatki] |
| Sirri & Tufano (1998) `sirritufano1998` | wypukła relacja przepływów do funduszy i ich wyników | tier I TFI („flow-performance”) | [dodatki] |
| Ghysels–Sinko–Valkanov (2007) `ghyselssinkovalkanov2007midas` | regresje MIDAS | tier II TFI (MIDAS); także podstawa GARCH-MIDAS | [dodatki] |

---

## Co w raporcie SGH cytuje się bezkrytycznie i czego brakuje

Raport SGH nie ma sekcji przeglądu literatury (zob. qmd, „Czego w raporcie nie ma wcale”). Problemem nie jest więc złe cytowanie, tylko **brak kotwiczenia wyników**. Bez cytowań przyjęte zostały w szczególności: lista architektur ze zlecenia, podział 70/15/15 i benchmark „średnia z 5 obs.”. Konkretne braki, z pozycjami, które je zamykają:

1. **Testy istotności dla poziomów i par zagnieżdżonych.** Brak DM/HLN dla wszystkich rankingów poziomów. Ponadto prawie wszystkie porównania SGH to „model vs RW”, gdzie RW jest zagnieżdżony w ARIMA/VAR. Właściwy jest wtedy Clark–West (`clarkwest2007`), a DM jest niedowymiarowany (`diebold2015dm20`). Przy ~190 wariantach brakuje korekty na przesiewanie: Reality Check / SPA (`white2000rc`, `hansen2005spa`), Romano–Wolf, MCS.

2. **Benchmarki zmienności i proxy.** Benchmark „średnia z 5 obs.” nie występuje w literaturze jako standard. Brakuje GARCH(1,1) (`hansenlunde2005`), HAR-RV (`corsi2009har`) i nowszego HARQ (`bollerslevpattonquaedvlieg2016harq`). Ocena na kwadracie zwrotu bez QLIKE jest wprost tym, co Andersen & Bollerslev (1998) pokazują jako źródło mylnie niskiej oceny modeli zmienności. Hansen & Lunde (2006) pokazują z kolei, że może to odwrócić ranking. Wynik SGH „19/20” jest więc nieinterpretowalny.

3. **Modele krzywej dochodowości jako modele prognostyczne.** Nelson–Siegel/Svensson/Diebold–Li służą w SGH tylko do wygładzania. Brakuje replikacji tabeli Diebolda–Li (RMSE per tenor i horyzont vs RW, w bp), AFNS (`christensendieboldrudebusch2011afns`), PCA (`littermanscheinkman1991`) i wariantu makro (`dieboldrudebuscharuoba2006`). Brakuje też odniesienia do tego, że RW jest dla rentowności trudny do pobicia (`duffee2002`). Wynik „+36,3% dla nachylenia” nie ma literaturowego punktu odniesienia.

4. **Ewaluacja kierunku decyzji RPP.** „96,9% trafności” to metryka niewłaściwa dla rzadkich zdarzeń uporządkowanych. Właściwe są RPS (`epstein1969rps`) i Brier (`brier1950`) na prognozie probabilistycznej, test kierunku Pesarana–Timmermanna oraz porównanie z modelem uwzględniającym rzadkość zmian (`hamiltonjorda2002`). Dane w dniu decyzji wymagają vintages (`orphanides2001realtime`, `croushorestark2001`). Bez tego wynik jest zawyżony dwoma niezależnymi mechanizmami.

5. **Reguła Taylora w poprawnej postaci.** Lista zlecenia wymienia regułę Taylora, a SGH jej nie liczy. U nas statyczna reguła z 1993 r. przegrywa z RW (−108% na h=12). Literatura (`claridagaligertler2000`) wskazuje, że empirycznie właściwa jest reguła z wygładzaniem stopy i członem forward-looking. To uwaga także do naszej implementacji.

6. **Prior „RW jest trudny do pobicia” dla FX i surowców.** Wynik SGH VAR/BVAR +5–6% dla EUR/PLN i USD/PLN jest sprzeczny z wieloletnią literaturą (`meeserogoff1983`, `cheungchinnpascual2005`, `rossi2013fx`). To nie dowodzi błędu, ale podnosi poprzeczkę dowodową: potrzebny jest Clark–West i test stabilności w podokresach. Analogicznie dla ropy (`alquistkilianvigfusson2013`).

7. **Przeciek w dekompozycjach (VMD/EMD).** Wariant SGH VMD+XGBoost dla ropy wymaga sprawdzenia, czy dekompozycja była liczona w każdym oknie tylko na danych do T. VMD/EMD na całej próbie to znany mechanizm sztucznie dobrych wyników. Raport tego nie dokumentuje.

8. **Brak benchmarków statystycznych i liniowych dla sieci.** Brakuje DLinear (`zeng2023dlinear`) jako minimum dla Transformerów i ETS/Theta/kombinacji (`makridakisspiliotisassimakopoulos2018`, `makridakis2020m4`) jako minimum dla ML. Przewaga sieci nad samym RW nie świadczy o wartości architektury.

9. **Architektury z listy przyjęte bez weryfikacji publikacji źródłowej.** Dotyczy to zlecenia, nie tylko SGH: HellFormer = Helformer (`kehinde2025helformer`, 2025, Journal of Big Data, deklarowane bardzo wysokie metryki), PSO-LSTM, VMD-CNN-BiLSTM, Anomaly Transformer z IsolationForest. Dla tej klasy prac należy najpierw odtworzyć wynik na protokole autorów, a potem na walk-forward z purge. Rozjazd jest wtedy informacją o protokole, nie o modelu.

10. **Metryka MAPE.** Poza argumentem praktycznym (`hyndmankoehler2006mase`) jest argument teoretyczny (`gneiting2011point`): MAPE nie jest spójna z prognozą średniej i premiuje prognozy zaniżone.

11. **Kalibracja przedziałów bez testu pokrycia.** Ani SGH, ani my nie raportujemy testów Kupca i Christoffersena. Nasze „49–87% przy 80%” powinno mieć p-wartość pokrycia warunkowego.

12. **Ocena ekonomiczna bez testu.** Dotyczy naszej części: porównanie Sharpe strategii vs buy&hold wymaga testu Ledoita–Wolfa (`ledoitwolf2008sharpe`) i deflacji o liczbę strategii (`baileylopezdeprado2014dsr`).

13. **Obszary 3–7 (kredyty, depozyty, TFI) bez żadnej literatury.** Standardowe punkty startowe to ARDL i bounds test (`pesaranshinsmith2001ardl`), ECM/VECM (`englegranger1987`, `johansen1991`), ankiety kredytowe (`lownmorgan2006`), FCI (`hatzius2010fci`), flow-performance (`sirritufano1998`) i MIDAS (`ghyselssinkovalkanov2007midas`).

## Uwagi do istniejącego `references.bib` (bez zmian w pliku)

Do decyzji autora:

- **Brak DOI przy 5 wpisach** wbrew regule „DOI obowiązkowe”: `nie2023patchtst`, `luo2024moderntcn`, `ansari2024chronos`, `das2024timesfm`, `romano2019cqr`. Zweryfikowane DOI preprintów arXiv (doi.org → 302):
  - `10.48550/arXiv.2211.14730` (PatchTST)
  - `10.48550/arXiv.2312.16635` (ModernTCN)
  - `10.48550/arXiv.2403.07815` (Chronos)
  - `10.48550/arXiv.2310.10688` (TimesFM)
  - `10.48550/arXiv.1905.03222` (CQR)
- **Klucz `hansenlunde2006jbes` sugeruje JBES**, a wpis poprawnie wskazuje Journal of Econometrics 131. Myli tylko nazwa klucza.
- **qmd, tabela „Odtwarzane pozycje”:** „ModernTCN (2023)” vs bib `year = {2024}` (ICLR 2024).
- **Publikacje recenzowane:** Chronos wyszedł w TMLR (2024), TimesFM w ICML 2024 (PMLR 235). Bib cytuje je jako preprinty arXiv. Nie zweryfikowałem tego tutaj, do ewentualnej aktualizacji.
- **Zdanie qmd o niskiej replikowalności literatury** („znane są prace pokazujące…”) nie ma cytowania. Kandydaci: `harveyliuzhu2016`, `houxuezhang2020`.

## Weryfikacja

- 80 DOI z Crossref (78 w części zweryfikowanej + Brown–Warner + odrzucony kandydat MacKinlay) sprawdzono przez `curl -I https://doi.org/<doi>`. Wynik: 79 × 302, 1 × 404 (MacKinlay, przeniesiony do „DO PÓŹNIEJSZEGO POTWIERDZENIA”).
- 7 DOI arXiv (10.48550/*): 7 × 302, metadane potwierdzone w DataCite.
- Metadane (tytuł, autorzy, czasopismo, tom, numer, strony) sprawdzone w `api.crossref.org/works`. Ręcznie uzupełnione strony końcowe dla 6 wpisów, gdzie Crossref podaje tylko stronę początkową (oznaczone komentarzem w bib).
