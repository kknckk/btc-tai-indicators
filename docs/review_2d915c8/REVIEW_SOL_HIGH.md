# Niezależna recenzja metodologiczna draftu `review.qmd` (commit `2d915c8`)

## Werdykt ogólny

Draft trafnie identyfikuje kilka rzeczywistych defektów kodu i artefaktów, ale część wniosków metodologicznych jest zbyt kategoryczna. Przed przekazaniem do PKO trzeba skorygować interpretację DM/MCS i sprawdzić spójność protokołu ewaluacji poza pięcioma wskazanymi defektami.

## D1 — potwierdzam diagnozę błędnego testu CQR

W `tests/test_evaluation_harness.py:110–119` kwantyle są skonstruowane jako `q_lo = y - 2` i `q_hi = y + 2`. `calibrate_frame_cqr` (`evaluation/calibration.py:222–242`) wywołuje `rolling_cqr` (`:153–178`), który dla przeszłych wierszy oblicza `cqr_scores = max(q_lo-y, y-q_hi)` (`:39–45`) i odejmuje/dodaje jego kwantyl do granic bieżącego przedziału. Wszystkie score wynoszą `-2`, więc kwantyl wynosi `-2`, przedział bieżący ma granice `y` i `y`, a raportowane pokrycie wynosi 1,0. Uruchomiony lokalnie pojedynczy test kończy się `1 failed`, z `empirical_coverage=1.0`, `median_width=0.0`, `n_eval=100`; asercja wymaga najwyżej 0,92. To deterministyczna wada danych testowych: kwantyle używają realizacji, którą miały prognozować.

Proponuję w drafcie wybrać naprawę z kwantylami wyznaczonymi bez znajomości `y` oraz sprawdzeniem pokrycia i szerokości. Samo dopuszczenie 100% pokrycia uczyniłoby test zielonym, ale nie sprawdzałoby własności kalibracji.

## D2 — potwierdzam błąd parametru DM, koryguję zakres wniosku i naprawy

`evaluation/mcs.py:71` rzeczywiście wywołuje `dm_matrix(..., h=1)` niezależnie od horyzontu; `experiments/run_economic.py:166` woła `mcs` bez `h`. W `evaluation/dm_test.py:21–30` ustawienie `h=1` usuwa wszystkie autokowariancje i wyłącza właściwą korektę HLN. Dla nakładających się celów 7- i 30-dniowych to wadliwy test DM. Jednak zdanie, że **zawsze** „zawyża istotność”, jest zbyt mocne: przy dodatniej autokorelacji strat zaniża wariancję i może zaniżać `p`, lecz znak i wielkość błędu zależą od rzeczywistych autokowariancji. Ponadto różnice kwadratowych strat przy zmiennej wariancji i uczących się modelach nie muszą być dokładnie MA(`h-1`). Liczb `+2,15/0,031` i `+0,38/0,703` z demonstracji w drafcie nie da się odtworzyć bez podania generatora danych, ziarna i skryptu.

Samo przekazanie `h` do `mcs` i dalej do `dm_matrix` naprawia ten konkretny błąd tabeli par DM, ale **nie naprawia eliminacji MCS**: `mcs.py:47–68` liczy własną statystykę i bootstrap niezależnie od wyniku DM. Bootstrap blokowy zachowuje zależność w obrębie wylosowanych bloków, lecz domyślny blok `ceil(T^(1/3))` (`:43`) nie uwzględnia horyzontu; dla `h=30` może być krótszy niż zakres nakładania. Co ważniejsze, kod wybiera najgorszy model na tej samej próbie i bootstrapuje tylko jego różnicę względem próbnego „najlepszego” (`:50–63`), zamiast rozkładu maksimum statystyk dla aktywnego zbioru. Nie ma więc podstaw, by nazywać ten wynik pełnym MCS Hansen–Lunde–Nason z kontrolą poziomu. Potrzebna jest walidacja na procesach z zależnością i remisami oraz poprawna procedura MCS; dla par DM należy użyć uzasadnionego estymatora długookresowej wariancji (np. HAC/Newey–West z jawnym pasmem i analizą wrażliwości), nie zakładać automatycznej poprawności przez samo `h`.

## D3 — potwierdzam niespójność, rozszerzam ją o XGBoost

Porównałem każdą komórkę tabel h=1/7/30 w `results/economic.md:9–40` z `metrics.economic` w `results/baselines.json`, po zaokrągleniu zgodnym z tabelą. Wszystkie Sharpe 10/20 bps i MaxDD 20 bps się zgadzają; również cały wiersz h=1 jest zgodny. W h=7 i h=30 błędnych jest **sześć**, a nie cztery, komórek turnover:

| Horyzont | Model | `economic.md` | `baselines.json` po zaokrągleniu |
|---|---|---:|---:|
| 7 | har_rv | 16,62 | 5,33 |
| 7 | garch | 16,62 | 14,95 |
| 7 | xgboost | 16,62 | 28,36 |
| 30 | har_rv | 16,62 | 1,98 |
| 30 | garch | 16,62 | 9,33 |
| 30 | xgboost | 16,62 | 3,32 |

`16,62` odpowiada po zaokrągleniu turnover GARCH dla h=1 (`16,619263...`). Jest to zgodne z podejrzeniem kopiowania, lecz draft nie ma dowodu, **kto** i **jak** przepisał te liczby. `experiments/run_economic.py:99–103` generuje turnover z danych, więc po usunięciu błędu źródłowego tabelę trzeba wygenerować ponownie i automatycznie porównać z JSON. Kolumna hit rate pozostaje `n/a`, bo w committed `baselines.json` ma wartości `null`; to dodatkowa luka w spójności wyników, bez wpływu na powyższe porównanie Sharpe/MaxDD.

## D4 — potwierdzam brak odtwarzalnego MCS z samych committed wyników; poprawiam ścieżkę

Za ignorowanie predykcji odpowiada `results/preds/.gitignore:1` (`*.parquet`). `data/.gitignore` dotyczy tylko drzewa `data/`, więc przypisanie mu reguły `results/preds/*.parquet` w drafcie jest błędne. `git ls-files results/preds` pokazuje jedynie `.gitignore`; w bieżącym katalogu nie ma plików predykcji. `experiments/run_baselines.py:75–76` zapisuje ramki `{model}_h{h}.parquet`. `experiments/run_economic.py:50–55, 152–175` czyta je i liczy MCS tylko w gałęzi z dostępnymi ramkami, a bez nich korzysta z `baselines.json`; `results/economic.md:18,29,40` stwierdza „MCS not computed” dla wszystkich trzech horyzontów. W committed `results/` nie ma tabeli MCS ani `economic.json`.

Wniosek praktyczny draftu jest trafny: świeży klon **nie odtworzy MCS z dostarczonych artefaktów**. Sformułowanie „niewykonalna sekwencja” jest zbyt mocne, bo można uruchomić `run_baselines` i potem `run_economic`, choć wymaga to pełnego przeliczenia. Potrzebny jest trwały artefakt parowanych strat lub predykcji, ewentualnie wersjonowany wynik MCS z metadanymi pochodzenia i procedurą odtworzenia. Dodatkowo `run_economic.py:157–169` uruchamia `mcs` przy dowolnej niepustej liczbie ramek; przy jednej ramce `mcs.py:31–40` zwróci trywialny „confidence_set”, więc skrypt powinien wymagać co najmniej dwóch wyrównanych modeli i jawnie odnotować skład panelu.

## D5 — potwierdzam błędy raportu Quarto, dodaję dwa pominięte szczegóły

1. `docs/_quarto/index.qmd:121,129` czyta `results/tsfm_full_chronos/metrics.json` i analogiczny plik TimesFM; oba nie istnieją. Dostępne są `tsfm.json` z obu katalogów. Ścieżki w komórkach Python nie uwzględniają też jednoznacznie katalogu wykonania projektu Quarto (`docs/_quarto/`), więc samo zastąpienie nazwy pliku może nie wystarczyć. Ponadto `index.qmd:137` czyta nieistniejący `results/full_seed42/metrics.json`; w tym katalogu są `neural.json` i `neural.md`. Przy wykonaniu tych komórek render zatrzyma się na błędzie odczytu.
2. `index.qmd:5` zawiera dosłowny napis `` `r Sys.Date()` `` w polu `date`. Dokument używa komórek Python (`:118,126,134`) i nie deklaruje silnika R. Zatem nie ma podstaw oczekiwać rozwinięcia daty przez R; trzeba użyć mechanizmu daty obsługiwanego przez Quarto albo wpisać datę jawnie.
3. Tabela `index.qmd:140–145` podaje skill Chronos `−5,19 / −3,55 / −7,10%`. W `results/tsfm_full_chronos/tsfm.json`, `horizons.{1,7,30}.metrics.rmse_skill_vs_rw`, są odpowiednio `−7,10 / −4,93 / −3,55%`. `−5,19%` to średnia z `aggregate.mean_rmse_skill_vs_rw`, nie wartość dla h=1. TimesFM ma horyzontalnie DirAcc `48,3 / 43,0 / 44,7%` oraz skill `−3,11 / −8,84 / −15,08%`; tabela podaje zaokrąglone średnie `45,3%` i `−9,0%`. Średnie są poprawnymi polami `aggregate`, ale nagłówek skill „h=1/7/30” czyni tę prezentację mylącą. Intencji autora („niezamierzone”) nie da się ustalić z plików.
4. `index.qmd:37` twierdzi, że modele SGH przetestowano na danych NBP/Yahoo, a `:33` przedstawia raport jako wyniki testów na 34 seriach SGH. Nie ma żadnego `results/sgh_*.json`; przykładowe `results/SGH_NBP_TRD32.md` i `SGH_BITCOIN_TRD58.md` zapowiadają dopiero zapis liczb po uruchomieniu eksperymentu. To nie uprawnia do sformułowania o przetestowaniu na tych danych. `index.qmd:251` mówi o 34 plikach `SGH_*.md`, podczas gdy w repo są **29**; również draftowe „32 pliki” jest niezgodne z tym licznikiem. `index.qmd:252` podaje „8 passed”, choć sam plik `tests/test_evaluation_harness.py` zawiera więcej testów i test CQR z D1 obecnie pada. Liczby i statusy trzeba generować z aktualnych artefaktów, z rozróżnieniem specyfikacji od wykonanych wyników.
