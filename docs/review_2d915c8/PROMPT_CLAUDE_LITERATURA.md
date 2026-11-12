# ZADANIE dla Claude Code — szersza analiza literatury do raportu review_bank

Repo: /opt/data/grok_push/btc-tai-indicators. Dokument: docs/review_2d915c8/review_bank.qmd.
Biblioteka: docs/review_2d915c8/references.bib (format q3, DOI obowiązkowe).
Kontekst merytoryczny: raport recenzuje prace SGH (zadanie 1.4, FENG) — prognozy
BTC, stopy NBP, FX (EUR/PLN, USD/PLN), WIG20, obligacje, surowce; klasy modeli:
GARCH/EGARCH/GJR, HAR-RV, VAR/BVAR/SVAR, ordered logit/probit, XGBoost/RF/GBM,
sieci (LSTM/BiLSTM/TFT/HellFormer), TSFM (PatchTST/Chronos/TimesFM), walk-forward,
testy DM/HLN, MCS, MASE/QLIKE, benchmark RW.

## Co masz zrobić
1. Przeczytaj references.bib i sekcję „Publikacje wybrane do odtworzenia" w qmd.
2. Napisz `docs/review_2d915c8/LITERATURA_ANALIZA.md` (PL) — przegląd uporządkowany
   po klasach zadania (każda klasa = podrozdział): 3–6 pozycji na klasę, dla każdej:
   co dokładnie odtwarzamy (wynik/tabela/figura), po co w protokole, status
   (jest w bib / do dodania). Na końcu sekcja „co w raporcie SGH cytuje się bezkrytycznie
   lub czego brakuje" — wskaż konkretne braki (np. testy istotności, benchmarki zmienności,
   modele krzywej dochodowości, ewaluacja kierunku decyzji RPP).
3. Przygotuj `docs/review_2d915c8/references_dodatki.bib` — nowe wpisy w formacie bib
   jak w references.bib. HARD RULE: każdy wpis musi mieć DOI, który zweryfikowałeś
   HEAD/GET na https://doi.org/<doi> (kod 200/301/302). Niezweryfikowany wpis w osobnej
   sekcji pliku oznaczonej „DO PÓŹNIEJSZEGO POTWIERDZENIA" — nie mieszać.
4. Nie dotykaj istniejących wpisów references.bib ani qmd. Nie commituj (zrobię sam).

Styl: po polsku, rzeczowo. Użyj WebFetch/WebSearch jeśli dostępne; jeśli nie —
curl https://api.crossref.org/works/<doi> dla metadanych.
