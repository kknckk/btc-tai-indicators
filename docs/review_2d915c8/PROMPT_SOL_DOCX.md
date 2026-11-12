# PROMPT dla sol — audyt liczbowy dokumentu review_bank.qmd

Jesteś recenzentem metodologicznym. Zadanie zleca Krzysztof Kanciak.
Odpowiadasz i piszesz wyłącznie po polsku. Nie pushujesz nic, nie zmieniasz plików repo
— jedyny Twój zapis to PLIK WYNIKU (append, inkrementalnie, sekcja po sekcji!).

## Wejście (repo: /opt/data/grok_push/btc-tai-indicators)
- Dokument do audytu: docs/review_2d915c8/review_bank.qmd (raport dla PKO BP;
  recenzja prac SGH zadanie 1.4 + nasze modele + rekomendacje).
- Raport źródłowy SGH: reports/pko-feng/zad1_4_30102026.txt
- Nasze wyniki referencyjne: results/sgh_bitcoin_tier1.json, sgh_bitcoin_tier2.json,
  sgh_nbp_rate_tier1.json (jeśli jest), sgh_nbp_rate_tier2.json — czytaj json przez python.

## Twierdzenia do weryfikacji (każde z osobna: werdykt — potwierdzam / obalam /
koryguję + twój przeliczony dowód; dopisz sekcję do pliku wyniku ZARAZ po każdym D):
- D1 (qmd, rozdział o istotności BTC): przy +1,8% poprawie SVR wobec RW na 30 dni
  i oknie testowym 09.2023–03.2025 (~19 niezależnych obserwacji) próg wykrywalności
  to ±~33% błędu, a do udowodnienia takiej poprawy potrzeba ~530 lat historii BTC.
  Sprawdź założenia: długość okna, nakładanie się okien h=30 (overlapping),
  wariancja błędu; przelicz próg z testu DM dla tej próby samodzielnie.
- D2 (qmd, tabela wyników NBP tier I): wartości skill vs RW h=1/3/12 mies.
  dla ordered logit/probit/Taylor/VAR/BVAR — sprawdź liczby wprost z
  results/sgh_nbp_rate_tier1.json (jeśli brak, użyj dostępnego json i zaznacz).
- D3 (qmd): cztery sieci GPU (BiLSTM, LSTM, TFT, HellFormer) mają skill_vs_rw = 0.0000
  na h=1/7/30 („RW clone") — sprawdź results/sgh_bitcoin_tier1.json i tier2.
- D4 (qmd): SVAR skill ok. −109/−256/−698% i RPP-NLP −33348/−4487/−260% (h=1/3/12 mies.)
  — sprawdź results/sgh_nbp_rate_tier2.json.
- D5 (qmd): SGH pokrył „4 z 11" obszarów listy zlecenia — zweryfikuj listę TRD
  w reports/pko-feng/zad1_4_30102026.txt i sam podaj licznik pokryty/niepokrytych.
- D6 (qmd): „190 kombinacji / 188 pełnych, 2 niepełne = VMD+XGBoost ropa; GARCH 19/20"
  — sprawdź w tekście raportu SGH.
- D7 (qmd/Część III): FX VAR/BVAR +5–6% MAPE na 30 dni jako jedyny obszar powyżej
  granicy wykrywalności — oceń, czy to wniosek uzasadniony próbą (jaka efektywna
  liczba obserwacji? czy overlapping h=30 koryguje?); jeśli nie masz json FX, oceń
  teoretycznie i oznacz jako szacunek.

## Wynik
- Plik: /opt/data/grok_push/btc-tai-indicators/docs/review_2d915c8/SOL_AUDIT_BANK.md
  (append inkrementalnie; nagłówek każdej sekcji: „## Dk — werdykt").
- Dla każdego D dodaj też 2-4 zdania propozycji, jak poprawić brzmienie w qmd
  (jeśli werdykt != potwierdzam).
- Ostatnim wpisem dodaj lista 3-5 brakujących analiz, które bank najbardziej
  chciałby zobaczyć w takim dokumencie.

## Ograniczenia
Nie ruszasz farmy, nie commitujesz, nie wysyłasz nic na zewnątrz. Każdą liczbę,
którą cytujesz, sam przelicz lub oznacz „odczyt z pliku X, linia Y".
