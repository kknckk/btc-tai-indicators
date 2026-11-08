# PROMPT dla recenzenta (sol-high / gpt-6.1-sol)

Jesteś niezależnym recenzentem metodologicznym. Zadanie zleca Krzysztof Kanciak (BTC-TAI / PKO BP).
Odpowiadasz po polsku.

## Wejścia (ścieżki absolutne, repo: /opt/data/grok_push/btc-tai-indicators, commit 2d915c8)

1. **Draft do recenzji**: `docs/review_2d915c8/review.qmd` — moje review prac po ostatnim commicie.
2. Kod pod recenzją: `evaluation/mcs.py`, `evaluation/dm_test.py`, `evaluation/walk_forward.py`,
   `evaluation/calibration.py`, `evaluation/targets.py`, `experiments/run_economic.py`.
3. Dane dowodowe: `results/baselines.json`, `results/economic.md`, `results/calibration.md`,
   `results/tsfm_full_chronos/tsfm.json`, `results/tsfm_full_timesfm/tsfm.json`.
4. Kontekst: `reports/pko-feng/REVIEW_SOL_zad1_4.md` (wcześniejsza recenzja raportu SGH,
   wykonana przez inny model — nie powielaj jej, oceniaj MOJE ustalenia z draftu).

## Co masz zweryfikować (z nazwy)

D1. Draft twierdzi, że test `test_calibrate_frame_cqr_on_quantile_synthetic` pada deterministycznie
    (pokrycie 1.0000 przy asercji ≤ 0.92), bo test konstruuje kwantyle z realizacji y ("wyrocznia").
    Oceń, czy ta diagnoza jest poprawna — prześledź `calibrate_frame_cqr` / `rolling_cqr` /
    `cqr_scores` i sam test (tests/test_evaluation_harness.py, linia ~110).

D2. Draft twierdzi, że `evaluation/mcs.py` wywołuje `dm_matrix(..., h=1)` niezależnie od horyzontu
    prognozy, co zawyża istotność dla h=7/30. Sprawdź kod i oceń, czy zarzut jest trafiony
    (dla strat z prognoz h-krokowych seria różnic jest MA(h-1); DM z h=1 używa tylko wariancji
    krótkookresowej). Oceń też, czy proponowana poprawka (przekazać h do mcs i dm_matrix) wystarczy,
    czy potrzeba coś więcej (np. HAC Newey-West, block bootstrap już jest w MCS — czy wystarczy
    zależność wewnątrz eliminacji).

D3. Draft twierdzi, że `results/economic.md` ma turnover 16.62 dla har_rv/garch na h=7 i h=30,
    a `results/baselines.json` ma na tych horyzontach 5.33/14.95 i 1.98/9.33. Zweryfikuj liczby
    w obu plikach i oceń, czy Sharpe/MaxDD się zgadzają (draft twierdzi, że tak).

D4. Draft twierdzi, że MCS nie da się policzyć na świeżym klonie, bo `results/preds/*.parquet`
    jest ignorowane przez git, a `run_economic.py` bez nich melduje "MCS not computed". Sprawdź
    `data/.gitignore`, `experiments/run_baselines.py` (czy pisze parquety), `run_economic.py`.

D5. Draft wymienia błędy w `docs/_quarto/index.qmd`: odczyt nieistniejącego `metrics.json`
    (jest `tsfm.json`), literał `` `r Sys.Date()` `` bez silnika R, tabela 3.4 z liczbami skill
    chronos −5,19/−3,55/−7,10% podczas gdy committed `tsfm_full_chronos/tsfm.json` ma
    −7,10/−4,93/−3,55% dla h=1/7/30, oraz zdanie o modelach SGH "przetestowanych" przy zerowej
    liczbie wyników (`results/sgh_*.json` nie istnieją). Zweryfikuj każdy punkt w plikach.

## Dodatkowo oceń

- Czy pominięto jakąś istotną wadę metodologiczną harnessu (walk_forward.py, targets.py,
  backtest.py), np. w definicji benchmarka naivnego, vol-targetingu, purge/embargo, rv_fwd.
- Czy którykolwiek wniosek draftu jest za mocny albo nietrafiony.
- Czy rekomendacje P0/P1/P2 są kompletne i realistyczne czasowo.

## Zasady

- Każdą liczbę sprawdzaj w plikach, nie na zaufaniu. Czytaj kod przed oceną.
- Nie zmieniaj żadnych plików. Nie puszuj. Nie odpalaj ciężkich treningów.
- Wolno Ci uruchamiać lokalny python (/opt/data/venvs/btcai/bin/python) do weryfikacji liczb.
- Raport zapisuj INCREMENTALNIE (po każdej sekcji dopisuj) do:
  `/opt/data/grok_push/btc-tai-indicators/docs/review_2d915c8/REVIEW_SOL_HIGH.md`
- Struktura: werdykt ogólny (2-3 zdania) -> D1..D5 (potwierdzam/obalam + dowód) ->
  pominięte wady -> ocena rekomendacji -> lista koniecznych poprawek w drafcie.
