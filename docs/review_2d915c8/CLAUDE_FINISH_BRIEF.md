# BRIEF dla Claude Code — domknięcie batcha 6 serii SGH (do uruchomienia po BATCH_ALL_EXITED)

Kontekst: repo `/opt/data/grok_push/btc-tai-indicators` (master, push `bash /opt/data/scripts/gitpush.sh <repo>`).
Python: `/opt/data/venvs/btcai/bin/python`. Pełny stan: `docs/review_2d915c8/BATCH_HANDOFF.md`.

Batch (proc_388c5db5154a) liczy 5 configów `*_ret_cpu` sekwencyjnie; config 1 (WIG20 tier1)
działa na kodzie sprzed 12:10 UTC → NIE zapisze `results/preds_sgh/*.parquet` (brak save_pred_frames
w zaimportowanym module). Pozostałe 4 configi odpalą nowy kod i zapiszą ramki.

## Zadania po `BATCH_ALL_EXITED` (w tej kolejności, każdy krok zweryfikuj)
1. Sprawdź RC wszystkich 5 configów w logu batcha oraz `ls results/*ret_cpu*` (json+md per config).
2. Zweryfikuj, czy metryki procentowe nie są nullami (znany warning overflow exp in metrics.py:23).
3. Rerun TYLKO configu 1 dla ramek: 
   `cd /opt/data/grok_push/btc-tai-indicators && export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 && nohup /opt/data/venvs/btcai/bin/python -m experiments.run_sgh_wig20_tier1 --config experiments/configs/sgh_wig20_tier1_ret_cpu.yaml --cache-only > /opt/data/cache/scratch/rerun_wig20_tier1.log 2>&1 &`
   (~5 h; predykcje i metryki wyjdą identyczne — config i kod metryk się nie zmieniły).
   UWAGA: rerun nadpisze results — zrób kopię JSON/MD configu 1 PRZED startem rerunu.
4. Po rerunie: zbiorcze DM/MCS `/opt/data/venvs/btcai/bin/python -m experiments.run_mcs_sgh`
   → results/sgh_mcs.json/.md (sprawdź, czy wszystkie (stem,h) z >=2 modelami się policzyły).
5. Dopisz wyniki liczbowe do sekcji „Rozszerzenie na 6 publicznych serii — protokół" w
   `docs/review_2d915c8/review_bank.qmd` (tabela skill_vs_rw / DM-p per seria×h; NIC nie usuwaj),
   render: `cd docs/review_2d915c8 && /opt/data/venvs/btcai/bin/quarto render review_bank.qmd`.
6. Review commita: `/opt/data/venvs/btcai/bin/python /opt/data/scripts/code_review.py . --range HEAD~1..HEAD`
   (Gemini przez agy — darmowe; fail-closed). Push: `bash /opt/data/scripts/gitpush.sh /opt/data/grok_push/btc-tai-indicators`.
   Po push zweryfikuj SHA przez API GitHub (GITHUB_TOKEN w /opt/data/.env).
7. Zostaw notkę w HANDOFF co zrobiono.

NIE RÓB: nie zmieniać configów, nie ruszać farmy NLFSR, nie wysyłać nic na zewnątrz.

## PILNE (wykryte po configu 1, 12.11 ~01:10 UTC): metryki ret_cpu skopane
Wyniki `results/sgh_wig20_tier1_ret_cpu.json/.md`: RMSE ~2201 (punkty indeksu!)
przy celu log-zwrotowym, skill_vs_rw ≈ -1.8e10, DM-p = 0/None. Hipoteza:
przy `target_kind=log_return` harness/aggregation miesza jednostki — modele
przewidują zwrot, a metryki/`vs_benchmarks` liczone na rekonstruowanym poziomie;
benchmark naive (targets.py:105) zwraca h*mean(logret), RW=0 — niespójne skale.
ZADANIE (priorytet nad checklistą): zdiagnozować gdzie jednostki się mieszają
(runner wig20 tier1: sekcje DM/skill + evaluation/metrics), naprawić, dodać test
regresyjny (mała seria syntetyczna: przy poprawnym setupie skill skonczony, RMSE
w jednostkach zwrotu), dopiero potem rerun configu 1 i reszta checklisty.
Pliki starego przebiegu zachowaj jako *_BROKEN.json/md (nie usuwaj — ślad audytowy).

## AKTUALIZACJA 13.11 ~16:25 UTC
- Config 4/5 (oil tier1) ZAKOŃCZONY BŁĘDEM: RC=1 — ValueError: Input X contains NaN
  (SVR w lstm_svr; źródło: ujemne ceny WTI 2020 → log NaN w cechach).
  WYMAGA: fix (imputer lub dropna w feature_fn dla oil) + re-run configu 4.
- Config 5/5 (wheat tier1) w toku.
- configs 1-3 zakończone RC=0 (json+md w results/, ramki parquet w preds_sgh).

## BATCH ZAKOŃCZONY 14.11 07:18 UTC (exit 0, 67 h)
- RC=0: wig20_tier1, wig20_tier2, gold_tier1, wheat_tier1. RC=1: oil_tier1 (NaN w SVR — fix + rerun).
- Wyniki: results/sgh_{wig20_tier1,wig20_tier2,gold_tier1,wheat_tier1}_ret_cpu.{json,md}
- Ramki: results/preds_sgh/ = 39 parquet (wig20 tier2: 21, gold: 9, wheat: 9; wig20 tier1 bez ramek — rerun).
- PIERWSZE zadanie: fix mieszania jednostek w metrykach (diagnoza w sekcji PILNE powyżej),
  potem przeliczenie wszystkich serii Z RAMEK (preds_sgh) i DM/MCS zbiorczy, rerun configu 1
  i ropy, dopisanie liczb do qmd per config, render, review, push.
