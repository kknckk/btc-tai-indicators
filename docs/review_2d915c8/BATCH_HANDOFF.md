# HANDOFF — batch 6 serii SGH (2026-11-11 ~15:00 CET)

## Co działa
Background **proc_388c5db5154a** (na ubu, kontener hermes): pętla po 5 configach
`sgh_{wig20_tier1,wig20_tier2,gold_tier1,oil_tier1,wheat_tier1}_ret_cpu`,
`--cache-only`, OMP/OPENBLAS/MKL=2. Start ~11:40 czasu lokalnego.
Watcher końca: **proc_45119a16cb77** (wypisze `BATCH_ALL_EXITED`).

## Stan postępu w chwili zapisu
- Konfig 1/5 (wig20 tier1): 4 modele (garch, egarch, gjr_garch, event_study)
  × 3 horyzonty (1,7,30) × 379 foldów. Liniłka logów `h=%d fold k/379`
  NIE zawiera nazwy modelu — licznik resetuje się per model (to norma, nie restart).
- Ostatnio widziane: gjr_garch h=1 ~360/379.
- Znany niegroźny warning: `RuntimeWarning overflow in exp` w `evaluation/metrics.py:23`
  (metryka procentowa exp() dla log-zwrotów; runner czyści nieskończoności na null).

## Po zakończeniu batcha (checklist dla dowolnego modelu)
1. `ls results/*ret_cpu*` — spodziewane JSON+MD per config; sprawdź RC_ w logu batcha.
2. Weryfikacja metryk procentowych na null-e (warning powyżej).
3. Łączny MCS/DM na wszystkich seriach z `results/preds_sgh/*.parquet`
   (ramki per-origin zapisuje `evaluation/preds_io.py`; istnieje `experiments/run_mcs_sgh.py`).
4. Dopisać rozdział „6 serii — wyniki" do `docs/review_2d915c8/review_bank.qmd`
   (bez usuwania istniejącej treści), render docx (quarto), review
   `python /opt/data/scripts/code_review.py <repo> --range`, push `bash /opt/data/scripts/gitpush.sh`.
5. W repo są już: configi ret_cpu, runnery 6 serii, `scripts/fetch_public_history.py`
   (WIG20 8448 obs od 1991 w cache `data/multiasset/wig20.csv`).

## Wniosek na przyszłość
Długie batche uruchamiać BEZ heartbeatów (tylko notify on exit) — każdy heartbeat
kosztuje pełną turę modelu.
