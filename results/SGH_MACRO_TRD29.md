# TRD-29 — dane makro/bankowe (sgh-3)

Repo: `experiments/macro_pl.py`, cache `data/macro_pl/`, dokumentacja `data/macro_pl/SPEC.yaml`, `data/macro_pl/SOURCES.md`.

## Zaimplementowane źródła

| Obszar | Seria(e) | Pobieranie | Cache | Uwagi licencji / dostępu |
|--------|----------|------------|-------|-------------------------|
| Stopa referencyjna NBP | `nbp_rate` | `scripts/macro_pl/fetch_nbp_reference_rate.py` → API `api.nbp.pl/api/interestrates/` | CSV+parquet | API publiczne NBP; status **do weryfikacji** w tej sesji (brak live HTTP w CI) |
| Rentowności 2/5/7/10Y | `bond_pl_2y` … `bond_pl_10y` | `fetch_bonds_pl_yields.py` → CSV z gov.pl (krzywa MF) | per tenor | **GPW Benchmark SPW** — produkt komercyjny/umowa; nie używany. Brak CSV → ręcznie `data/macro_pl/raw/mf_yield_curve.csv` |
| Kredyty + depozyty | `credit_*`, `deposits_households` | `fetch_nbp_credits_deposits.py` → XLS NBP (heurystyka nagłówków) | per seria | Statystyka monetarna nbportal; URL domyślny może wymagać korekty — **do weryfikacji** |
| Aktywa TFI | `tfi_assets` | `fetch_tfi_assets.py` → XLS IZFiA | CSV+parquet | **Analizy.pl** — płatne; tylko ręczny eksport, bez automatu |

## Uruchomienie

```bash
python -m experiments.macro_pl --check
python -m experiments.macro_pl --check --series nbp_rate --refresh
pytest tests/test_macro_pl.py -q
```

## Nie zmyślane serie

Gdy pobranie się nie powiedzie, loader **nie** generuje danych syntetycznych — zwraca błąd lub (przy `--cache-only`) czyta istniejący cache. Testy używają fixture’ów HTTP/plików wyłącznie w `tests/`.

## Kolejne kroki (operacyjne)

1. Uruchomić `--check` w środowisku z siecią i zaktualizować `status: verified` w `SPEC.yaml` dla źródeł, które zwróciły 200 OK.
2. Zweryfikować aktualny URL pliku XLS kredytów/depozytów na nbportal i ewentualnie nadpisać `NBP_CREDITS_XLS_DEFAULT` w `macro_pl.py`.
3. Jeśli PKO BP wymaga krzywej SPW z GPW Benchmark — osobna umowa licencyjna poza tym repozytorium.
