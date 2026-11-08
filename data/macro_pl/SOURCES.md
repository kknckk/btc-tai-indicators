# Źródła danych makro/bankowych (SGH, TRD-29)

Skrypty: `scripts/macro_pl/fetch_*.py` · loader: `experiments/macro_pl.py` · cache: `data/macro_pl/`.

| Seria | Skrypt | Status | Licencja / dostęp | Częstotliwość |
|-------|--------|--------|-------------------|---------------|
| `nbp_rate` | `fetch_nbp_reference_rate.py` | OK (snapshot XML static.nbp.pl) | NBP, dane publiczne; `api.nbp.pl/api/interestrates` zwraca 404 od ~2026 → fetcher bierze `static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml` (fallback: legacy JSON API, potem cache) | ad hoc (RPP) |
| `gus_cpi_yoy`, `gus_pmi` | `fetch_gus_macro.py` | OK z substytutami (Eurostat) | `gus_cpi_yoy` := Eurostat HICP PL CP00 RCH_A (nie CPI GUS — inny koszyk, korelacja ~0,97); `gus_pmi` := DG ECFIN ESI SA dla PL (proxy; PMI S&P Global komercyjny). BDL bez klucza: limit 100 req/12 h; wcześniejsze ID zmiennych były błędne. Fallback: `raw/gus_macro.csv` | miesięczna |
| `bond_pl_*` | `fetch_bonds_pl_yields.py` | do weryfikacji (MF CSV) | Dane publiczne MF; **GPW Benchmark SPW = komercyjny** | dzienna |
| `credit_*`, `deposits_households` | `fetch_nbp_credits_deposits.py` | do weryfikacji (XLS NBP) | Statystyka monetarna NBP / nbportal | miesięczna |
| `tfi_assets` | `fetch_tfi_assets.py` | do weryfikacji (IZFiA) | Raporty IZFiA; **Analizy.pl = płatne, bez automatu** | miesięczna |

## GPW Benchmark (krzywa SPW)

Indeksy rentowności SPW (2Y/5Y/7Y/10Y) są dystrybuowane przez GPW Benchmark. Dostęp do plików dziennych zwykle wymaga **umowy licencyjnej** lub produktu płatnego — loader **nie** udaje posiadania tych plików. W raporcie i `SPEC.yaml` seria obligacji jest mapowana na **krajową krzywą dochodowości MF** (gov.pl), o ile plik CSV jest publicznie dostępny.

## Analizy.pl (aktywa TFI)

Serwis komercyjny; brak publicznego API w zadaniu. Nie pobieramy automatycznie. Alternatywa: pliki statystyczne **IZFiA**.

## Weryfikacja

```bash
python -m experiments.macro_pl --check
```

Ustawia `verified_at` w cache meta i drukuje raport źródeł. Bez sieci użyj `--cache-only` po ręcznym wgraniu plików surowych do `data/macro_pl/raw/`.
