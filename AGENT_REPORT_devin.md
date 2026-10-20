# AGENT_REPORT_devin — TSFM zero-shot, GCP i API predykcji

**Branch:** `agent/devin` · **Data:** 2026-10-20 · **Środowisko:** Python 3.13.5, venv `/opt/data/venvs/btcai`, torch 2.14.1+cpu, chronos-forecasting 2.3.2, timesfm 3.0.2 (API 2.5), fastapi 0.153, pydantic 2.14.

## Co działa — potwierdzone realnym uruchomieniem

* **`models/tsfm_wrapper.py`** — jednolity kontrakt `fit()`/`predict()` → `p10/p50/p90`.
  Adaptery leniwie importują biblioteki; brak pakietu daje `DependencyMissing`
  z komendą instalacyjną (sprawdzone na Moirai: `pip install uni2ts 'gluonts[torch]'`).
  `StubForecaster` (jawny random walk + empiryczne kwantyle) zawsze ma `stub=true`.
* **`experiments/run_tsfm.py`** — rolling origin na realnym `data_ingestion/csv/PriceUSD.csv`
  (5789 obs., 2010-07-18 → 2026-05-23), 40 originów/horyzont, context 256, CPU.
  `--dry-run` (stub) przechodzi w **1,18 s**; realny Chronos-Bolt w **13,1 s**,
  realny TimesFM 2.5 200M w **111,4 s** (pierwsze uruchomienie z pobraniem wag).
* **API** — `GET /api/v1/models` + `POST /api/v1/predict` wpięte w `api/main.py`;
  `pytest tests/test_predict_api.py -q` → **7 passed in 2.42 s**. Tryb stub jest
  domyślny przy braku artefaktu, z `fallback_reason` i `disclaimer`.
* **GCP** — `bash -n` dla wszystkich skryptów, YAML joba Vertex parsuje się w PyYAML
  (`--print-spec`, `--accelerator none` poprawnie pomija akcelerator), body JSON
  schedulera waliduje `json.tool`, a `gcp/inference_job/main.py` w `DRY_RUN=1`
  policzył 3 wiersze prognozy na realnym CSV.

## Wyniki (realne, nie stub)

| model | h | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | skill vs RW | pokrycie p10–p90 |
|---|---|---|---|---|---|---|---|---|---|
| chronos-bolt | 1 | 1372.32 | 1145.59 | 1.47 | 1.47 | 40.0% | 55.0% | −19.39% | 92.5% |
| chronos-bolt | 7 | 3006.73 | 2587.69 | 3.33 | 3.35 | 47.5% | 55.0% | −4.54% | 100.0% |
| chronos-bolt | 30 | 9314.38 | 7698.63 | 9.83 | 10.58 | 52.5% | 52.5% | −20.52% | 72.5% |
| timesfm 2.5 | 1 | 1309.06 | 1042.28 | 1.34 | 1.33 | 52.5% | 55.0% | −13.88% | 62.5% |
| timesfm 2.5 | 7 | 2831.36 | 2372.04 | 3.05 | 3.03 | 67.5% | 55.0% | **+1.56%** | 62.5% |
| timesfm 2.5 | 30 | 5883.81 | 5175.32 | 6.66 | 6.59 | 95.0% | 52.5% | **+23.87%** | 67.5% |

Wniosek: Chronos-Bolt zero-shot **przegrywa** z random walk na wszystkich horyzontach;
TimesFM wygrywa na 7 i 30 dniach. **Zastrzeżenie:** 40 originów to jeden reżim rynkowy
(kwiecień–maj 2026, trend wzrostowy), więc DirAcc 95% przy h=30 jest niestabilne;
pokrycie przedziałów TimesFM 62–67% jest poniżej nominalnego 80% (pasma za wąskie),
Chronos 72–100% (za szerokie przy h=7). To sygnał do kalibracji, nie do strategii.

## Blokery (dokładnie, bez zaokrągleń)

* **`gcloud` nie jest zainstalowane** na tej maszynie → nie wysłałem CustomJoba,
  nie wdrożyłem Cloud Run Joba, nie utworzyłem Schedulerów ani tabel w BigQuery.
* **Docker CLI jest (26.1.5), ale daemon nie działa**:
  `Cannot connect to the Docker daemon at unix:///var/run/docker.sock` → obrazów
  `Dockerfile.train` i `inference_job/Dockerfile` nie zbudowałem; ich poprawność
  opiera się na przeglądzie i `py_compile`, nie na buildzie.
* **Moirai** — brak `uni2ts` i `gluonts`; adapter jest napisany, ale nieprzetestowany.
* **Pre-existing, nietknięte:** `tests/test_api_endpoints.py` nie kolekcjonuje się
  (`ImportError: get_econometrics_results` w `api/routers/literature.py`) — błąd
  zastany na `master`; pozostałe testy: `12 passed`.

## Checklist GCP dla właściciela

1. `export GCP_PROJECT_ID=...` + włączony **billing**.
2. `gcloud services enable aiplatform run cloudbuild artifactregistry bigquery storage cloudscheduler`.
3. Service accounts `btc-tsfm-runner` i `btc-scheduler` + role z tabeli w `gcp/README.md`.
4. Bucket GCS + repo Artifact Registry (`btc-tai`).
5. `bq query --use_legacy_sql=false < gcp/bigquery_schema.sql`.
6. Build: `gcloud builds submit --config gcp/cloudbuild.yaml --substitutions=_IMAGE_URI=... .`
7. `./gcp/submit_vertex_job.sh` (domyślnie n1-standard-8 + 1×T4).
8. Artefakt: `python gcp/make_artifact.py --metrics results/tsfm.json --out /tmp/model.json`
   i `gcloud storage cp /tmp/model.json gs://BUCKET/tsfm/model.json`.
9. Deploy Cloud Run Joba (`gcloud run jobs deploy ...`, komenda w README) + `./gcp/scheduler.sh`.

**Ręcznie, po stronie człowieka:** project ID, billing, włączenie API, utworzenie SA
i bindingów IAM, bucket/AR/BQ, decyzja o `ALLOW_TSFM=1` (obraz +1,5 GB przez torch)
oraz ewentualny `HF_TOKEN` (limity HuggingFace). Nikt i nic nie zrobi tego za Ciebie —
skrypty tylko wołają `gcloud`, którego tu nie ma.

## Pliki

`models/tsfm_wrapper.py` · `experiments/run_tsfm.py` · `results/tsfm.{json,md}` ·
`results/timesfm/` · `results/dryrun/` · `api/routers/predict.py` · `api/main.py` ·
`api/Dockerfile` · `tests/test_predict_api.py` · `gcp/` (Dockerfile.train, cloudbuild*.yaml,
entrypoint.sh, submit_vertex_job.sh, inference_job/, scheduler.sh, bigquery_schema.sql,
make_artifact.py, README.md).
