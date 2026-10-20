# GCP: trening (Vertex AI) i inferencja (Cloud Run Jobs) modeli TSFM

Katalog `gcp/` zawiera wszystko, co potrzebne do uruchomienia zero-shot ewaluacji
TSFM na Vertex AI oraz codziennej inferencji do BigQuery. **Żaden plik nie zawiera
hardkodowanego project ID** – wszędzie `GCP_PROJECT_ID` z env i placeholder
`your-gcp-project`.

## Pliki

| Plik | Rola |
|---|---|
| `Dockerfile.train` | obraz Python 3.11-slim: torch CPU + `chronos-forecasting`, uruchamia `experiments/run_tsfm.py` |
| `cloudbuild.yaml` | build obrazu treningowego (Dockerfile poza katalogiem głównym) |
| `submit_vertex_job.sh` | Vertex AI CustomJob: machine type, akcelerator, GCS output; `--print-spec` bez gcloud |
| `entrypoint.sh` | pobiera cenę z GCS (opcjonalnie), uruchamia ewaluację, wgrywa `tsfm.json`/`tsfm.md` do GCS |
| `inference_job/` | Cloud Run Job: artefakt z GCS → prognoza → BigQuery (`predictions`, `model_runs`) |
| `scheduler.sh` | Cloud Scheduler: codzienny retrain (Vertex) + codzienna inferencja (Cloud Run Job) |
| `bigquery_schema.sql` | DDL: `predictions`, `model_runs`, widok `predictions_latest` |
| `make_artifact.py` | dopasowuje parametr `sigma_daily`/`drift_daily` i zapisuje artefakt JSON dla API/joba |

## Przepływ

```
Cloud Scheduler (02:00 UTC) ──► Vertex AI CustomJob (obraz treningowy)
                                      │  experiments/run_tsfm.py
                                      ▼
                              GCS: gs://BUCKET/tsfm/{tsfm.json,tsfm.md}
                                      │  (opcjonalnie make_artifact.py → model.json)
Cloud Scheduler (03:30 UTC) ──► Cloud Run Job (inference_job)
                                      │  artefakt z GCS + cena z BigQuery/CSV
                                      ▼
                              BigQuery: predictions + model_runs
                                      │
                                      ▼
                              FastAPI GET /api/v1/predict (opcjonalnie)
```

## Wymagania wstępne

* `gcloud` CLI (Cloud SDK) zalogowane: `gcloud auth login`, `gcloud config set project $GCP_PROJECT_ID`
* projekt z **włączonym billingiem** (Vertex AI i Cloud Run Jobs nie działają na free tier bez karty)
* role właściciela/edytora przy pierwszym setupie (potem wystarczą role z tabeli IAM)
* opcjonalnie Docker lokalnie (build robi Cloud Build, więc nie jest wymagany)

## Krok po kroku

### 0. Zmienne

```bash
export GCP_PROJECT_ID="twoj-projekt"          # NIE hardkodujemy w repo
export GCP_REGION="europe-west4"
export GCS_BUCKET="${GCP_PROJECT_ID}-btc-ml"
export AR_REPO="btc-tai"
export GCS_OUTPUT_URI="gs://${GCS_BUCKET}/tsfm"
export SCHEDULER_SERVICE_ACCOUNT="btc-scheduler@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
```

### 1. API i rejestry

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  cloudscheduler.googleapis.com \
  --project="$GCP_PROJECT_ID"

gcloud storage buckets create "gs://${GCS_BUCKET}" --location="$GCP_REGION" --project="$GCP_PROJECT_ID"

gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker --location="$GCP_REGION" --project="$GCP_PROJECT_ID"

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"
```

### 2. BigQuery

```bash
bq --project_id="$GCP_PROJECT_ID" query --use_legacy_sql=false < gcp/bigquery_schema.sql
```

Skrypt tworzy dataset `btc_indicators` (o ile nie istnieje), tabele `predictions`
(partycjonowanie po `as_of`, klaster po `model, step`) i `model_runs` oraz widok
`predictions_latest`.

### 3. Service accounts i IAM

```bash
# SA do treningu/inferencji
gcloud iam service-accounts create btc-tsfm-runner \
  --display-name="BTC TSFM runner" --project="$GCP_PROJECT_ID"

# SA dla Cloud Scheduler (wywołuje Vertex + Cloud Run)
gcloud iam service-accounts create btc-scheduler \
  --display-name="BTC TSFM scheduler" --project="$GCP_PROJECT_ID"

RUNNER="btc-tsfm-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
SCHED="btc-scheduler@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

for ROLE in roles/aiplatform.user roles/bigquery.dataEditor roles/storage.objectAdmin roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${RUNNER}" --role="$ROLE" --condition=None
done

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${SCHED}" --role="roles/aiplatform.user" --condition=None
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${SCHED}" --role="roles/run.developer" --condition=None
# Scheduler wysyła requesty jako SCHED, więc potrzebuje też uprawnień do tworzenia jobów:
gcloud iam service-accounts add-iam-policy-binding "$RUNNER" \
  --member="serviceAccount:${SCHED}" --role="roles/iam.serviceAccountUser" \
  --project="$GCP_PROJECT_ID"

# Vertex uruchamia kontener jako RUNNER
gcloud iam service-accounts add-iam-policy-binding "$RUNNER" \
  --member="serviceAccount:${RUNNER}" --role="roles/iam.serviceAccountUser" \
  --project="$GCP_PROJECT_ID"
```

| Rola | Kto | Po co |
|---|---|---|
| `roles/aiplatform.user` | runner, scheduler | tworzenie/uruchamianie CustomJobów |
| `roles/bigquery.dataEditor` | runner | insert do `predictions`, `model_runs` |
| `roles/bigquery.jobUser` | runner | zapytania (DELETE przed insertem) |
| `roles/storage.objectAdmin` | runner | odczyt artefaktu, zapis wyników w GCS |
| `roles/run.developer` | scheduler | `jobs.run` na Cloud Run Job |
| `roles/iam.serviceAccountUser` | scheduler → runner | akt „jako” SA runnera |
| `roles/logging.logWriter` | runner | logi do Cloud Logging |

### 4. Obraz treningowy i Vertex AI

```bash
# build + CustomJob (domyślnie n1-standard-8 + 1×T4)
./gcp/submit_vertex_job.sh

# warianty
./gcp/submit_vertex_job.sh --machine-type n1-standard-4 --accelerator none
./gcp/submit_vertex_job.sh --no-build --accelerator NVIDIA_L4 --output-uri gs://$GCS_BUCKET/tsfm
./gcp/submit_vertex_job.sh --print-spec          # podgląd YAML bez gcloud
```

Wyniki: `gs://$GCS_BUCKET/tsfm/tsfm.json` i `tsfm.md` (metryki RMSE/MAE/MAPE/sMAPE,
DirAcc, skill vs random walk). Job **nie robi fine-tuningu** – to zero-shot
ewaluacja; artefakt „modelu” to plik JSON z konfiguracją i parametrami.

### 5. Artefakt modelu (opcjonalnie, ale zalecane)

```bash
# dopasowanie sigma/drift z realnej historii + metryki z runu
python gcp/make_artifact.py \
  --metrics results/tsfm.json \
  --model stub --context-len 256 \
  --out /tmp/model.json

gcloud storage cp /tmp/model.json "gs://${GCS_BUCKET}/tsfm/model.json"
```

Bez artefaktu API i job inferencji działają w trybie **stub** (jawny random walk
z empirycznymi kwantylami) – nigdy nie udają modelu TSFM.

### 6. Cloud Run Job (inferencja)

```bash
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/tsfm-inference:latest"

gcloud builds submit --project="$GCP_PROJECT_ID" \
  --config=gcp/cloudbuild.inference.yaml \
  --substitutions="_IMAGE_URI=${IMAGE_URI}" .

gcloud run jobs deploy btc-tsfm-inference \
  --project="$GCP_PROJECT_ID" --region="$GCP_REGION" \
  --image="$IMAGE_URI" \
  --service-account="btc-tsfm-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --memory=1Gi --cpu=1 --max-retries=1 --task-timeout=900s \
  --set-env-vars="GCP_PROJECT_ID=${GCP_PROJECT_ID},BQ_DATASET=btc_indicators,GCS_ARTIFACT_URI=gs://${GCS_BUCKET}/tsfm/model.json,ALLOW_TSFM=0,HORIZON=7"

gcloud run jobs execute btc-tsfm-inference --project="$GCP_PROJECT_ID" --region="$GCP_REGION" --wait
```

`ALLOW_TSFM=1` wymaga obrazu zbudowanego z `--build-arg INSTALL_TSFM=1`
(+~1.5 GB przez torch) i artefaktu wskazującego model `chronos-bolt`/`chronos`.

### 7. Scheduler

```bash
export SCHEDULER_SERVICE_ACCOUNT="btc-scheduler@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
./gcp/scheduler.sh
DRY_RUN=1 ./gcp/scheduler.sh     # podgląd komend
```

Tworzy/aktualizuje dwa joby:
* `btc-tsfm-retrain-daily` – 02:00 UTC, POST do Vertex `customJobs`,
* `btc-tsfm-inference-daily` – 03:30 UTC, POST do Cloud Run Jobs API (`:run`).

### 8. Weryfikacja

```bash
gcloud ai custom-jobs list --project="$GCP_PROJECT_ID" --region="$GCP_REGION" --limit=3
gcloud run jobs executions list --job=btc-tsfm-inference --region="$GCP_REGION" --limit=3
bq --project_id="$GCP_PROJECT_ID" query --use_legacy_sql=false \
  'SELECT model, mode, as_of, step, point, p10, p90 FROM `btc_indicators.predictions_latest` ORDER BY as_of DESC LIMIT 20'
```

## Koszty (orientacyjne – sprawdź aktualny cennik GCP)

| Zasób | Stawka orientacyjna | Typowy koszt runu |
|---|---|---|
| Vertex CustomJob `n1-standard-8` | ~0,38 USD/h | ~0,05 USD (8 min) |
| Vertex `1×NVIDIA_TESLA_T4` | ~0,35 USD/h | ~0,05 USD (8 min) |
| Vertex `1×NVIDIA_L4` | ~0,70 USD/h | ~0,10 USD (8 min) |
| Cloud Run Job (1 vCPU, 1 GiB) | ~0,000024 USD/vCPU-s | < 0,01 USD/dzień |
| Cloud Build (build obrazu) | 120 min/dzień gratis | 0 USD przy 1 buildzie |
| Cloud Storage (wyniki + artefakt) | ~0,02 USD/GB/mies. | < 0,01 USD/mies. |
| BigQuery storage + query | 10 GB/mies. gratis; query 1 TB/mies. gratis | 0 USD przy tej skali |
| Cloud Scheduler | 3 joby/mies. gratis | 0 USD |

Miesięczny koszt przy 1 retrainie i 1 inferencji dziennie: **~3–5 USD** (CPU/1×T4).
Największy koszt to GPU i czas trzymania joba – ustaw `--origins 40` i CPU tam,
gdzie nie potrzebujesz GPU. Uwaga: pierwsze uruchomienie pobiera wagi modelu
z HuggingFace (~250 MB dla Chronos-Bolt-small) – w Vertex jest to ruch wychodzący
w ramach maszyny, bez dodatkowych opłat.

## Troubleshooting

| Objaw | Przyczyna / fix |
|---|---|
| `PERMISSION_DENIED: aiplatform.jobs.create` | brak `roles/aiplatform.user` dla SA schedulera |
| Job wisi na „Pulling image” | brak `roles/artifactregistry.reader` dla SA runnera |
| `DependencyMissing: Model 'timesfm' wymaga pakietu 'timesfm'` | obraz bez `timesfm`; użyj `chronos-bolt` albo dobuduj zależność |
| API zwraca `mode="stub"`, `fallback_reason` | brak artefaktu w `PREDICT_ARTIFACT_DIR`/GCS – patrz krok 5 |
| `403` przy zapisie do BigQuery | brak `roles/bigquery.dataEditor` lub zły `BQ_DATASET` |
| Scheduler: `INVALID_ARGUMENT` przy retrainie | sprawdź `PRINT_JOB_BODY=1 ./gcp/scheduler.sh`, czy JSON jest poprawny |

## Co jest sprawdzone lokalnie, a co wymaga GCP

* **Sprawdzone bez GCP:** składnia `bash -n`, `--print-spec` w `submit_vertex_job.sh`
  (YAML parsuje się w PyYAML), `PRINT_JOB_BODY` w `scheduler.sh` (JSON waliduje
  `json.tool`), `DRY_RUN=1` w jobie inferencji na realnym CSV (5 wierszy prognozy),
  testy API (`tests/test_predict_api.py`).
* **Wymaga GCP (nie da się zweryfikować w tym środowisku):** build obrazów w Cloud
  Build, uruchomienie CustomJoba, zapis do BigQuery, Cloud Scheduler, IAM.
  `gcloud` nie jest zainstalowane na maszynie deweloperskiej, na której powstawał kod.
