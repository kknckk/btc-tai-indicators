#!/usr/bin/env bash
# Cloud Scheduler: codzienny retrain (Vertex AI CustomJob) + codzienna inferencja
# (Cloud Run Job). Zero hardkodowanego project ID – wszystko z env.
#
# Użycie:
#   export GCP_PROJECT_ID=moj-projekt
#   export SCHEDULER_SERVICE_ACCOUNT=btc-scheduler@moj-projekt.iam.gserviceaccount.com
#   ./gcp/scheduler.sh
#   DRY_RUN=1 ./gcp/scheduler.sh          # wydrukuj komendy bez wywołań gcloud
#
# Kolejność: 02:00 UTC retrain (Vertex), 03:30 UTC inferencja (Cloud Run Job).
# Scheduler wymaga, by oba joby już istniały (patrz gcp/README.md).
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project}"
REGION="${GCP_REGION:-europe-west4}"
SCHEDULER_LOCATION="${SCHEDULER_LOCATION:-${REGION}}"
TIMEZONE="${SCHEDULER_TIMEZONE:-UTC}"
RETRAIN_SCHEDULE="${RETRAIN_SCHEDULE:-0 2 * * *}"
INFERENCE_SCHEDULE="${INFERENCE_SCHEDULE:-30 3 * * *}"

RETRAIN_SCHEDULER_NAME="${RETRAIN_SCHEDULER_NAME:-btc-tsfm-retrain-daily}"
INFERENCE_SCHEDULER_NAME="${INFERENCE_SCHEDULER_NAME:-btc-tsfm-inference-daily}"
INFERENCE_JOB_NAME="${INFERENCE_JOB_NAME:-btc-tsfm-inference}"

SA_EMAIL="${SCHEDULER_SERVICE_ACCOUNT:-${PROJECT_ID}-scheduler@${PROJECT_ID}.iam.gserviceaccount.com}"

TRAIN_IMAGE_URI="${TRAIN_IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/btc-tai/tsfm-train:latest}"
GCS_OUTPUT_URI="${GCS_OUTPUT_URI:-gs://${PROJECT_ID}-btc-ml/tsfm}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"   # "none" => CPU only
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
TSFM_MODEL="${TSFM_MODEL:-chronos-bolt}"
TSFM_HORIZONS="${TSFM_HORIZONS:-1,7,30}"
TSFM_ORIGINS="${TSFM_ORIGINS:-60}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$PROJECT_ID" == "your-gcp-project" ]]; then
  echo "BLAD: ustaw GCP_PROJECT_ID – placeholder 'your-gcp-project'." >&2
  exit 1
fi

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ %s\n' "$*"
  else
    "$@"
  fi
}

# ---- 1. Inferencja: Cloud Scheduler -> Cloud Run Jobs API ------------------
RUN_JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${INFERENCE_JOB_NAME}:run"

echo "== Scheduler inferencji: ${INFERENCE_SCHEDULER_NAME} (${INFERENCE_SCHEDULE} ${TIMEZONE}) =="
if gcloud scheduler jobs describe "$INFERENCE_SCHEDULER_NAME" --location="$SCHEDULER_LOCATION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  run gcloud scheduler jobs update http "$INFERENCE_SCHEDULER_NAME" \
    --location="$SCHEDULER_LOCATION" \
    --project="$PROJECT_ID" \
    --schedule="$INFERENCE_SCHEDULE" \
    --time-zone="$TIMEZONE" \
    --uri="$RUN_JOB_URI" \
    --http-method=POST \
    --oauth-service-account-email="$SA_EMAIL" \
    --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
else
  run gcloud scheduler jobs create http "$INFERENCE_SCHEDULER_NAME" \
    --location="$SCHEDULER_LOCATION" \
    --project="$PROJECT_ID" \
    --schedule="$INFERENCE_SCHEDULE" \
    --time-zone="$TIMEZONE" \
    --uri="$RUN_JOB_URI" \
    --http-method=POST \
    --oauth-service-account-email="$SA_EMAIL" \
    --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
fi

# ---- 2. Retrain: Cloud Scheduler -> Vertex AI customJobs.create ------------
JOB_BODY="$(mktemp)"
trap 'rm -f "$JOB_BODY"' EXIT

# Cloud Scheduler wysyła ten JSON jako body POST-a do Vertex AI.
{
  echo '{'
  echo '  "displayName": "btc-tsfm-retrain-scheduled",'
  echo '  "jobSpec": {'
  echo '    "workerPoolSpecs": ['
  echo '      {'
  echo '        "machineSpec": {'
  echo "          \"machineType\": \"${MACHINE_TYPE}\""
  if [[ "$ACCELERATOR_TYPE" != "none" && -n "$ACCELERATOR_TYPE" ]]; then
    echo "          ,\"acceleratorType\": \"${ACCELERATOR_TYPE}\""
    echo "          ,\"acceleratorCount\": ${ACCELERATOR_COUNT}"
  fi
  echo '        },'
  echo '        "replicaCount": "1",'
  echo '        "containerSpec": {'
  echo "          \"imageUri\": \"${TRAIN_IMAGE_URI}\","
  echo '          "args": ["--model", "'"${TSFM_MODEL}"'", "--horizons", "'"${TSFM_HORIZONS}"'", "--origins", "'"${TSFM_ORIGINS}"'"],'
  echo '          "env": [{"name": "GCS_OUTPUT_URI", "value": "'"${GCS_OUTPUT_URI}"'"}]'
  echo '        }'
  echo '      }'
  echo '    ],'
  echo '    "baseOutputDirectory": {"outputUriPrefix": "'"${GCS_OUTPUT_URI}"'"}'
  echo '  }'
  echo '}'
} > "$JOB_BODY"

# Walidacja JSON zanim wyślemy go do Schedulera.
python3 -m json.tool "$JOB_BODY" >/dev/null

if [[ "${PRINT_JOB_BODY:-0}" == "1" ]]; then
  cat "$JOB_BODY"
  exit 0
fi

VERTEX_URI="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/customJobs"
echo "== Scheduler retrainu: ${RETRAIN_SCHEDULER_NAME} (${RETRAIN_SCHEDULE} ${TIMEZONE}) =="
if gcloud scheduler jobs describe "$RETRAIN_SCHEDULER_NAME" --location="$SCHEDULER_LOCATION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  run gcloud scheduler jobs update http "$RETRAIN_SCHEDULER_NAME" \
    --location="$SCHEDULER_LOCATION" \
    --project="$PROJECT_ID" \
    --schedule="$RETRAIN_SCHEDULE" \
    --time-zone="$TIMEZONE" \
    --uri="$VERTEX_URI" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body-from-file="$JOB_BODY" \
    --oauth-service-account-email="$SA_EMAIL" \
    --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
else
  run gcloud scheduler jobs create http "$RETRAIN_SCHEDULER_NAME" \
    --location="$SCHEDULER_LOCATION" \
    --project="$PROJECT_ID" \
    --schedule="$RETRAIN_SCHEDULE" \
    --time-zone="$TIMEZONE" \
    --uri="$VERTEX_URI" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body-from-file="$JOB_BODY" \
    --oauth-service-account-email="$SA_EMAIL" \
    --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
fi

cat <<EOF

Gotowe. Podgląd:
  gcloud scheduler jobs list --location=${SCHEDULER_LOCATION} --project=${PROJECT_ID}
Ręczne uruchomienie:
  gcloud scheduler jobs run ${INFERENCE_SCHEDULER_NAME} --location=${SCHEDULER_LOCATION} --project=${PROJECT_ID}
Logi inferencji:
  gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="${INFERENCE_JOB_NAME}"' --limit=50 --project=${PROJECT_ID}
EOF
