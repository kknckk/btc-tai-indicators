#!/usr/bin/env bash
# Wysyła Vertex AI CustomJob (zero-shot TSFM) na podstawie obrazu z Artifact Registry.
#
# Zero hardkodowanych project ID: wszystko z env / flag, z placeholderem.
#
# Przykłady
# ---------
#   export GCP_PROJECT_ID=moj-projekt GCS_BUCKET=moj-projekt-btc-ml
#   ./gcp/submit_vertex_job.sh
#   ./gcp/submit_vertex_job.sh --machine-type n1-standard-4 --accelerator none
#   ./gcp/submit_vertex_job.sh --print-spec          # tylko podejrzyj YAML
#
# Wymagania: gcloud CLI zalogowane, włączone API: aiplatform, cloudbuild,
# artifactregistry, storage. Szczegóły i role IAM: gcp/README.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project}"
REGION="${GCP_REGION:-europe-west4}"
AR_REPO="${ARTIFACT_REGISTRY_REPO:-btc-tai}"
IMAGE_NAME="${IMAGE_NAME:-tsfm-train}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}}"
BUCKET="${GCS_BUCKET:-${PROJECT_ID}-btc-ml}"
OUTPUT_URI="${GCS_OUTPUT_URI:-gs://${BUCKET}/tsfm}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"   # "none" => CPU only
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
DISPLAY_NAME="${DISPLAY_NAME:-btc-tsfm-$(date -u +%Y%m%d-%H%M%S)}"
SERVICE_ACCOUNT="${VERTEX_SERVICE_ACCOUNT:-}"
MODEL="${TSFM_MODEL:-chronos-bolt}"
HORIZONS="${TSFM_HORIZONS:-1,7,30}"
ORIGINS="${TSFM_ORIGINS:-60}"
CONTEXT_LEN="${TSFM_CONTEXT_LEN:-256}"
PRICE_CSV_GCS="${PRICE_CSV_GCS:-}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
PRINT_SPEC=0

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Flagi:
  --machine-type TYPE     np. n1-standard-8 (domyslnie), n1-standard-4, a2-highgpu-1g
  --accelerator TYPE      NVIDIA_TESLA_T4 (domyslnie), NVIDIA_L4, NVIDIA_A100_40GB, none
  --accelerator-count N   domyslnie 1
  --output-uri URI        GCS output, np. gs://bucket/tsfm
  --model NAME            chronos-bolt | chronos | timesfm | moirai | stub
  --horizons LIST         np. 1,7,30
  --origins N             liczba originow rolling
  --no-build              nie buduj obrazu, uzyj istniejacego IMAGE_URI
  --print-spec            wypisz YAML joba i wyjdz (bez gcloud)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine-type)      MACHINE_TYPE="$2"; shift 2 ;;
    --accelerator)       ACCELERATOR_TYPE="$2"; shift 2 ;;
    --accelerator-count) ACCELERATOR_COUNT="$2"; shift 2 ;;
    --output-uri)        OUTPUT_URI="$2"; shift 2 ;;
    --model)             MODEL="$2"; shift 2 ;;
    --horizons)          HORIZONS="$2"; shift 2 ;;
    --origins)           ORIGINS="$2"; shift 2 ;;
    --no-build)          BUILD_IMAGE=0; shift ;;
    --print-spec)        PRINT_SPEC=1; shift ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "Nieznana flaga: $1" >&2; usage; exit 64 ;;
  esac
done

if [[ "$PROJECT_ID" == "your-gcp-project" && "$PRINT_SPEC" -eq 0 ]]; then
  echo "BLAD: ustaw GCP_PROJECT_ID (albo wyeksportuj projekt) – placeholder 'your-gcp-project'." >&2
  echo "       Np.: export GCP_PROJECT_ID=moj-projekt" >&2
  exit 1
fi

if [[ "$PRINT_SPEC" -eq 0 ]] && ! command -v gcloud >/dev/null 2>&1; then
  echo "BLAD: brak gcloud CLI w PATH. Zainstaluj Google Cloud SDK." >&2
  exit 127
fi

if [[ "$BUILD_IMAGE" -eq 1 && "$PRINT_SPEC" -eq 0 ]]; then
  echo "== Budowa obrazu ${IMAGE_URI} (Cloud Build, kontekst = repo root) =="
  gcloud builds submit \
    --project="$PROJECT_ID" \
    --config=gcp/cloudbuild.yaml \
    --substitutions="_IMAGE_URI=${IMAGE_URI}" \
    .
fi

SPEC_FILE="$(mktemp)"
trap 'rm -f "$SPEC_FILE"' EXIT

{
  echo "displayName: ${DISPLAY_NAME}"
  echo "jobSpec:"
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    echo "  serviceAccount: ${SERVICE_ACCOUNT}"
  fi
  echo "  workerPoolSpecs:"
  echo "  - machineSpec:"
  echo "      machineType: ${MACHINE_TYPE}"
  if [[ "$ACCELERATOR_TYPE" != "none" && -n "$ACCELERATOR_TYPE" ]]; then
    echo "      acceleratorType: ${ACCELERATOR_TYPE}"
    echo "      acceleratorCount: ${ACCELERATOR_COUNT}"
  fi
  echo "    replicaCount: \"1\""
  echo "    containerSpec:"
  echo "      imageUri: ${IMAGE_URI}"
  echo "      args: [\"--model\", \"${MODEL}\", \"--horizons\", \"${HORIZONS}\", \"--origins\", \"${ORIGINS}\", \"--context-len\", \"${CONTEXT_LEN}\"]"
  echo "      env:"
  echo "        - name: GCS_OUTPUT_URI"
  echo "          value: \"${OUTPUT_URI}\""
  if [[ -n "$PRICE_CSV_GCS" ]]; then
    echo "        - name: PRICE_CSV_GCS"
    echo "          value: \"${PRICE_CSV_GCS}\""
  fi
  echo "  baseOutputDirectory:"
  echo "    outputUriPrefix: \"${OUTPUT_URI}\""
} > "$SPEC_FILE"

if [[ "$PRINT_SPEC" -eq 1 ]]; then
  cat "$SPEC_FILE"
  exit 0
fi

echo "== Vertex AI CustomJob: ${DISPLAY_NAME} =="
echo "   region=${REGION} machine=${MACHINE_TYPE} accelerator=${ACCELERATOR_TYPE} (x${ACCELERATOR_COUNT})"
echo "   output=${OUTPUT_URI}"
gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --config="$SPEC_FILE"

cat <<EOF

Gotowe. Monitorowanie:
  gcloud ai custom-jobs list --project=${PROJECT_ID} --region=${REGION} --limit=5
  gcloud ai custom-jobs describe <JOB_ID> --project=${PROJECT_ID} --region=${REGION}
Logi:
  gcloud logging read 'resource.type="ml_job" AND labels.job_id="<JOB_ID>"' --limit=50 --project=${PROJECT_ID}
Wyniki:
  gcloud storage ls ${OUTPUT_URI}
EOF
