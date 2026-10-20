#!/usr/bin/env bash
# Entrypoint obrazu treningowego: uruchamia zero-shot TSFM i wgrywa wyniki do GCS.
#
# Zmienne środowiskowe:
#   GCS_OUTPUT_URI  – katalog docelowy, np. gs://moj-bucket/tsfm/ (opcjonalny)
#   PRICE_CSV_GCS   – gs://.../PriceUSD.csv; jeśli podane, pobiera świeżą cenę
#   OUT_DIR         – katalog lokalny na wyniki (domyślnie /tmp/results)
set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/results}"
mkdir -p "$OUT_DIR"

ARGS=(--out-dir "$OUT_DIR")

if [[ -n "${PRICE_CSV_GCS:-}" ]]; then
  echo "[entrypoint] Pobieram cenę z ${PRICE_CSV_GCS}"
  python - "$PRICE_CSV_GCS" <<'PY'
import sys
from google.cloud import storage

uri = sys.argv[1]
bucket_name, blob_name = uri[len("gs://"):].split("/", 1)
storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename("/tmp/PriceUSD.csv")
print("[entrypoint] Zapisano /tmp/PriceUSD.csv")
PY
  ARGS+=(--csv /tmp/PriceUSD.csv)
fi

echo "[entrypoint] python -m experiments.run_tsfm ${ARGS[*]} $*"
set +e
python -m experiments.run_tsfm "${ARGS[@]}" "$@"
STATUS=$?
set -e
echo "[entrypoint] run_tsfm exit=${STATUS}"

if [[ -n "${GCS_OUTPUT_URI:-}" ]]; then
  echo "[entrypoint] Wgrywam wyniki do ${GCS_OUTPUT_URI}"
  python - "$OUT_DIR" "$GCS_OUTPUT_URI" <<'PY'
import os
import sys
from pathlib import Path
from google.cloud import storage

out_dir, uri = Path(sys.argv[1]), sys.argv[2].rstrip("/")
bucket_name, prefix = uri[len("gs://"):].split("/", 1)
bucket = storage.Client().bucket(bucket_name)
uploaded = 0
for path in sorted(out_dir.glob("tsfm*.json")) + sorted(out_dir.glob("tsfm*.md")):
    blob = f"{prefix.rstrip('/')}/{path.name}" if prefix else path.name
    bucket.blob(blob).upload_from_filename(str(path))
    print(f"[entrypoint] gs://{bucket_name}/{blob}")
    uploaded += 1
print(f"[entrypoint] Wgrano {uploaded} plików")
if uploaded == 0:
    raise SystemExit("Brak plików wynikowych do wgrania")
PY
else
  echo "[entrypoint] GCS_OUTPUT_URI nie ustawione – wyniki zostają lokalnie w ${OUT_DIR}"
fi

exit "$STATUS"
