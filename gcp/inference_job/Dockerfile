# Cloud Run Job: artefakt z GCS -> prognoza -> BigQuery.
# Build context = katalog główny repo:
#   gcloud builds submit --tag $IMAGE_URI --file gcp/inference_job/Dockerfile .
# (albo przez gcp/cloudbuild.inference.yaml)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1     PYTHONPATH=/app

WORKDIR /app

COPY gcp/inference_job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Opcjonalnie: realny model TSFM w jobie (obraz rośnie o ~1.5 GB przez torch).
# Domyślnie 0 – job serwuje stub albo artefakt statystyczny bez torch.
ARG INSTALL_TSFM=0
RUN if [ "$INSTALL_TSFM" = "1" ]; then         pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 &&         pip install --no-cache-dir chronos-forecasting>=2.0;     fi

COPY models/ /app/models/
COPY api/csv/PriceUSD.csv /app/api/csv/PriceUSD.csv
COPY gcp/inference_job/main.py /app/main.py

CMD ["python", "/app/main.py"]
