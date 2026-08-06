#!/bin/bash
set -e

PROJECT_ID="btc-ind"
REGION="europe-central2"
IMAGE_NAME="gcr.io/${PROJECT_ID}/btc-pipeline-job"
JOB_NAME="btc-pipeline-job"

echo "1. Budowanie obrazu Dockera na Google Cloud Build..."
gcloud builds submit --tag $IMAGE_NAME ..

echo "2. Tworzenie / Aktualizacja zadania Cloud Run Job..."
# Sprawdzenie czy job istnieje
if gcloud beta run jobs describe $JOB_NAME --region $REGION >/dev/null 2>&1; then
    gcloud beta run jobs update $JOB_NAME \
      --image $IMAGE_NAME \
      --region $REGION
else
    gcloud beta run jobs create $JOB_NAME \
      --image $IMAGE_NAME \
      --region $REGION \
      --memory 4Gi \
      --cpu 2 \
      --task-timeout 3600s \
      --max-retries 1
fi

echo "Gotowe. Możesz uruchomić proces manualnie z konsoli GCP lub wpisując:"
echo "gcloud beta run jobs execute $JOB_NAME --region $REGION"
