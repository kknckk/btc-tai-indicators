#!/bin/bash
set -e

PROJECT_ID="btc-ind"
REGION="europe-central2"
JOB_NAME="btc-pipeline-job"
SCHEDULER_NAME="btc-pipeline-daily-trigger"
SCHEDULE="0 2 * * *" # Codziennie o 02:00 UTC (kiedy dzienne świeczki i on-chain są już gotowe)

echo "Konfigurowanie Cloud Scheduler dla $JOB_NAME na $SCHEDULE"

# Wymagane uprawnienia: Cloud Run Invoker dla Service Account
SERVICE_ACCOUNT="compute@developer.gserviceaccount.com" # Przykładowe - zmień na właściwe

if gcloud scheduler jobs describe $SCHEDULER_NAME --location $REGION >/dev/null 2>&1; then
    gcloud scheduler jobs update http $SCHEDULER_NAME \
      --location $REGION \
      --schedule "$SCHEDULE"
else
    gcloud scheduler jobs create http $SCHEDULER_NAME \
      --location $REGION \
      --schedule "$SCHEDULE" \
      --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
      --http-method=POST \
      --oauth-service-account-email=$SERVICE_ACCOUNT
fi

echo "Sukces! Zadanie przypięte do harmonogramu."
