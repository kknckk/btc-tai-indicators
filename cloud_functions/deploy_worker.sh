#!/bin/bash
# Skrypt do wdrażania Cloud Function z Cloud Scheduler (Pub/Sub trigger)

PROJECT_ID="btc-ind"
REGION="europe-central2"
TOPIC_NAME="trigger-daily-worker"
SCHEDULER_JOB="daily-worker-job"

echo "Deploying Pub/Sub topic..."
gcloud pubsub topics create $TOPIC_NAME --project=$PROJECT_ID || echo "Topic already exists."

echo "Deploying Cloud Function..."
cd daily_worker
gcloud functions deploy daily-btc-worker \
    --runtime python311 \
    --trigger-topic $TOPIC_NAME \
    --entry-point run_daily_jobs \
    --region $REGION \
    --project $PROJECT_ID \
    --memory 512MB \
    --timeout 540s \
    --set-env-vars GCP_PROJECT_ID=$PROJECT_ID,BQ_DATASET=btc_indicators,BQ_TABLE=daily_metrics

cd ..

echo "Deploying Cloud Scheduler Job..."
gcloud scheduler jobs create pubsub $SCHEDULER_JOB \
    --schedule "0 2 * * *" \
    --topic projects/$PROJECT_ID/topics/$TOPIC_NAME \
    --message-body "run" \
    --time-zone "UTC" \
    --location $REGION \
    --project $PROJECT_ID || echo "Scheduler job already exists."

echo "Wdrożenie zakończone!"
