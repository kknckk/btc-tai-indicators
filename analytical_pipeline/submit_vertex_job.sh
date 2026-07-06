#!/bin/bash
# Skrypt do uruchomienia treningu ML w chmurze (Google Vertex AI)

PROJECT_ID="TWOJ_PROJEKT_GCP"
REGION="europe-west4"
IMAGE_URI="gcr.io/${PROJECT_ID}/btc-tai-ml-pipeline:latest"

# 1. Zbuduj i wyślij obraz do Container Registry (lub Artifact Registry)
# docker build -t ${IMAGE_URI} .
# docker push ${IMAGE_URI}

# 2. Uruchom Custom Training Job w Vertex AI (wymaga gcloud CLI)
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name="btc-ml-training" \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=${IMAGE_URI} \
  --project=${PROJECT_ID}

echo "Zadanie wysłane do Vertex AI. Przejdź do konsoli GCP, aby monitorować postęp."
