FROM python:3.11-slim

WORKDIR /app

# Build context = katalog główny repo (potrzebny models/tsfm_wrapper.py):
#   gcloud builds submit --tag $IMAGE .
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app
# Wrapper TSFM jest wspólny dla API, eksperymentów i joba inferencji.
COPY models/ /app/models
ENV PYTHONPATH=/app

# Cloud Run expects the app to listen on the port defined by the PORT environment variable.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
