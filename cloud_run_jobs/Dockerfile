FROM python:3.12-slim

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Instalacja zależności
RUN pip install --no-cache-dir -r analytical_pipeline/requirements.txt
RUN pip install --no-cache-dir -r data_ingestion/requirements.txt

# Ustawienie zmiennej środowiskowej
ENV PYTHONPATH=/app

# Domyślne wykonanie całego pipeline
CMD ["bash", "analytical_pipeline/run_all.sh"]
