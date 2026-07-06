FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrypoint for Vertex AI Custom Training Job
ENTRYPOINT ["python3", "run_ml_dl.py"]
