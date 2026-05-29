import pandas as pd
from google.cloud import bigquery
import requests
import time
import os
from datetime import datetime, timedelta

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "twoj-projekt-gcp")
DATASET_ID = "btc_indicators"
TABLE_ID = "mempool_fees_daily"

def get_daily_median_fee(target_date_str: str) -> float:
    """
    Uproszczona funkcja do pobierania mediany fee z mempool.space dla danego dnia.
    Wymagałoby paginacji po blokach z danego dnia. Ze względu na rate limity API,
    w tym skrypcie znajduje się tylko szkielet do wykorzystania w workerze.
    """
    # Placeholder: tutaj należy zaimplementować pobieranie bloków z danego dnia
    # używając endpointu https://mempool.space/api/v1/blocks/{startHeight}
    # i wyciąganie średniej z 'medianFee' dla wszystkich bloków z target_date.
    
    # Dla celów MVP worker może zapytać o ostatnie bloki:
    response = requests.get("https://mempool.space/api/v1/blocks")
    if response.status_code == 200:
        blocks = response.json()
        median_fees = [b['extras']['medianFee'] for b in blocks if 'extras' in b and 'medianFee' in b['extras']]
        if median_fees:
            return sum(median_fees) / len(median_fees)
    return 0.0

def run_daily_worker():
    today = datetime.utcnow().strftime('%Y-%m-%d')
    median_fee = get_daily_median_fee(today)
    
    df = pd.DataFrame([{
        "time": pd.to_datetime(today),
        "FeeMedNtv": median_fee
    }])
    
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    
    print(f"Zapis median fee do {table_ref}...")
    try:
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        print("Zapisano pomyślnie.")
    except Exception as e:
        print(f"Błąd zapisu: {e}")

if __name__ == "__main__":
    run_daily_worker()
