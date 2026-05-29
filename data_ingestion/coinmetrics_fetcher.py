import pandas as pd
from google.cloud import bigquery
import requests
from io import StringIO
import os

# Konfiguracja - możesz zmienić te wartości przed uruchomieniem
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "twoj-projekt-gcp")
DATASET_ID = "btc_indicators"
TABLE_ID = "coinmetrics_daily"

COINMETRICS_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

# Wskaźniki, które chcemy wyciągnąć z CoinMetrics
TARGET_COLUMNS = [
    "time",
    "CapRealUSD",
    "SplyAct30d",
    "SplyAct180d",
    "SplyAct1yr",
    "FeeTotNtv",
    "FeeTotUSD",
    "CapAct1yrUSD",
    "AdrActCnt",
    "AdrBalCnt",
    "BlkCnt",
    "BlkSizeMeanByte",
    "UTXOCnt",
    "CapMrktCurUSD",
    "DiffMean",
    "HashRate",
    "SplyCur",
    "TxCnt",
    "TxTfrValNtv",
    "TxTfrValUSD",
    "TxTfrValAdjNtv",
    "TxTfrValAdjUSD"
]

def fetch_and_upload():
    print("Pobieranie btc.csv z CoinMetrics...")
    response = requests.get(COINMETRICS_URL)
    response.raise_for_status()
    
    print("Parsowanie CSV...")
    df = pd.read_csv(StringIO(response.text))
    
    # Filtrowanie tylko dostępnych interesujących nas kolumn
    available_columns = [col for col in TARGET_COLUMNS if col in df.columns]
    df = df[available_columns].copy()
    
    # Rzutowanie czasu na datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Utworzenie klienta BQ
    client = bigquery.Client(project=PROJECT_ID)
    
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    csv_filename = "coinmetrics_history.csv"
    print(f"Zapisywanie wyfiltrowanych danych do lokalnego pliku {csv_filename}...")
    df.to_csv(csv_filename, index=False)
    print("Zapisano CSV lokalnie. Gotowe do analizy lub eksportu gdzie indziej.")

    # Konfiguracja joba (nadpisywanie lub dopisywanie - tutaj nadpisujemy, bo to pełna historia)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    
    print(f"Wgrywanie danych do {table_ref} w BigQuery...")
    job = client.load_table_from_dataframe(
        df, table_ref, job_config=job_config
    )
    
    job.result()  # Czekanie na zakończenie
    print(f"Pomyślnie załadowano {job.output_rows} wierszy do {table_ref}.")

if __name__ == "__main__":
    fetch_and_upload()
