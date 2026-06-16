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
    "TxTfrValAdjUSD",
    # Nowe wskaźniki (Grupa 1, 2, 3)
    "IssContNtv",        # Grupa 1
    "TxMeanByte",        # Grupa 1
    "PriceUSD",          # Grupa 2
    "FeeMedUSD",         # Grupa 2
    "TxTfrValMeanNtv",   # Grupa 2
    "TxTfrValMeanUSD",   # Grupa 2
    "RevAllTimeUSD",     # Grupa 2
    "TxTfrValDayDst",    # Grupa 3
    "VelCur1yr",         # Grupa 3
    "NVTAdj",            # Grupa 3
    "NVTAdj90"           # Grupa 3
]

def fetch_and_upload():
    print("Pobieranie btc.csv z CoinMetrics...")
    response = requests.get(COINMETRICS_URL)
    response.raise_for_status()
    
    print("Parsowanie CSV...")
    df = pd.read_csv(StringIO(response.text))
    
    # Filtrowanie tylko dostępnych interesujących nas kolumn
    available_columns = [col for col in TARGET_COLUMNS if col in df.columns]
    missing_columns = set(TARGET_COLUMNS) - set(available_columns)
    if missing_columns:
        print(f"Ostrzeżenie: Brakujące kolumny w CoinMetrics: {missing_columns}")

    df = df[available_columns].copy()
    
    # Rzutowanie czasu na datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Odcięcie danych na 2026-06-01 dla CSV
    df_csv_export = df[df['time'] <= pd.to_datetime("2026-06-01")].copy()

    # Tworzymy folder csv/ w katalogu data_ingestion
    csv_dir = os.path.join(os.path.dirname(__file__), "csv")
    os.makedirs(csv_dir, exist_ok=True)

    print("Generowanie oddzielnych plików CSV dla każdego wskaźnika...")
    for col in available_columns:
        if col == "time":
            continue
        
        # Wyciągamy kolumnę 'time' i wskaźnik, usuwamy wiersze gdzie wskaźnik jest NaN
        metric_df = df_csv_export[['time', col]].dropna().copy()
        # Formatowanie daty na YYYY-MM-DD (FastAPI oczekuje stringa jako klucza)
        metric_df['time'] = metric_df['time'].dt.strftime('%Y-%m-%d')
        
        filepath = os.path.join(csv_dir, f"{col}.csv")
        metric_df.to_csv(filepath, index=False)
    
    print(f"Wygenerowano plików CSV: {len(available_columns) - 1} w folderze {csv_dir}")

    csv_filename = os.path.join(os.path.dirname(__file__), "coinmetrics_history.csv")
    print(f"Zapisywanie całego wyfiltrowanych danych do lokalnego pliku {csv_filename}...")
    df.to_csv(csv_filename, index=False)
    print("Zapisano całościowe CSV lokalnie. Gotowe do analizy lub eksportu gdzie indziej.")

    # Próba eksportu do BigQuery
    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
        )
        
        print(f"Wgrywanie danych do {table_ref} w BigQuery...")
        job = client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        
        job.result()  # Czekanie na zakończenie
        print(f"Pomyślnie załadowano {job.output_rows} wierszy do {table_ref}.")
    except Exception as e:
        print(f"Pominięto eksport do BigQuery. Błąd: {e}")

if __name__ == "__main__":
    fetch_and_upload()
