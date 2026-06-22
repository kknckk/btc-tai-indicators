import os
import argparse
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import credentials

def main(project_id: str):
    print(f"Inicjalizacja klienta BigQuery dla projektu: {project_id}")
    
    token = os.environ.get("GCLOUD_TOKEN")
    if token:
        creds = credentials.Credentials(token)
        client = bigquery.Client(project=project_id, credentials=creds)
    else:
        client = bigquery.Client(project=project_id)
    
    dataset_id = "btc_indicators"
    table_id = "coinmetrics_daily"
    dataset_ref = f"{project_id}.{dataset_id}"
    table_ref = f"{dataset_ref}.{table_id}"
    
    # Create dataset if it doesn't exist
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_ref} istnieje.")
    except NotFound:
        print(f"Tworzenie datasetu {dataset_ref}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Może być zmienione na EU w razie potrzeby
        client.create_dataset(dataset, timeout=30)
        print("Dataset utworzony pomyślnie.")

    # Process all CSVs
    csv_dir = os.path.join(os.path.dirname(__file__), "csv")
    if not os.path.exists(csv_dir):
        print(f"Katalog {csv_dir} nie istnieje. Przerywam.")
        return

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        print("Brak plików CSV do załadowania.")
        return

    print(f"Znaleziono {len(csv_files)} plików CSV. Rozpoczęcie łączenia...")
    
    # Read the first CSV to initialize the merged DataFrame
    first_file = csv_files[0]
    merged_df = pd.read_csv(os.path.join(csv_dir, first_file))
    merged_df['time'] = pd.to_datetime(merged_df['time'])

    # Merge remaining CSVs
    for filename in csv_files[1:]:
        df = pd.read_csv(os.path.join(csv_dir, filename))
        df['time'] = pd.to_datetime(df['time'])
        merged_df = pd.merge(merged_df, df, on="time", how="outer")

    print(f"Połączono wskaźniki. Całkowita liczba wierszy: {len(merged_df)}")
    
    # Sort by time
    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    
    # Upload to BigQuery
    print(f"Przesyłanie danych do {table_ref}...")
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", # Zastępujemy tabelę
    )
    
    job = client.load_table_from_dataframe(
        merged_df, table_ref, job_config=job_config
    )
    job.result()  # Wait for the job to complete
    
    table = client.get_table(table_ref)
    print(f"Pomyślnie załadowano {table.num_rows} wierszy i {len(table.schema)} kolumn do {table_ref}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload BTC indicators to BigQuery")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    args = parser.parse_args()
    main(args.project)
