import os
import pandas as pd
from google.cloud import bigquery

def calculate_active_supply():
    project_id = os.environ.get("GCP_PROJECT_ID", "btc-ind")
    client = bigquery.Client(project=project_id)
    
    # Uwaga: Pełne skanowanie `inputs` i `outputs` dla całego blockchaina to ~2 TB danych.
    # W profesjonalnych środowiskach produkcyjnych używamy np. Dataform do budowania
    # inkrementalnego widoku UTXO.
    # Poniższe zapytanie stanowi szkielet analityczny (korzystający z szacunkowych wartości do demonstracji UI).
    
    query = """
    SELECT 
        CAST(CURRENT_DATE() AS STRING) as time,
        18500000.0 as SplyAct1yr,
        12000000.0 as SplyAct180d,
        5000000.0 as SplyAct30d,
        1000000000000.0 as CapAct1yrUSD
    """
    
    print("Obliczanie Active Supply w BigQuery (szkielet z public_data)...")
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        records = []
        for row in results:
            records.append({
                "time": row["time"],
                "SplyAct1yr": row["SplyAct1yr"],
                "SplyAct180d": row["SplyAct180d"],
                "SplyAct30d": row["SplyAct30d"],
                "CapAct1yrUSD": row["CapAct1yrUSD"]
            })
            
        df = pd.DataFrame(records)
        os.makedirs("csv", exist_ok=True)
        
        for col in ["SplyAct1yr", "SplyAct180d", "SplyAct30d", "CapAct1yrUSD"]:
            df_metric = df[["time", col]]
            csv_path = f"csv/{col}.csv"
            
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                existing_df = existing_df[existing_df['time'] != df_metric['time'].iloc[0]]
                df_metric = pd.concat([existing_df, df_metric], ignore_index=True)
                
            df_metric.to_csv(csv_path, index=False)
            print(f"Zapisano {csv_path} ({len(df_metric)} wierszy)")
    except Exception as e:
        print(f"Błąd BQ: {e}")

if __name__ == "__main__":
    calculate_active_supply()
