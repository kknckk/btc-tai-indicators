import os
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta

def calculate_cdd():
    project_id = os.environ.get("GCP_PROJECT_ID", "btc-ind")
    client = bigquery.Client(project=project_id)
    
    target_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    sql_path = os.path.join(os.path.dirname(__file__), 'sql', 'cdd.sql')
    with open(sql_path, 'r') as f:
        query = f.read()
        
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", target_date)
        ]
    )
    
    print(f"Obliczanie CDD w BigQuery dla {target_date}...")
    try:
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        
        records = []
        for row in results:
            records.append({
                "time": str(row["date"]),
                "TxTfrValDayDst": row["cdd"]
            })
            
        if not records:
            print("Brak wyników CDD.")
            return

        df = pd.DataFrame(records)
        os.makedirs("csv", exist_ok=True)
        
        col = "TxTfrValDayDst"
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
    calculate_cdd()
