import os
from google.cloud import bigquery
from datetime import datetime, timedelta
import requests

def run_daily_jobs(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic."""
    project_id = os.environ.get("GCP_PROJECT_ID", "btc-ind")
    dataset_id = os.environ.get("BQ_DATASET", "btc_indicators")
    table_id = os.environ.get("BQ_TABLE", "daily_metrics")
    
    client = bigquery.Client(project=project_id)
    target_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Uruchamianie daily workera dla daty: {target_date}")
    
    # 1. Pobranie FeeRate z Mempool API
    fee_rate = 0.0
    try:
        response = requests.get("https://mempool.space/api/v1/blocks")
        if response.status_code == 200:
            blocks = response.json()
            median_fees = [b['extras']['medianFee'] for b in blocks if 'extras' in b and 'medianFee' in b['extras']]
            if median_fees:
                fee_rate = sum(median_fees) / len(median_fees)
    except Exception as e:
        print(f"Błąd pobierania FeeRate: {e}")

    # 2. Obliczenie CDD (Coin Days Destroyed)
    cdd_val = 0.0
    cdd_query = f"""
    SELECT
      SUM(DATE_DIFF(DATE(i.block_timestamp), DATE(o.block_timestamp), DAY) * (o.value / 100000000.0)) AS cdd
    FROM
      `bigquery-public-data.crypto_bitcoin.inputs` AS i
    JOIN
      `bigquery-public-data.crypto_bitcoin.outputs` AS o
    ON
      i.spent_transaction_hash = o.transaction_hash
      AND i.spent_output_index = o.index
    WHERE
      DATE(i.block_timestamp) = '{target_date}'
    """
    try:
        results = client.query(cdd_query).result()
        for row in results:
            cdd_val = row["cdd"] or 0.0
    except Exception as e:
        print(f"Błąd BQ dla CDD: {e}")

    # 3. Obliczenie Active Supply
    sply30d, sply180d, sply1yr = 0.0, 0.0, 0.0
    active_query = f"""
    WITH unspent_outputs AS (
      SELECT o.value, DATE(o.block_timestamp) AS creation_date
      FROM `bigquery-public-data.crypto_bitcoin.outputs` AS o
      LEFT JOIN `bigquery-public-data.crypto_bitcoin.inputs` AS i
      ON o.transaction_hash = i.spent_transaction_hash
        AND o.index = i.spent_output_index
        AND DATE(i.block_timestamp) <= '{target_date}'
      WHERE DATE(o.block_timestamp) <= '{target_date}'
        AND i.spent_transaction_hash IS NULL
    )
    SELECT
      SUM(CASE WHEN DATE_DIFF('{target_date}', creation_date, DAY) <= 30 THEN value ELSE 0 END) / 100000000.0 AS SplyAct30d,
      SUM(CASE WHEN DATE_DIFF('{target_date}', creation_date, DAY) <= 180 THEN value ELSE 0 END) / 100000000.0 AS SplyAct180d,
      SUM(CASE WHEN DATE_DIFF('{target_date}', creation_date, DAY) <= 365 THEN value ELSE 0 END) / 100000000.0 AS SplyAct1yr
    FROM unspent_outputs
    """
    try:
        results = client.query(active_query).result()
        for row in results:
            sply30d = row["SplyAct30d"] or 0.0
            sply180d = row["SplyAct180d"] or 0.0
            sply1yr = row["SplyAct1yr"] or 0.0
    except Exception as e:
        print(f"Błąd BQ dla Active Supply: {e}")

    # 4. Zapis do BigQuery
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    rows_to_insert = [
        {
            "date": target_date,
            "FeeRateSatVb": fee_rate,
            "TxTfrValDayDst": cdd_val,
            "SplyAct30d": sply30d,
            "SplyAct180d": sply180d,
            "SplyAct1yr": sply1yr
        }
    ]
    try:
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors == []:
            print("Pomyślnie dodano wiersz do BigQuery.")
        else:
            print(f"Błędy przy dodawaniu do BQ: {errors}")
    except Exception as e:
        print(f"Błąd zapisu do BQ: {e}")
