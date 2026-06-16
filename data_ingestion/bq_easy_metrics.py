import os
import pandas as pd
from google.cloud import bigquery

# Pobiera łatwe wskaźniki z BigQuery (zastępując płatne API CoinMetrics)
def fetch_easy_bq_metrics():
    project_id = os.environ.get("GCP_PROJECT_ID", "btc-ind")
    
    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        print(f"Nie można zainicjalizować BQ: {e}")
        return

    # DiffMean
    print("Pobieranie DiffMean (na bazie 'bits')...")
    query_blocks = """
        SELECT 
            CAST(DATE(timestamp) AS STRING) as time,
            MIN(bits) as bits
        FROM `bigquery-public-data.crypto_bitcoin.blocks`
        WHERE DATE(timestamp) <= '2026-06-01'
        GROUP BY time
        ORDER BY time
    """
    try:
        results = client.query(query_blocks).result()
        records = []
        for r in results:
            bits_hex = r["bits"]
            if not bits_hex: continue
            val = int(bits_hex, 16)
            exponent = val >> 24
            coefficient = val & 0xffffff
            target = coefficient * (256 ** (exponent - 3))
            max_target = 0xFFFF * (256 ** (0x1d - 3))
            difficulty = max_target / target if target > 0 else 0
            records.append({"time": r["time"], "DiffMean": difficulty})
            
        df_diff = pd.DataFrame(records)
        df_diff.to_csv("csv/DiffMean.csv", index=False)
        print("Zapisano DiffMean.csv")
    except Exception as e:
        print(f"Błąd DiffMean: {e}")

    # TxMeanByte, FeeMedNtv
    print("Pobieranie TxMeanByte i FeeMedNtv (zajmie chwilę)...")
    query_tx = """
        SELECT
            CAST(DATE(block_timestamp) AS STRING) as time,
            AVG(size) as TxMeanByte,
            APPROX_QUANTILES(fee, 100)[OFFSET(50)] as FeeMedNtv
        FROM `bigquery-public-data.crypto_bitcoin.transactions`
        WHERE DATE(block_timestamp) <= '2026-06-01'
        GROUP BY time
        ORDER BY time
    """
    
    try:
        results = client.query(query_tx).result()
        records = []
        for r in results:
            records.append({
                "time": r["time"], 
                "TxMeanByte": r["TxMeanByte"],
                "FeeMedNtv": r["FeeMedNtv"]
            })
        df_tx = pd.DataFrame(records)
        df_tx[["time", "TxMeanByte"]].to_csv("csv/TxMeanByte.csv", index=False)
        
        # FeeMedUSD = FeeMedNtv * PriceUSD
        if os.path.exists("csv/PriceUSD.csv"):
            df_price = pd.read_csv("csv/PriceUSD.csv")
            df_fee = df_tx[["time", "FeeMedNtv"]].merge(df_price, on="time", how="inner")
            # Fee in BigQuery is in Satoshis usually, or BTC? Actually BQ fee is in Satoshis.
            # Konwersja z Decimal na float
            df_fee["FeeMedNtv"] = df_fee["FeeMedNtv"].astype(float)
            df_fee["FeeMedUSD"] = (df_fee["FeeMedNtv"] / 100_000_000) * df_fee["PriceUSD"]
            df_fee[["time", "FeeMedUSD"]].to_csv("csv/FeeMedUSD.csv", index=False)
            print("Zapisano TxMeanByte.csv oraz FeeMedUSD.csv")
        else:
            print("Zapisano TxMeanByte.csv (FeeMedUSD pominięte brak PriceUSD.csv)")
            
    except Exception as e:
        print(f"Błąd BQ tx: {e}")

if __name__ == "__main__":
    os.makedirs("csv", exist_ok=True)
    fetch_easy_bq_metrics()
