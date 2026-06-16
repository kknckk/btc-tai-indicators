from google.cloud import bigquery
import pandas as pd
import os

os.environ["GCP_PROJECT_ID"] = "btc-ind"
client = bigquery.Client(project="btc-ind")
query = """
SELECT 
    CAST(DATE(timestamp) AS STRING) as time,
    bits
FROM `bigquery-public-data.crypto_bitcoin.blocks`
WHERE DATE(timestamp) >= '2024-05-01' AND DATE(timestamp) <= '2024-05-05'
LIMIT 10
"""
results = client.query(query).result()
for r in results:
    bits_hex = r["bits"]
    print(bits_hex)
