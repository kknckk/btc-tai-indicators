from google.cloud import bigquery
import os
os.environ["GCP_PROJECT_ID"] = "btc-ind"
client = bigquery.Client(project="btc-ind")
query = """
SELECT bits FROM `bigquery-public-data.crypto_bitcoin.blocks` LIMIT 1
"""
results = client.query(query).result()
for r in results:
    print(r["bits"])
