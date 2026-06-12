import pytest
from google.cloud import bigquery
import os

PROJECT_ID = "btc-ind"

@pytest.fixture
def bq_client():
    return bigquery.Client(project=PROJECT_ID)

def test_active_supply(bq_client):
    sql_path = os.path.join(os.path.dirname(__file__), "..", "sql", "active_supply.sql")
    with open(sql_path, "r") as f:
        query = f.read()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", "2024-05-01")
        ]
    )
    
    query_job = bq_client.query(query, job_config=job_config)
    results = list(query_job.result())
    
    assert len(results) == 1
    row = results[0]
    
    sply_1yr = row["SplyAct1yr"]
    print(f"Otrzymane SplyAct1yr dla 2024-05-01: {sply_1yr} BTC")
    
    assert sply_1yr is not None
    # Bitcoina aktywna podaż roczna w hossach zazwyczaj oscyluje w 4M - 8M BTC
    assert sply_1yr > 1000000, "Active supply 1yr jest nienaturalnie małe (poniżej 1M)"
    assert sply_1yr < 21000000, "Active supply 1yr nie może przekraczać 21 milionów"

def test_cdd(bq_client):
    sql_path = os.path.join(os.path.dirname(__file__), "..", "sql", "cdd.sql")
    with open(sql_path, "r") as f:
        query = f.read()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", "2024-05-01")
        ]
    )
    
    query_job = bq_client.query(query, job_config=job_config)
    results = list(query_job.result())
    
    if len(results) > 0:
        row = results[0]
        cdd = row["cdd"]
        print(f"Otrzymane CDD dla 2024-05-01: {cdd}")
        assert cdd > 0, "Coin Days Destroyed powinno być dodatnie"
