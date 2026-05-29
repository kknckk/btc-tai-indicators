import os
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

app = FastAPI(title="BTC Indicators API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # w produkcji ustalić domenę dashboardu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "twoj-projekt-gcp")
DATASET_ID = "btc_indicators"
COINMETRICS_TABLE = "coinmetrics_daily"

client = bigquery.Client(project=PROJECT_ID)

class DataPoint(BaseModel):
    time: str
    value: float

class IndicatorResponse(BaseModel):
    indicator: str
    data: List[DataPoint]

@app.get("/api/v1/metrics/available")
def get_available_metrics():
    """Zwraca listę dostępnych wskaźników z tabeli."""
    return {
        "indicators": [
            "CapRealUSD", "SplyAct30d", "SplyAct180d", "SplyAct1yr",
            "FeeTotNtv", "FeeTotUSD", "CapAct1yrUSD", "AdrActCnt",
            "AdrBalCnt", "BlkCnt", "BlkSizeMeanByte", "UTXOCnt",
            "CapMrktCurUSD", "DiffMean", "HashRate", "SplyCur",
            "TxCnt", "TxTfrValNtv", "TxTfrValUSD", "TxTfrValAdjNtv",
            "TxTfrValAdjUSD", "MVRV_Z", "SOPR", "NUPL", "RealizedPrice",
            "PuellMultiple", "FeeMedNtv"
        ]
    }

@app.get("/api/v1/metrics/values", response_model=List[IndicatorResponse])
def get_metrics_values(
    metrics: List[str] = Query(..., description="Lista wskaźników (np. CapRealUSD, SOPR)"),
    start_date: str = Query("2009-01-01", description="Data początkowa RRRR-MM-DD"),
    end_date: str = Query("2030-01-01", description="Data końcowa RRRR-MM-DD")
):
    """Pobiera historyczne dane dla wybranych wskaźników."""
    # Walidacja wejścia zapobiegająca SQL Injection w nazwach kolumn
    available = get_available_metrics()["indicators"]
    valid_metrics = []
    for m in metrics:
        if m in available:
            valid_metrics.append(m)
    
    if not valid_metrics:
        raise HTTPException(status_code=400, detail="Żaden z podanych wskaźników nie jest obsługiwany.")
    
    select_columns = ", ".join(valid_metrics)
    query = f"""
        SELECT 
            CAST(time AS STRING) as time,
            {select_columns}
        FROM `{PROJECT_ID}.{DATASET_ID}.{COINMETRICS_TABLE}`
        WHERE time >= DATE(@start_date) AND time <= DATE(@end_date)
        ORDER BY time ASC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
            bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        
        # Przekształcenie wyników BQ do formatu JSON przyjaznego dla wykresów
        response_data = {metric: [] for metric in valid_metrics}
        
        for row in results:
            for metric in valid_metrics:
                val = row[metric]
                if val is not None:
                    response_data[metric].append(DataPoint(time=row["time"], value=val))
        
        final_response = [
            IndicatorResponse(indicator=m, data=response_data[m]) 
            for m in valid_metrics
        ]
        return final_response
        
    except GoogleAPIError as e:
        # Pamiętaj, aby na produkcji nie zwracać szczegółów błędów BQ
        raise HTTPException(status_code=500, detail=str(e))
