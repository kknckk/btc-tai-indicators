import os
import csv
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

app = FastAPI(title="BTC Indicators API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "btc-ind")
DATASET_ID = "btc_indicators"
COINMETRICS_TABLE = "coinmetrics_daily"
USE_LOCAL_CSV = os.environ.get("USE_LOCAL_CSV", "false").lower() == "true"

if not USE_LOCAL_CSV:
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Błąd inicjalizacji BigQuery: {e}. Jeśli chcesz użyć plików CSV użyj USE_LOCAL_CSV=true")
        client = None

class DataPoint(BaseModel):
    time: str
    value: float

class IndicatorResponse(BaseModel):
    indicator: str
    data: List[DataPoint]

@app.get("/api/v1/metrics/available")
def get_available_metrics():
    """Zwraca listę dostępnych wskaźników z tabeli lub plików CSV."""
    if USE_LOCAL_CSV:
        csv_dir = os.path.join(os.path.dirname(__file__), "..", "data_ingestion", "csv")
        if not os.path.exists(csv_dir):
            return {"indicators": []}
        indicators = [f.replace(".csv", "") for f in os.listdir(csv_dir) if f.endswith(".csv")]
        return {"indicators": indicators}
    else:
        return {
            "indicators": [
                "AdrActCnt", "AdrBalCnt", "BlkCnt", "BlkIntMean", "BlkSizeMeanByte",
                "CapAct1yrUSD", "CapFutExp10yrUSD", "CapMrktCurUSD", "CapMVRVCur", "CapRealUSD",
                "CDD", "FeeMeanNtv", "FeeMeanUSD", "FeeMedNtv", "FeeRevPct", "FeeTotNtv", "FeeTotUSD",
                "FlowInExNtv", "FlowInExUSD", "FlowOutExNtv", "FlowOutExUSD", "HashRate",
                "IssContPctAnn", "IssContPctDay", "IssTotNtv", "IssTotUSD", "MVRV_Z", "NUPL",
                "PriceBTC", "PriceUSD", "PuellMultiple", "RealizedPrice", "RevUSD", "SOPR",
                "SplyAct180d", "SplyAct1yr", "SplyAct30d", "SplyCur", "SplyExpFut10yr", "TxCnt",
                "TxCntSec", "TxTfrCnt", "TxTfrValNtv", "TxTfrValUSD", "UTXOCnt"
            ]
        }

@app.get("/api/v1/metrics/values", response_model=List[IndicatorResponse])
def get_metrics_values(
    metrics: List[str] = Query(..., description="Lista wskaźników (np. CapRealUSD, SOPR)"),
    start_date: str = Query("2009-01-01", description="Data początkowa RRRR-MM-DD"),
    end_date: str = Query("2030-01-01", description="Data końcowa RRRR-MM-DD")
):
    """Pobiera historyczne dane dla wybranych wskaźników."""
    available = get_available_metrics()["indicators"]
    valid_metrics = [m for m in metrics if m in available]
    
    if not valid_metrics:
        raise HTTPException(status_code=400, detail="Żaden z podanych wskaźników nie jest obsługiwany.")
    
    if USE_LOCAL_CSV:
        csv_dir = os.path.join(os.path.dirname(__file__), "..", "data_ingestion", "csv")
        final_response = []
        for m in valid_metrics:
            filepath = os.path.join(csv_dir, f"{m}.csv")
            data_points = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader) # time, value
                    for row in reader:
                        if len(row) >= 2:
                            time_str, val_str = row[0], row[1]
                            if start_date <= time_str <= end_date:
                                try:
                                    data_points.append(DataPoint(time=time_str, value=float(val_str)))
                                except ValueError:
                                    continue
            final_response.append(IndicatorResponse(indicator=m, data=data_points))
        return final_response
    else:
        # Kod obsługujący BigQuery
        if client is None:
            raise HTTPException(status_code=500, detail="Klient BigQuery nie został zainicjalizowany.")
        select_columns = ", ".join(valid_metrics)
        query = f"""
            SELECT 
                CAST(time AS STRING) as time,
                {select_columns}
            FROM `{PROJECT_ID}.{DATASET_ID}.{COINMETRICS_TABLE}`
            WHERE DATE(time) >= DATE(@start_date) AND DATE(time) <= DATE(@end_date)
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
            raise HTTPException(status_code=500, detail=str(e))

