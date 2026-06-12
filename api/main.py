import os
import csv
from typing import List, Optional, Dict
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from cachetools import TTLCache, cached
import numpy as np
from scipy.stats import pearsonr

app = FastAPI(title="BTC Indicators API (Advanced)")

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

# Cache memory for BigQuery responses (12 hours TTL, since data updates daily)
bq_cache = TTLCache(maxsize=100, ttl=12 * 60 * 60)

if not USE_LOCAL_CSV:
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Błąd inicjalizacji BigQuery: {e}. Jeśli chcesz użyć plików CSV użyj USE_LOCAL_CSV=true")
        client = None
else:
    client = None

class DataPoint(BaseModel):
    time: str
    value: float

class IndicatorResponse(BaseModel):
    indicator: str
    data: List[DataPoint]

class CorrelationResponse(BaseModel):
    indicator: str
    correlation: float

@cached(cache=bq_cache)
def get_dynamic_available_metrics() -> List[str]:
    """Zwraca listę dostępnych wskaźników dynamicznie."""
    if USE_LOCAL_CSV:
        csv_dir = os.path.join(os.path.dirname(__file__), "..", "data_ingestion", "csv")
        if not os.path.exists(csv_dir):
            return []
        return [f.replace(".csv", "") for f in os.listdir(csv_dir) if f.endswith(".csv")]
    else:
        if client is None:
            return []
        query = f"""
            SELECT column_name 
            FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMNS` 
            WHERE table_name = '{COINMETRICS_TABLE}' AND column_name != 'time'
        """
        try:
            results = client.query(query).result()
            return [row.column_name for row in results]
        except GoogleAPIError as e:
            print(f"Błąd podczas dynamicznego pobierania kolumn: {e}")
            return []

@app.get("/api/v1/metrics/available")
def get_available_metrics():
    indicators = get_dynamic_available_metrics()
    return {"indicators": indicators}

@cached(cache=bq_cache)
def fetch_metrics_data(valid_metrics_tuple: tuple, start_date: str, end_date: str):
    valid_metrics = list(valid_metrics_tuple)
    if USE_LOCAL_CSV:
        csv_dir = os.path.join(os.path.dirname(__file__), "..", "data_ingestion", "csv")
        final_response = []
        for m in valid_metrics:
            filepath = os.path.join(csv_dir, f"{m}.csv")
            data_points = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
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
        if client is None:
            raise Exception("Klient BigQuery nie został zainicjalizowany.")
        
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
        
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        
        response_data = {metric: [] for metric in valid_metrics}
        for row in results:
            for metric in valid_metrics:
                val = row[metric]
                if val is not None:
                    response_data[metric].append(DataPoint(time=row["time"], value=val))
        
        return [IndicatorResponse(indicator=m, data=response_data[m]) for m in valid_metrics]


@app.get("/api/v1/metrics/values", response_model=List[IndicatorResponse])
def get_metrics_values(
    metrics: List[str] = Query(..., description="Lista wskaźników (np. CapRealUSD, SOPR)"),
    start_date: str = Query("2009-01-01", description="Data początkowa RRRR-MM-DD"),
    end_date: str = Query("2030-01-01", description="Data końcowa RRRR-MM-DD")
):
    available = get_dynamic_available_metrics()
    valid_metrics = [m for m in metrics if m in available]
    
    if not valid_metrics:
        raise HTTPException(status_code=400, detail="Żaden z podanych wskaźników nie jest obsługiwany.")
    
    try:
        # Przekazujemy tuple do cachowania, bo listy nie są hashable
        return fetch_metrics_data(tuple(valid_metrics), start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics/correlation", response_model=List[CorrelationResponse])
def get_metrics_correlation(
    days: int = Query(90, description="Ilość ostatnich dni do analizy korelacji z ceną (domyślnie 90)"),
    target_metric: str = Query("PriceUSD", description="Wskaźnik bazowy (domyślnie PriceUSD)")
):
    """
    Oblicza korelację Pearsona między target_metric (np. PriceUSD) 
    a wszystkimi innymi wskaźnikami dla ostatnich `days` dni.
    """
    available = get_dynamic_available_metrics()
    if target_metric not in available:
        raise HTTPException(status_code=400, detail=f"Target metric {target_metric} nie istnieje.")
    
    # Dla uproszczenia pobieramy dane z ostatniego roku by zapewnić bezpieczeństwo cachowania (lub stałe daty)
    # W zaawansowanej aplikacji powinnismy zdefiniowac start/end date
    # Pobierzmy stały zakres ostatnich kilku lat żeby wykorzystać globalny cache
    end_date = "2030-01-01"
    start_date = "2018-01-01" # ograniczamy zapytanie dla przyspieszenia
    
    try:
        all_data = fetch_metrics_data(tuple(available), start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Słownik: indicator -> slownik: date -> value
    data_map = {}
    for indicator_obj in all_data:
        data_map[indicator_obj.indicator] = {dp.time: dp.value for dp in indicator_obj.data}
        
    target_data = data_map.get(target_metric, {})
    
    # Sort dates and get last N dates
    all_target_dates = sorted(list(target_data.keys()))
    recent_dates = all_target_dates[-days:] if len(all_target_dates) > days else all_target_dates
    
    if not recent_dates:
        return []

    correlations = []
    
    target_values = [target_data[d] for d in recent_dates]
    
    for indicator in available:
        if indicator == target_metric:
            continue
            
        ind_data = data_map.get(indicator, {})
        
        # Filtrujemy tylko te daty w ktorych istnieją oba punkty
        valid_target = []
        valid_ind = []
        
        for d in recent_dates:
            if d in target_data and d in ind_data:
                valid_target.append(target_data[d])
                valid_ind.append(ind_data[d])
                
        if len(valid_target) > 30: # Wymagamy min 30 wspolnych dni dla sensownej korelacji
            corr, _ = pearsonr(valid_target, valid_ind)
            if not np.isnan(corr):
                correlations.append(CorrelationResponse(indicator=indicator, correlation=float(corr)))
                
    # Sortuj po najsilniejszej bezwzględnej korelacji malejąco
    correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
    return correlations
