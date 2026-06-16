import requests
import pandas as pd
import os

ENDPOINTS = {
    "SOPR": "https://api.bitcoin-data.com/v1/sopr",
    "CDD": "https://api.bitcoin-data.com/v1/cdd",
    "LTH_MVRV": "https://api.bitcoin-data.com/v1/lth-mvrv",
    "STH_MVRV": "https://api.bitcoin-data.com/v1/sth-mvrv"
}

def fetch_bgeometrics():
    os.makedirs("csv", exist_ok=True)
    cutoff_date = pd.to_datetime("2026-06-01")
    
    for metric_name, url in ENDPOINTS.items():
        print(f"Pobieranie {metric_name} z BGeometrics...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # format: [{"d":"2022-06-12","unixTs":1654992000,"sopr":0.9935}, ...]
            df = pd.DataFrame(data)
            
            # Heurystyka klucza json (np. LTH_MVRV -> lthMvrv, SOPR -> sopr)
            val_col = metric_name.lower()
            if metric_name == "LTH_MVRV": val_col = "lthMvrv"
            if metric_name == "STH_MVRV": val_col = "sthMvrv"
            
            if val_col not in df.columns:
                print(f"Brak kolumny {val_col} w odpowiedzi dla {metric_name}")
                continue
                
            df['time'] = pd.to_datetime(df['d'])
            df = df[['time', val_col]].rename(columns={val_col: metric_name})
            df = df[df['time'] <= cutoff_date]
            
            df = df.dropna()
            
            csv_path = f"csv/{metric_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Zapisano {csv_path} ({len(df)} wierszy)")
        except Exception as e:
            print(f"Błąd podczas pobierania {metric_name}: {e}")

if __name__ == "__main__":
    fetch_bgeometrics()
