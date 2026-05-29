import pandas as pd
import requests
import os
from datetime import datetime

# Słownik endpointów z blockchain.com (klucz to nazwa wskaźnika)
ENDPOINTS = {
    "BlkSizeMeanByte": "https://api.blockchain.info/charts/avg-block-size?timespan=all&format=json",
    "UTXOCnt": "https://api.blockchain.info/charts/utxo-count?timespan=all&format=json",
    "TxTfrValNtv": "https://api.blockchain.info/charts/estimated-transaction-volume?timespan=all&format=json",
    "TxTfrValUSD": "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=all&format=json"
}

def fetch_and_save():
    os.makedirs("csv", exist_ok=True)
    cutoff_date = pd.to_datetime("2026-06-01")
    
    saved_count = 0
    for metric_name, url in ENDPOINTS.items():
        print(f"Pobieranie {metric_name} z blockchain.com...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "values" not in data:
                print(f"Błąd struktury w {metric_name}")
                continue
                
            # Parsowanie JSON do Pandas DataFrame
            # data["values"] ma strukturę [{"x": timestamp, "y": value}, ...]
            records = data["values"]
            df = pd.DataFrame(records)
            
            # Przekształcenie unix timestamp (x) na datę (time)
            df['time'] = pd.to_datetime(df['x'], unit='s')
            df['time'] = df['time'].dt.strftime('%Y-%m-%d')
            
            # Zmiana nazwy 'y' na nazwę wskaźnika
            df = df.rename(columns={'y': metric_name})
            
            # Konwersja BlkSizeMeanByte z MB na Bajty (jeśli to ten wskaźnik)
            if metric_name == "BlkSizeMeanByte":
                df[metric_name] = df[metric_name] * 1_000_000
                
            # Filtrujemy tylko czas do 2026-06-01
            df['time_dt'] = pd.to_datetime(df['time'])
            df = df[df['time_dt'] <= cutoff_date]
            
            # Usuwamy robocze kolumny i czyścimy NaN
            df = df[['time', metric_name]].dropna()
            
            csv_path = f"csv/{metric_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Zapisano {csv_path} ({len(df)} wierszy)")
            saved_count += 1
            
        except Exception as e:
            print(f"Błąd podczas pobierania {metric_name}: {e}")

    print(f"\nSukces! Wygenerowano dodatkowe {saved_count} plików CSV (Faza 1).")

if __name__ == "__main__":
    fetch_and_save()
