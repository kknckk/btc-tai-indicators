import os
import pandas as pd
import requests

def validate_coinmetrics():
    print("=== Validating with CoinMetrics API ===")
    metric = "AdrActCnt"
    url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics={metric}&limit_per_asset=5"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(base_dir, "data_ingestion", "csv", f"{metric}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            latest_csv_date = df['time'].max()
            
            # Zapytajmy API o tę konkretną datę
            api_url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics={metric}&start_time={latest_csv_date}&end_time={latest_csv_date}"
            res = requests.get(api_url).json()
            
            if "data" in res and len(res["data"]) > 0:
                api_value = float(res["data"][0][metric])
                
                our_row = df[df['time'] == latest_csv_date]
                our_value = float(our_row.iloc[0].iloc[1])
                
                diff = abs(api_value - our_value)
                perc_diff = (diff / api_value) * 100 if api_value != 0 else 0
                
                print(f"Data weryfikacji: {latest_csv_date}")
                print(f"CoinMetrics API: {api_value}")
                print(f"Nasz wynik: {our_value}")
                print(f"Różnica: {perc_diff:.2f}%")
                
                if perc_diff < 5:
                    print("✅ Wskaźnik poprawnie wyliczony.")
                else:
                    print("❌ Ostrzeżenie: rozbieżność!")
            else:
                print(f"API nie zwróciło danych dla {latest_csv_date}")
        else:
            print(f"Brak pliku {csv_path}")
    except Exception as e:
        print(f"Błąd podczas połączenia: {e}")

if __name__ == "__main__":
    validate_coinmetrics()
