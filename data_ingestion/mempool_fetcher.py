import pandas as pd
import requests
import os
from datetime import datetime

def get_daily_median_fee(target_date_str: str) -> float:
    # MVP
    response = requests.get("https://mempool.space/api/v1/blocks")
    if response.status_code == 200:
        blocks = response.json()
        median_fees = [b['extras']['medianFee'] for b in blocks if 'extras' in b and 'medianFee' in b['extras']]
        if median_fees:
            # median fee rate in sat/vB
            avg_sats_vb = sum(median_fees) / len(median_fees)
            return avg_sats_vb
    return 0.0

def run_daily_worker():
    today = datetime.utcnow().strftime('%Y-%m-%d')
    median_fee_rate = get_daily_median_fee(today)
    
    os.makedirs("csv", exist_ok=True)
    csv_path = "csv/FeeRateSatVb.csv"
    
    new_row = pd.DataFrame([{"time": today, "FeeRateSatVb": median_fee_rate}])
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['time'] != today]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
        
    df.to_csv(csv_path, index=False)
    print(f"Zapisano FeeRateSatVb.csv ({len(df)} wierszy).")

if __name__ == "__main__":
    run_daily_worker()
