import pandas as pd
import numpy as np
from utils import load_merged_data, save_results

def compute_derived_metrics():
    df = load_merged_data()
    
    # Zapewnij posortowanie po dacie
    df = df.sort_values('time').reset_index(drop=True)
    
    # 1. Log-returns dla PriceUSD
    if 'PriceUSD' in df.columns:
        df['PriceUSD_log'] = np.log(df['PriceUSD'])
        df['r_t'] = df['PriceUSD_log'].diff()
        
        # Realized volatility (7-dniowa i 30-dniowa)
        df['rv_t'] = df['r_t'] ** 2
        df['rv_7d'] = df['r_t'].rolling(window=7).std() * np.sqrt(365) # Annualized
        df['rv_30d'] = df['r_t'].rolling(window=30).std() * np.sqrt(365)
        
        # Zmienna kierunkowa (1 up, 0 down)
        df['dir_t1'] = (df['r_t'].shift(-1) > 0).astype(int)

    # 2. NetFlows
    if all(c in df.columns for c in ['FlowInExUSD', 'FlowOutExUSD']):
        df['NetFlowUSD'] = df['FlowInExUSD'] - df['FlowOutExUSD']
    
    if all(c in df.columns for c in ['FlowInExNtv', 'FlowOutExNtv']):
        df['NetFlowNtv'] = df['FlowInExNtv'] - df['FlowOutExNtv']
        
    # 3. ShareExUSD (Udział giełd w wolumenie)
    if all(c in df.columns for c in ['FlowInExUSD', 'FlowOutExUSD', 'TxTfrValUSD']):
        df['ShareExUSD'] = (df['FlowInExUSD'] + df['FlowOutExUSD']) / df['TxTfrValUSD']
        df['ShareExUSD'] = df['ShareExUSD'].replace([np.inf, -np.inf], np.nan)
        
    # 4. Proxy dla PM (Profitability Miners)
    if all(c in df.columns for c in ['RevUSD', 'IssTotUSD']):
        df['PM_proxy'] = df['RevUSD'] / df['IssTotUSD'].replace(0, np.nan)
        
    # NVT (Network Value to Transactions)
    if all(c in df.columns for c in ['CapMrktCurUSD', 'TxTfrValUSD']):
        df['NVT'] = df['CapMrktCurUSD'] / df['TxTfrValUSD'].replace(0, np.nan)
        
    save_results(df, "derived_metrics.csv")
    print(f"Obliczono pochodne miary. Liczba wierszy: {len(df)}")
    
    # Zapis wybranych miar do katalogu z danymi, by API mogło je serwować
    import os
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_ingestion", "csv")
    if os.path.exists(csv_dir):
        export_metrics = ['ShareExUSD', 'NetFlowUSD', 'r_t', 'rv_30d']
        for m in export_metrics:
            if m in df.columns:
                out_df = df[['time', m]].dropna()
                out_path = os.path.join(csv_dir, f"{m}.csv")
                out_df.to_csv(out_path, index=False, header=['time', 'value'])
                print(f"Wyeksportowano {m}.csv do API.")

if __name__ == "__main__":
    compute_derived_metrics()
