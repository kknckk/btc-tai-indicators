import pandas as pd
import requests
from io import StringIO
import os

COINMETRICS_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

def fetch_and_derive():
    print("Pobieranie btc.csv z CoinMetrics...")
    response = requests.get(COINMETRICS_URL)
    response.raise_for_status()
    
    print("Parsowanie CSV...")
    df = pd.read_csv(StringIO(response.text))
    df['time'] = pd.to_datetime(df['time'])
    cutoff_date = pd.to_datetime("2026-06-01")
    df = df[df['time'] <= cutoff_date].copy()

    # Bazowe wskaźniki (dostępne bezpośrednio w darmowym CSV)
    base_metrics = [
        "AdrActCnt", "AdrBalCnt", "BlkCnt", "CapMVRVCur", "CapMrktCurUSD", 
        "FeeTotNtv", "FlowInExNtv", "FlowInExUSD", "FlowOutExNtv", "FlowOutExUSD",
        "HashRate", "IssTotNtv", "IssTotUSD", "PriceBTC", "PriceUSD", 
        "SplyCur", "SplyExpFut10yr", "TxCnt", "TxTfrCnt"
    ]
    
    print("Obliczanie wskaźników pochodnych (DERIVED)...")
    # 1. CapRealUSD (z MVRV = CapMrkt / CapReal -> CapReal = CapMrkt / MVRV)
    df["CapRealUSD"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
    
    # 2. RealizedPrice
    df["RealizedPrice"] = df["CapRealUSD"] / df["SplyCur"]
    
    # 3. NUPL
    df["NUPL"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
    
    # 4. MVRV_Z
    # MVRV Z-Score = (Market Cap – Realized Cap) / StdDev(Market Cap)
    market_cap_std = df["CapMrktCurUSD"].expanding().std()
    df["MVRV_Z"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / market_cap_std
    
    # 5. FeeTotUSD
    df["FeeTotUSD"] = df["FeeTotNtv"] * df["PriceUSD"]
    
    # 6. RevUSD (Przychód górników = Nowa Podaż w USD + Opłaty w USD)
    df["RevUSD"] = df["IssTotUSD"] + df["FeeTotUSD"]
    
    # 7. FeeRevPct
    df["FeeRevPct"] = 100 * (df["FeeTotUSD"] / df["RevUSD"])
    
    # 8. PuellMultiple
    ma365_rev = df["RevUSD"].rolling(window=365, min_periods=1).mean()
    df["PuellMultiple"] = df["RevUSD"] / ma365_rev
    
    # 9. FeeMeanNtv & FeeMeanUSD
    df["FeeMeanNtv"] = df["FeeTotNtv"] / df["TxCnt"]
    df["FeeMeanUSD"] = df["FeeTotUSD"] / df["TxCnt"]
    
    # 10. BlkIntMean
    df["BlkIntMean"] = 86400 / df["BlkCnt"]
    
    # 11. CapFutExp10yrUSD
    df["CapFutExp10yrUSD"] = df["SplyExpFut10yr"] * df["PriceUSD"]
    
    # 12. IssContPctDay & IssContPctAnn
    df["IssContPctDay"] = (df["IssTotNtv"] / df["SplyCur"]) * 100
    df["IssContPctAnn"] = df["IssContPctDay"] * 365
    
    # 13. TxCntSec
    df["TxCntSec"] = df["TxCnt"] / 86400

    derived_metrics = [
        "CapRealUSD", "RealizedPrice", "NUPL", "MVRV_Z", "FeeTotUSD", "RevUSD", 
        "FeeRevPct", "PuellMultiple", "FeeMeanNtv", "FeeMeanUSD", "BlkIntMean",
        "CapFutExp10yrUSD", "IssContPctDay", "IssContPctAnn", "TxCntSec"
    ]
    
    all_metrics = base_metrics + derived_metrics
    
    os.makedirs("csv", exist_ok=True)
    
    saved_count = 0
    for metric in all_metrics:
        if metric in df.columns:
            df_metric = df[["time", metric]].dropna()
            # Tylko zapisujemy jeśli są jakieś dane
            if not df_metric.empty:
                csv_path = f"csv/{metric}.csv"
                df_metric.to_csv(csv_path, index=False)
                saved_count += 1
                
    print(f"Zapisano {saved_count} wskaźników jako pliki CSV!")

if __name__ == "__main__":
    fetch_and_derive()
