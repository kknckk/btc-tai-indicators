import pandas as pd
import os
import shutil

CSV_DIR = os.path.join(os.path.dirname(__file__), "csv")

def safe_read(metric_name):
    path = os.path.join(CSV_DIR, f"{metric_name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def compute_derived():
    print("Obliczanie Derived Metrics...")
    
    # 1. IssContNtv (Kopia IssTotNtv)
    iss_tot = safe_read("IssTotNtv")
    if iss_tot is not None:
        iss_cont = iss_tot.rename(columns={"IssTotNtv": "IssContNtv"})
        iss_cont.to_csv(os.path.join(CSV_DIR, "IssContNtv.csv"), index=False)
        print("Wygenerowano IssContNtv.csv")
        
    # 2. TxTfrValMeanNtv & TxTfrValMeanUSD
    tx_val_ntv = safe_read("TxTfrValNtv")
    tx_val_usd = safe_read("TxTfrValUSD")
    tx_cnt = safe_read("TxTfrCnt")
    
    if tx_cnt is not None:
        if tx_val_ntv is not None:
            df = tx_val_ntv.merge(tx_cnt, on="time", how="inner")
            df["TxTfrValMeanNtv"] = df["TxTfrValNtv"] / df["TxTfrCnt"]
            df[["time", "TxTfrValMeanNtv"]].to_csv(os.path.join(CSV_DIR, "TxTfrValMeanNtv.csv"), index=False)
            print("Wygenerowano TxTfrValMeanNtv.csv")
            
        if tx_val_usd is not None:
            df = tx_val_usd.merge(tx_cnt, on="time", how="inner")
            df["TxTfrValMeanUSD"] = df["TxTfrValUSD"] / df["TxTfrCnt"]
            df[["time", "TxTfrValMeanUSD"]].to_csv(os.path.join(CSV_DIR, "TxTfrValMeanUSD.csv"), index=False)
            print("Wygenerowano TxTfrValMeanUSD.csv")

    # 3. RevAllTimeUSD (Skumulowany przychód = Opłaty + Emisja)
    fee_tot = safe_read("FeeTotUSD")
    iss_tot_usd = safe_read("IssTotUSD")
    if fee_tot is not None and iss_tot_usd is not None:
        df = fee_tot.merge(iss_tot_usd, on="time", how="inner").sort_values("time")
        df["RevAllTimeUSD"] = (df["FeeTotUSD"] + df["IssTotUSD"]).cumsum()
        df[["time", "RevAllTimeUSD"]].to_csv(os.path.join(CSV_DIR, "RevAllTimeUSD.csv"), index=False)
        print("Wygenerowano RevAllTimeUSD.csv")

    # 4. VelCur1yr (Prawdziwa prędkość z 1-roczną sumą)
    sply_cur = safe_read("SplyCur")
    if tx_val_ntv is not None and sply_cur is not None:
        df = tx_val_ntv.merge(sply_cur, on="time", how="inner")
        # Zabezpieczenie przed dzieleniem przez 0
        df = df[df["SplyCur"] > 0]
        df["TxTfrValNtv_1yr_sum"] = df["TxTfrValNtv"].rolling(window=365, min_periods=1).sum()
        df["VelCur1yr"] = df["TxTfrValNtv_1yr_sum"] / df["SplyCur"]
        df[["time", "VelCur1yr"]].to_csv(os.path.join(CSV_DIR, "VelCur1yr.csv"), index=False)
        print("Wygenerowano VelCur1yr.csv")
        
    # 5. NVT_Naive (Market Cap / TxTfrValUSD)
    cap_mrkt = safe_read("CapMrktCurUSD")
    if cap_mrkt is not None and tx_val_usd is not None:
        df = cap_mrkt.merge(tx_val_usd, on="time", how="inner")
        df = df[df["TxTfrValUSD"] > 0]
        df["NVT_Naive"] = df["CapMrktCurUSD"] / df["TxTfrValUSD"]
        df[["time", "NVT_Naive"]].to_csv(os.path.join(CSV_DIR, "NVT_Naive.csv"), index=False)
        print("Wygenerowano NVT_Naive.csv")

        # 6. NVTAdj90 (Market Cap / 90-day moving average of TxTfrValUSD)
        df["TxTfrValUSD_SMA90"] = df["TxTfrValUSD"].rolling(window=90, min_periods=1).mean()
        df["NVTAdj90"] = df["CapMrktCurUSD"] / df["TxTfrValUSD_SMA90"]
        df[["time", "NVTAdj90"]].to_csv(os.path.join(CSV_DIR, "NVTAdj90.csv"), index=False)
        print("Wygenerowano NVTAdj90.csv")

if __name__ == "__main__":
    compute_derived()
