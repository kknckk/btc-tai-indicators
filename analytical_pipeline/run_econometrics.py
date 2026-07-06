import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from utils import save_results

def run_paper1_garch(df: pd.DataFrame):
    df_clean = df.dropna(subset=['r_t', 'AdrActCnt', 'TxCntSec', 'FeeTotUSD'])
    # Scale returns for GARCH stability
    returns = df_clean['r_t'] * 100
    
    am = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    res = am.fit(disp='off')
    
    df_clean['h_t'] = res.conditional_volatility
    
    metrics = ['AdrActCnt', 'TxCntSec', 'FeeTotUSD']
    reg_results = []
    
    for m in metrics:
        if m in df_clean.columns:
            X = np.log(df_clean[m] + 1)
            X = sm.add_constant(X)
            y = df_clean['h_t']
            model = sm.OLS(y, X).fit()
            reg_results.append({
                "metric": m,
                "coef": model.params[m],
                "p_value": model.pvalues[m],
                "significant": model.pvalues[m] < 0.05
            })
            
    corr_matrix = df_clean[['AdrActCnt', 'TxCntSec', 'TxTfrCnt', 'TxTfrValUSD']].corr().to_dict()
    
    return {
        "garch_summary": res.summary().as_text(),
        "regressions": reg_results,
        "correlation_heatmap": corr_matrix,
        "h_t_series": df_clean[['time', 'PriceUSD', 'h_t']].tail(365).to_dict(orient='records')
    }

def run_paper2_netflows(df: pd.DataFrame):
    df_clean = df.dropna(subset=['r_t', 'NetFlowUSD'])
    df_clean['NetFlowUSD_lag1'] = df_clean['NetFlowUSD'].shift(1)
    df_clean = df_clean.dropna()
    
    X = df_clean[['NetFlowUSD', 'NetFlowUSD_lag1']]
    X = sm.add_constant(X)
    
    y_r = df_clean['r_t'] * 100
    model_r = sm.OLS(y_r, X).fit()
    
    y_rv = df_clean['rv_t'] * 10000
    model_rv = sm.OLS(y_rv, X).fit()
    
    return {
        "returns_model": {
            "coef_lag0": model_r.params['NetFlowUSD'],
            "pval_lag0": model_r.pvalues['NetFlowUSD'],
            "coef_lag1": model_r.params['NetFlowUSD_lag1'],
            "pval_lag1": model_r.pvalues['NetFlowUSD_lag1']
        },
        "volatility_model": {
            "coef_lag0": model_rv.params['NetFlowUSD'],
            "pval_lag0": model_rv.pvalues['NetFlowUSD'],
            "coef_lag1": model_rv.params['NetFlowUSD_lag1'],
            "pval_lag1": model_rv.pvalues['NetFlowUSD_lag1']
        }
    }

def run_paper10_pca(df: pd.DataFrame):
    features = ['TxCnt', 'TxTfrCnt', 'TxTfrValUSD', 'UTXOCnt', 'AdrActCnt', 'FeeTotUSD', 'BlkSizeMeanByte']
    # Check what features exist
    features = [f for f in features if f in df.columns]
    
    df_clean = df.dropna(subset=features + ['rv_30d'])
    X = df_clean[features]
    # Normalize
    X = (X - X.mean()) / X.std()
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    
    df_clean['PC1'] = pcs[:, 0]
    df_clean['PC2'] = pcs[:, 1]
    
    threshold = df_clean['rv_30d'].quantile(0.90)
    df_clean['high_vol'] = (df_clean['rv_30d'] > threshold).astype(int)
    
    # Logistic regression: PC1, PC2 -> high_vol
    logit_model = sm.Logit(df_clean['high_vol'], sm.add_constant(df_clean[['PC1', 'PC2']])).fit(disp=0)
    
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "logit_summary": logit_model.summary().as_text(),
        "p_values": {
            "PC1": logit_model.pvalues['PC1'],
            "PC2": logit_model.pvalues['PC2']
        }
    }

def run_paper3_ardl(df: pd.DataFrame):
    # Proxy SOPR with NUPL, PM_proxy is Miner Profitability
    if 'NUPL' not in df.columns or 'PM_proxy' not in df.columns or 'r_t' not in df.columns:
        return {}
        
    df_clean = df.dropna(subset=['r_t', 'NUPL', 'PM_proxy']).copy()
    
    # ADF Test
    adf_nupl = adfuller(df_clean['NUPL'])
    adf_pm = adfuller(df_clean['PM_proxy'])
    
    # Granger Causality (NUPL -> r_t)
    data_gc = df_clean[['r_t', 'NUPL']].values
    gc_res = grangercausalitytests(data_gc, maxlag=3, verbose=False)
    
    # Extract p-value of lag 1 for SSR F test
    p_val_lag1 = gc_res[1][0]['ssr_ftest'][1]
    
    return {
        "adf_tests": {
            "NUPL": {"statistic": float(adf_nupl[0]), "p_value": float(adf_nupl[1])},
            "PM_proxy": {"statistic": float(adf_pm[0]), "p_value": float(adf_pm[1])}
        },
        "granger_causality": {
            "NUPL_causes_return_lag1_pval": float(p_val_lag1)
        }
    }

def run_paper8_cdd(df: pd.DataFrame):
    # Proxy CDD with TxTfrValUSD
    if 'TxTfrValUSD' not in df.columns or 'rv_30d' not in df.columns:
        return {}
        
    df_clean = df.dropna(subset=['TxTfrValUSD', 'rv_30d']).copy()
    
    # Find 95th percentile
    p95 = df_clean['TxTfrValUSD'].quantile(0.95)
    
    df_extreme = df_clean[df_clean['TxTfrValUSD'] > p95]
    df_normal = df_clean[df_clean['TxTfrValUSD'] <= p95]
    
    return {
        "p95_threshold": float(p95),
        "extreme_days_count": len(df_extreme),
        "normal_days_count": len(df_normal),
        "mean_volatility_extreme": float(df_extreme['rv_30d'].mean()),
        "mean_volatility_normal": float(df_normal['rv_30d'].mean())
    }

def run_paper9_exchange(df: pd.DataFrame):
    if 'ShareExUSD' not in df.columns or 'r_t' not in df.columns:
        return {}
        
    df_clean = df.dropna(subset=['ShareExUSD', 'r_t']).copy()
    
    # Rolling correlation (90 days)
    rolling_corr = df_clean['ShareExUSD'].rolling(window=90).corr(df_clean['r_t'].abs())
    
    df_clean['rolling_corr'] = rolling_corr
    df_clean = df_clean.dropna(subset=['rolling_corr'])
    
    return {
        "mean_rolling_corr": float(df_clean['rolling_corr'].mean()),
        "max_rolling_corr": float(df_clean['rolling_corr'].max()),
        "min_rolling_corr": float(df_clean['rolling_corr'].min()),
        "rolling_corr_series": df_clean[['time', 'rolling_corr']].tail(365).to_dict(orient='records')
    }

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    df_path = os.path.join(results_dir, "derived_metrics.csv")
    if not os.path.exists(df_path):
        print("Brak derived_metrics.csv - uruchom najpierw compute_derived_metrics.py")
        return
        
    df = pd.read_csv(df_path)
    df['time'] = pd.to_datetime(df['time'])
    
    results = {}
    
    # Paper 1
    try:
        results['paper1'] = run_paper1_garch(df)
        print("Paper 1 (GARCH) OK")
    except Exception as e:
        print(f"Paper 1 Error: {e}")
        
    # Paper 2
    try:
        if 'NetFlowUSD' in df.columns:
            results['paper2'] = run_paper2_netflows(df)
            print("Paper 2 (NetFlow) OK")
    except Exception as e:
        print(f"Paper 2 Error: {e}")
        
    # Paper 10
    try:
        results['paper10'] = run_paper10_pca(df)
        print("Paper 10 (PCA) OK")
    except Exception as e:
        print(f"Paper 10 Error: {e}")
        
    # Paper 3
    try:
        results['paper3'] = run_paper3_ardl(df)
        print("Paper 3 (ARDL/Granger Proxy) OK")
    except Exception as e:
        print(f"Paper 3 Error: {e}")
        
    # Paper 8
    try:
        results['paper8'] = run_paper8_cdd(df)
        print("Paper 8 (CDD Proxy) OK")
    except Exception as e:
        print(f"Paper 8 Error: {e}")

    # Paper 9
    try:
        results['paper9'] = run_paper9_exchange(df)
        print("Paper 9 (Exchange concentration) OK")
    except Exception as e:
        print(f"Paper 9 Error: {e}")
        
    with open(os.path.join(results_dir, "econometrics_results.json"), "w") as f:
        json.dump(results, f, default=str)
    
    print("Wyniki ekonometryczne zapisane.")

if __name__ == "__main__":
    main()
