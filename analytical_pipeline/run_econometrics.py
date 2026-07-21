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
    
    # All requested on-chain activity metrics
    metrics = ['AdrActCnt', 'TxCnt', 'TxCntSec', 'TxTfrCnt', 'TxTfrValUSD', 
               'UTXOCnt', 'BlkCnt', 'BlkSizeMeanByte', 'FeeTotUSD', 'FeeMeanUSD', 'FeeMedNtv']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df_clean.columns]
    
    reg_results = []
    for m in available_metrics:
        # log(x + 1) to avoid log(0), handle negative just in case
        X_val = np.log(df_clean[m].clip(lower=0) + 1)
        
        valid_idx = ~(X_val.isna() | np.isinf(X_val) | df_clean['h_t'].isna() | np.isinf(df_clean['h_t']))
        if valid_idx.sum() < 30:
            continue
            
        X = sm.add_constant(X_val[valid_idx])
        y = df_clean.loc[valid_idx, 'h_t']
        
        try:
            model = sm.OLS(y, X).fit()
            reg_results.append({
                "metric": m,
                "coef": float(model.params[m]),
                "p_value": float(model.pvalues[m]),
                "significant": bool(model.pvalues[m] < 0.05)
            })
        except:
            pass
            
    corr_matrix = df_clean[available_metrics].corr().to_dict()
    
    # Flag for h_t > 90th percentile
    h_t_90 = df_clean['h_t'].quantile(0.90)
    df_clean['is_high_vol'] = df_clean['h_t'] > h_t_90
    
    h_t_series = df_clean[['time', 'PriceUSD', 'h_t', 'is_high_vol']].tail(365).copy()
    h_t_series['time'] = h_t_series['time'].dt.strftime('%Y-%m-%d')
    
    return {
        "garch_summary": res.summary().as_text(),
        "regressions": reg_results,
        "correlation_heatmap": corr_matrix,
        "h_t_series": h_t_series.to_dict(orient='records')
    }

def run_paper2_netflows(df: pd.DataFrame):
    if 'NetFlowUSD' not in df.columns or 'r_t' not in df.columns or 'rv_t' not in df.columns or 'PriceUSD' not in df.columns:
        return {}
    df_clean = df.dropna(subset=['r_t', 'NetFlowUSD', 'rv_t', 'PriceUSD']).copy()
    if len(df_clean) < 10:
        return {}
        
    df_clean['NetFlowUSD_lag1'] = df_clean['NetFlowUSD'].shift(1)
    df_clean = df_clean.dropna(subset=['NetFlowUSD_lag1', 'NetFlowUSD', 'r_t', 'rv_t', 'PriceUSD']).copy()
    if len(df_clean) < 10:
        return {}
    
    X = df_clean[['NetFlowUSD', 'NetFlowUSD_lag1']]
    X = sm.add_constant(X)
    
    y_r = df_clean['r_t'] * 100
    model_r = sm.OLS(y_r, X).fit()
    
    y_rv = df_clean['rv_t'] * 10000
    model_rv = sm.OLS(y_rv, X).fit()
    
    # Prepare time series for charting
    ts_data = df_clean[['time', 'PriceUSD', 'NetFlowUSD', 'r_t']].tail(365).copy()
    ts_data['time'] = ts_data['time'].dt.strftime('%Y-%m-%d')
    ts_data['r_t'] = ts_data['r_t'] * 100
    
    # Calculate quantiles to identify high positive/negative flows
    nf_q90 = df_clean['NetFlowUSD'].quantile(0.90)
    nf_q10 = df_clean['NetFlowUSD'].quantile(0.10)
    ts_data['is_extreme_flow'] = (ts_data['NetFlowUSD'] > nf_q90) | (ts_data['NetFlowUSD'] < nf_q10)
    
    return {
        "returns_model": {
            "coef_lag0": float(model_r.params['NetFlowUSD']),
            "pval_lag0": float(model_r.pvalues['NetFlowUSD']),
            "coef_lag1": float(model_r.params['NetFlowUSD_lag1']),
            "pval_lag1": float(model_r.pvalues['NetFlowUSD_lag1'])
        },
        "volatility_model": {
            "coef_lag0": float(model_rv.params['NetFlowUSD']),
            "pval_lag0": float(model_rv.pvalues['NetFlowUSD']),
            "coef_lag1": float(model_rv.params['NetFlowUSD_lag1']),
            "pval_lag1": float(model_rv.pvalues['NetFlowUSD_lag1'])
        },
        "time_series": ts_data.to_dict(orient='records'),
        "scatter_data": ts_data[['NetFlowUSD', 'r_t']].to_dict(orient='records')
    }

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def run_paper10_pca(df: pd.DataFrame):
    features = ['TxCnt', 'TxTfrCnt', 'TxTfrValUSD', 'UTXOCnt', 'AdrActCnt', 'FeeTotUSD', 'BlkSizeMeanByte']
    features = [f for f in features if f in df.columns]
    
    if len(features) < 2 or 'rv_30d' not in df.columns or 'PriceUSD' not in df.columns:
        return {}
        
    df_clean = df.dropna(subset=features + ['rv_30d', 'PriceUSD']).copy()
    X = df_clean[features]
    X = (X - X.mean()) / X.std()
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    
    df_clean['PC1'] = pcs[:, 0]
    df_clean['PC2'] = pcs[:, 1]
    
    threshold = df_clean['rv_30d'].quantile(0.95)
    df_clean['high_vol'] = (df_clean['rv_30d'] > threshold).astype(int)
    
    # Logistic regression: PC1, PC2 -> high_vol
    X_logit = sm.add_constant(df_clean[['PC1', 'PC2']])
    logit_model = sm.Logit(df_clean['high_vol'], X_logit).fit(disp=0)
    
    # Predictions
    df_clean['pred_prob'] = logit_model.predict(X_logit)
    df_clean['pred_class'] = (df_clean['pred_prob'] > 0.5).astype(int)
    
    y_true = df_clean['high_vol']
    y_pred = df_clean['pred_class']
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
    
    # EWI is basically the probability or a linear combination
    df_clean['EWI'] = df_clean['pred_prob']
    ewi_threshold = df_clean['EWI'].quantile(0.90)
    df_clean['is_ewi_alert'] = df_clean['EWI'] > ewi_threshold
    
    ts_data = df_clean[['time', 'PriceUSD', 'rv_30d', 'EWI', 'is_ewi_alert']].tail(365).copy()
    ts_data['time'] = ts_data['time'].dt.strftime('%Y-%m-%d')
    
    scatter_data = df_clean[['PC1', 'PC2', 'high_vol']].tail(500).copy()
    
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "classification_metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        },
        "time_series": ts_data.to_dict(orient='records'),
        "scatter_data": scatter_data.to_dict(orient='records')
    }

def run_paper3_ardl(df: pd.DataFrame):
    sopr_col = 'SOPR' if 'SOPR' in df.columns else 'NUPL'
    if sopr_col not in df.columns or 'PM_proxy' not in df.columns or 'AdrActCnt' not in df.columns or 'PriceUSD' not in df.columns:
        return {}
        
    df_clean = df.dropna(subset=['PriceUSD', sopr_col, 'PM_proxy', 'AdrActCnt']).copy()
    
    # Resample to weekly to avoid noise and speed up ARDL
    df_clean.set_index('time', inplace=True)
    df_w = df_clean[['PriceUSD', sopr_col, 'PM_proxy', 'AdrActCnt']].resample('W').mean().dropna()
    df_w['time'] = df_w.index.strftime('%Y-%m-%d')
    df_w.reset_index(drop=True, inplace=True)
    
    # Identify high/low quantiles for the UI
    for col in [sopr_col, 'PM_proxy', 'AdrActCnt']:
        q_high = df_w[col].quantile(0.85)
        q_low = df_w[col].quantile(0.15)
        df_w[f'{col}_is_extreme'] = (df_w[col] > q_high) | (df_w[col] < q_low)
        
    # ADF Test
    adf_sopr = adfuller(df_w[sopr_col])
    adf_pm = adfuller(df_w['PM_proxy'])
    
    # Simple proxy for ARDL (lag 1 for y, lag 0 and 1 for X)
    # y_t = c + a*y_{t-1} + b0*X_t + b1*X_{t-1}
    df_ardl = df_w.copy()
    df_ardl['Price_lag1'] = df_ardl['PriceUSD'].shift(1)
    df_ardl[f'{sopr_col}_lag1'] = df_ardl[sopr_col].shift(1)
    df_ardl['PM_lag1'] = df_ardl['PM_proxy'].shift(1)
    df_ardl['AdrActCnt_lag1'] = df_ardl['AdrActCnt'].shift(1)
    df_ardl = df_ardl.dropna()
    
    X_cols = [f'{sopr_col}', f'{sopr_col}_lag1', 'PM_proxy', 'PM_lag1', 'AdrActCnt', 'AdrActCnt_lag1']
    X = df_ardl[['Price_lag1'] + X_cols]
    X = sm.add_constant(X)
    y = df_ardl['PriceUSD']
    
    model = sm.OLS(y, X).fit()
    
    # Long run multiplier for X: (b0 + b1) / (1 - a)
    a = model.params['Price_lag1']
    lr_sopr = (model.params[f'{sopr_col}'] + model.params[f'{sopr_col}_lag1']) / (1 - a) if a != 1 else 0
    lr_pm = (model.params['PM_proxy'] + model.params['PM_lag1']) / (1 - a) if a != 1 else 0
    lr_adr = (model.params['AdrActCnt'] + model.params['AdrActCnt_lag1']) / (1 - a) if a != 1 else 0
    
    short_term_effects = []
    for col in X_cols:
        short_term_effects.append({
            "variable": col,
            "coef": float(model.params[col]),
            "p_value": float(model.pvalues[col]),
            "significant": bool(model.pvalues[col] < 0.05)
        })
        
    return {
        "adf_tests": {
            sopr_col: {"statistic": float(adf_sopr[0]), "p_value": float(adf_sopr[1])},
            "PM_proxy": {"statistic": float(adf_pm[0]), "p_value": float(adf_pm[1])}
        },
        "short_term_ardl": short_term_effects,
        "long_term_multipliers": {
            sopr_col: float(lr_sopr),
            "PM_proxy": float(lr_pm),
            "AdrActCnt": float(lr_adr)
        },
        "time_series": df_w.tail(156).to_dict(orient='records'), # Last 3 years
        "sopr_column": sopr_col
    }

def run_paper8_cdd(df: pd.DataFrame):
    # Proxy CDD with TxTfrValUSD if not available
    cdd_col = 'CDD' if 'CDD' in df.columns else 'TxTfrValUSD'
    if cdd_col not in df.columns or 'rv_30d' not in df.columns or 'PriceUSD' not in df.columns:
        return {}
        
    cols_to_keep = ['time', 'PriceUSD', cdd_col, 'rv_30d']
    sply_cols = [c for c in ['SplyAct30d', 'SplyAct180d', 'SplyAct1yr'] if c in df.columns]
    cols_to_keep += sply_cols
    
    df_clean = df.dropna(subset=['PriceUSD', cdd_col, 'rv_30d']).copy()
    
    # Find 95th percentile of CDD
    p95 = df_clean[cdd_col].quantile(0.95)
    df_clean['is_extreme_cdd'] = df_clean[cdd_col] > p95
    
    # Regression: RV_t = a + b1*CDD_t + b2*SplyAct_t
    reg_features = [cdd_col]
    if len(sply_cols) > 0:
        sply_candidate = sply_cols[-1]
        # Check if the column is entirely null
        if df_clean[sply_candidate].notna().sum() > 30:
            reg_features.append(sply_candidate)
            df_clean = df_clean.dropna(subset=reg_features)
            
    if len(df_clean) < 10:
        return {}
        
    X = df_clean[reg_features]
    X = sm.add_constant(X)
    y = df_clean['rv_30d']
    model = sm.OLS(y, X).fit()
    
    reg_results = []
    for f in reg_features:
        reg_results.append({
            "variable": f,
            "coef": float(model.params[f]),
            "p_value": float(model.pvalues[f]),
            "significant": bool(model.pvalues[f] < 0.05)
        })
        
    ts_data = df_clean[['time', 'PriceUSD', cdd_col, 'is_extreme_cdd'] + sply_cols].tail(365).copy()
    ts_data['time'] = ts_data['time'].dt.strftime('%Y-%m-%d')
    
    scatter_data = df_clean[[cdd_col, 'rv_30d']].tail(365).rename(columns={cdd_col: 'cdd'})
    
    return {
        "p95_threshold": float(p95),
        "cdd_column": cdd_col,
        "regression": reg_results,
        "time_series": ts_data.to_dict(orient='records'),
        "scatter_data": scatter_data.to_dict(orient='records')
    }

def run_paper9_exchange(df: pd.DataFrame):
    if 'ShareExUSD' not in df.columns or 'r_t' not in df.columns or 'PriceUSD' not in df.columns:
        return {}
        
    df_clean = df.dropna(subset=['ShareExUSD', 'r_t', 'PriceUSD']).copy()
    
    # Rolling correlation (30 days)
    rolling_corr = df_clean['ShareExUSD'].rolling(window=30).corr(df_clean['rv_30d'] if 'rv_30d' in df.columns else df_clean['r_t'].abs())
    
    df_clean['rolling_corr'] = rolling_corr
    df_clean = df_clean.dropna(subset=['rolling_corr'])
    
    q80 = df_clean['ShareExUSD'].quantile(0.80)
    df_clean['is_high_share'] = df_clean['ShareExUSD'] > q80
    
    ts_data = df_clean[['time', 'PriceUSD', 'ShareExUSD', 'rolling_corr', 'is_high_share']].tail(365).copy()
    ts_data['time'] = ts_data['time'].dt.strftime('%Y-%m-%d')
    
    return {
        "stats": {
            "mean": float(df_clean['ShareExUSD'].mean()),
            "median": float(df_clean['ShareExUSD'].median()),
            "q20": float(df_clean['ShareExUSD'].quantile(0.20)),
            "q80": float(q80),
            "max": float(df_clean['ShareExUSD'].max())
        },
        "time_series": ts_data.to_dict(orient='records')
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
