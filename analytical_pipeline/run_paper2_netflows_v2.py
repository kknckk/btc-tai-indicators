import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import sys

# Ensure analytical_pipeline directory is in sys.path
sys.path.append(os.path.dirname(__file__))
from utils import load_merged_data

def run_paper2_netflows_v2():
    warnings.filterwarnings('ignore')
    
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "derived_metrics.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_merged_data()
        
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 1. Feature Construction
    if 'FlowInExUSD' in df.columns and 'FlowOutExUSD' in df.columns:
        df['NetFlowUSD'] = df['FlowInExUSD'] - df['FlowOutExUSD']
    elif 'NetFlowUSD' not in df.columns:
        raise ValueError("Missing required flow metrics (FlowInExUSD, FlowOutExUSD or NetFlowUSD).")
    
    if 'CapMrktCurUSD' in df.columns and 'NetFlowUSD' in df.columns:
        # Avoid division by zero
        df['CapMrktCurUSD'] = df['CapMrktCurUSD'].replace(0, np.nan)
        df['NetFlow_z'] = df['NetFlowUSD'] / df['CapMrktCurUSD']
    else:
        # Fallback to log1p
        df['NetFlow_z'] = np.sign(df['NetFlowUSD']) * np.log1p(np.abs(df['NetFlowUSD']))
        
    # Scale to basis points for readability if it's very small
    df['NetFlow_z'] = df['NetFlow_z'] * 10000.0 # bps of market cap
        
    # Calculate returns and RV if missing
    if 'r_t' not in df.columns:
        df['r_t'] = df['PriceUSD'].pct_change()
        
    df['rv_7d'] = df['r_t'].rolling(7).std() * np.sqrt(365)
    df['rv_30d'] = df['r_t'].rolling(30).std() * np.sqrt(365)
    
    # Lags for exogenous variables
    df['NetFlow_z_lag1'] = df['NetFlow_z'].shift(1)
    df['NetFlow_z_lag7'] = df['NetFlow_z'].shift(7)
    df['r_t_lag1'] = df['r_t'].shift(1)
    df['r_t_lag7'] = df['r_t'].shift(7)
    df['rv_7d_lag1'] = df['rv_7d'].shift(1)
    df['rv_30d_lag1'] = df['rv_30d'].shift(1)
    
    req_cols = ['time', 'PriceUSD', 'r_t', 'rv_7d', 'rv_30d', 'NetFlow_z', 
                'NetFlow_z_lag1', 'NetFlow_z_lag7', 'r_t_lag1', 'r_t_lag7']
    
    # Forward filling could be an option if there are gaps, but we will just dropna
    df_clean = df.dropna(subset=req_cols).copy().reset_index(drop=True)
    
    # Base HAC OLS Models
    def fit_ols_hac(y_col, x_cols, data):
        y = data[y_col]
        X = sm.add_constant(data[x_cols])
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        
        res = {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "coefficients": {}
        }
        for col in X.columns:
            res["coefficients"][col] = {
                "coef": float(model.params[col]),
                "p_value": float(model.pvalues[col]),
                "t_stat": float(model.tvalues[col]),
                "HAC_se": float(model.bse[col])
            }
        return res
        
    ols_returns = fit_ols_hac('r_t', ['NetFlow_z_lag1', 'NetFlow_z_lag7', 'r_t_lag1', 'r_t_lag7'], df_clean)
    ols_vol_7d = fit_ols_hac('rv_7d', ['NetFlow_z_lag1', 'NetFlow_z_lag7', 'rv_7d_lag1'], df_clean)
    ols_vol_30d = fit_ols_hac('rv_30d', ['NetFlow_z_lag1', 'NetFlow_z_lag7', 'rv_30d_lag1'], df_clean)
    
    # 3. Local Projections (Jorda) for Returns
    local_projections_ret = {}
    max_h = 7
    for h in range(1, max_h + 1):
        df_clean[f'fwd_ret_{h}d'] = (df_clean['PriceUSD'].shift(-h) - df_clean['PriceUSD']) / df_clean['PriceUSD']
        
    df_lp = df_clean.dropna(subset=[f'fwd_ret_{h}d' for h in range(1, max_h+1)]).copy()
    
    for h in range(1, max_h + 1):
        lp_res = fit_ols_hac(f'fwd_ret_{h}d', ['NetFlow_z', 'r_t'], df_lp) 
        local_projections_ret[f"h={h}"] = {
            "NetFlow_z_coef": lp_res["coefficients"]["NetFlow_z"]["coef"],
            "NetFlow_z_pval": lp_res["coefficients"]["NetFlow_z"]["p_value"],
            "NetFlow_z_tstat": lp_res["coefficients"]["NetFlow_z"]["t_stat"]
        }
        
    # 4. Granger Causality
    gc_data_rt = df_clean[['r_t', 'NetFlow_z']].values 
    gc_data_rv = df_clean[['rv_7d', 'NetFlow_z']].values
    
    granger_rt = {}
    granger_rv = {}
    
    try:
        gc_res_rt = grangercausalitytests(gc_data_rt, maxlag=7, verbose=False)
        for lag in range(1, 8):
            granger_rt[f"lag_{lag}"] = {
                "F_stat": float(gc_res_rt[lag][0]['ssr_ftest'][0]),
                "p_value": float(gc_res_rt[lag][0]['ssr_ftest'][1])
            }
            
        gc_res_rv = grangercausalitytests(gc_data_rv, maxlag=7, verbose=False)
        for lag in range(1, 8):
            granger_rv[f"lag_{lag}"] = {
                "F_stat": float(gc_res_rv[lag][0]['ssr_ftest'][0]),
                "p_value": float(gc_res_rv[lag][0]['ssr_ftest'][1])
            }
    except Exception as e:
        print(f"Granger causality failed: {e}")
        
    # 5. Robustness (Subperiods)
    periods = {
        "2017-2020": ("2017-01-01", "2020-12-31"),
        "2021-2023": ("2021-01-01", "2023-12-31"),
        "2024-2026": ("2024-01-01", "2026-12-31")
    }
    
    subperiod_results = {}
    for p_name, (start_d, end_d) in periods.items():
        sub_df = df_clean[(df_clean['time'] >= start_d) & (df_clean['time'] <= end_d)].copy()
        if len(sub_df) < 50:
            continue
            
        sub_ols_ret = fit_ols_hac('r_t', ['NetFlow_z_lag1', 'NetFlow_z_lag7', 'r_t_lag1', 'r_t_lag7'], sub_df)
        sub_ols_vol = fit_ols_hac('rv_7d', ['NetFlow_z_lag1', 'NetFlow_z_lag7', 'rv_7d_lag1'], sub_df)
        
        subperiod_results[p_name] = {
            "observations": len(sub_df),
            "returns_model": sub_ols_ret,
            "volatility_7d_model": sub_ols_vol
        }
        
    # 6. JSON structure
    output_json = {
        "paper2_netflows_v2": {
            "models": {
                "returns_hac_ols": ols_returns,
                "volatility_7d_hac_ols": ols_vol_7d,
                "volatility_30d_hac_ols": ols_vol_30d
            },
            "local_projections": local_projections_ret,
            "granger_causality": {
                "NetFlow_causes_Returns": granger_rt,
                "NetFlow_causes_Volatility": granger_rv
            },
            "subperiod_robustness": subperiod_results,
            "normalization": "NetFlowUSD / CapMrktCurUSD * 10000 (bps)",
            "sample_period": {
                "start": df_clean['time'].min().strftime('%Y-%m-%d'),
                "end": df_clean['time'].max().strftime('%Y-%m-%d'),
                "n_obs": len(df_clean)
            }
        }
    }
    
    out_file = os.path.join(results_dir, "paper2_netflows_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        
    # 7. Console Output
    print("\n" + "=" * 95)
    print("  PAPER 2 V2 NETFLOWS INFERENCE (Chi, Chu, Hao 2024 Re-estimation)")
    print("=" * 95)
    print("Baseline Returns Model (HAC SE):")
    for var, m in ols_returns['coefficients'].items():
        print(f"  {var:<20}: {m['coef']:+9.4f} (p={m['p_value']:.4f})")
        
    print("\nBaseline Volatility 7D Model (HAC SE):")
    for var, m in ols_vol_7d['coefficients'].items():
        print(f"  {var:<20}: {m['coef']:+9.4f} (p={m['p_value']:.4f})")
        
    print("\nLocal Projections (Returns h=1..7) - NetFlow_z effect:")
    for h, m in local_projections_ret.items():
        print(f"  {h:<10}: coef={m['NetFlow_z_coef']:+9.4f}, p={m['NetFlow_z_pval']:.4f}")
        
    print("=" * 95)
    print(f"Results successfully saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper2_netflows_v2()
