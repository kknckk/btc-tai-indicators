import os
import json
import sys
import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests

# Ensure analytical_pipeline directory is in sys.path
sys.path.append(os.path.dirname(__file__))
from utils import load_merged_data

def run_paper1_garch_v2():
    # 1. Load Data
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "derived_metrics.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_merged_data()
        
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 2. Select Candidate Network Activity Features
    activity_candidates = [
        'AdrActCnt', 'TxCntSec', 'FeeTotUSD', 'FeeMeanUSD',
        'HashRate', 'BlkSizeMeanByte', 'UTXOCnt', 'RevUSD'
    ]
    active_feats = [f for f in activity_candidates if f in df.columns and df[f].notnull().mean() > 0.50]
    
    df_clean = df.dropna(subset=['time', 'r_t'] + active_feats).copy().reset_index(drop=True)
    r_t_pct = df_clean['r_t'].values * 100.0
    times = df_clean['time'].dt.strftime('%Y-%m-%d').values
    
    # 3. Fit Baseline GARCH(1,1) Normal vs Student-t
    am_norm = arch_model(r_t_pct, vol='Garch', p=1, q=1, dist='normal')
    res_norm = am_norm.fit(disp='off')
    
    am_t = arch_model(r_t_pct, vol='Garch', p=1, q=1, dist='t')
    res_t = am_t.fit(disp='off')
    
    # Standardized residual diagnostics
    std_resid_norm = pd.Series(res_norm.std_resid).dropna()
    std_resid_t = pd.Series(res_t.std_resid).dropna()
    
    lb_norm_10 = float(acorr_ljungbox(std_resid_norm, lags=[10], return_df=True)['lb_pvalue'].values[0])
    lb_norm_20 = float(acorr_ljungbox(std_resid_norm, lags=[20], return_df=True)['lb_pvalue'].values[0])
    arch_norm_10 = float(het_arch(std_resid_norm, maxlag=10)[1])
    arch_norm_20 = float(het_arch(std_resid_norm, maxlag=20)[1])
    
    lb_t_10 = float(acorr_ljungbox(std_resid_t, lags=[10], return_df=True)['lb_pvalue'].values[0])
    lb_t_20 = float(acorr_ljungbox(std_resid_t, lags=[20], return_df=True)['lb_pvalue'].values[0])
    arch_t_10 = float(het_arch(std_resid_t, maxlag=10)[1])
    arch_t_20 = float(het_arch(std_resid_t, maxlag=20)[1])
    
    garch_comparison = {
        "Normal": {
            "AIC": float(res_norm.aic), "BIC": float(res_norm.bic),
            "omega": float(res_norm.params.get('omega', 0)),
            "alpha": float(res_norm.params.get('alpha[1]', 0)),
            "beta": float(res_norm.params.get('beta[1]', 0)),
            "persistence": float(res_norm.params.get('alpha[1]', 0) + res_norm.params.get('beta[1]', 0)),
            "diagnostics": {
                "ljung_box_p10": lb_norm_10, "ljung_box_p20": lb_norm_20,
                "arch_lm_p10": arch_norm_10, "arch_lm_p20": arch_norm_20
            }
        },
        "Student_t": {
            "AIC": float(res_t.aic), "BIC": float(res_t.bic),
            "omega": float(res_t.params.get('omega', 0)),
            "alpha": float(res_t.params.get('alpha[1]', 0)),
            "beta": float(res_t.params.get('beta[1]', 0)),
            "nu_degrees_freedom": float(res_t.params.get('nu', 0)),
            "persistence": float(res_t.params.get('alpha[1]', 0) + res_t.params.get('beta[1]', 0)),
            "diagnostics": {
                "ljung_box_p10": lb_t_10, "ljung_box_p20": lb_t_20,
                "arch_lm_p10": arch_t_10, "arch_lm_p20": arch_t_20
            }
        }
    }
    
    # 4. HAC (Newey-West) Regression of Conditional Variance h_t on Network Activity Lags
    h_t = res_t.conditional_volatility ** 2
    reg_df = pd.DataFrame({"h_t": h_t})
    
    for feat in active_feats:
        s = np.log1p(df_clean[feat].values) if df_clean[feat].min() >= 0 else df_clean[feat].values
        reg_df[f"{feat}_lag1"] = pd.Series(s).shift(1)
        reg_df[f"{feat}_lag7"] = pd.Series(s).shift(7)
        
    reg_clean = reg_df.dropna().copy()
    y_reg = reg_clean["h_t"]
    X_cols = [c for c in reg_clean.columns if c != "h_t"]
    X_mat = sm.add_constant(reg_clean[X_cols])
    
    ols_hac = sm.OLS(y_reg, X_mat).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    
    p_vals = ols_hac.pvalues.drop('const')
    params = ols_hac.params.drop('const')
    bse_hac = ols_hac.bse.drop('const')
    t_stats = ols_hac.tvalues.drop('const')
    
    reject_bonf, p_bonf, _, _ = multipletests(p_vals.values, alpha=0.05, method='bonferroni')
    reject_bh, p_bh, _, _ = multipletests(p_vals.values, alpha=0.05, method='fdr_bh')
    
    hac_regression_results = {}
    for idx, var_name in enumerate(X_cols):
        hac_regression_results[var_name] = {
            "coef": float(params[var_name]),
            "HAC_se": float(bse_hac[var_name]),
            "t_stat": float(t_stats[var_name]),
            "p_value": float(p_vals[var_name]),
            "p_bonferroni": float(p_bonf[idx]),
            "sig_bonferroni": bool(reject_bonf[idx]),
            "p_bh": float(p_bh[idx]),
            "sig_bh": bool(reject_bh[idx])
        }

    # 5. GARCH-X Specification (Exogenous network activity in variance equation)
    # Using log(AdrActCnt)_lag1 as key exogenous activity proxy
    s_act_lag1 = pd.Series(np.log1p(df_clean['AdrActCnt'])).shift(1).dropna().values
    r_t_garch_x = r_t_pct[1:]
    
    am_gx = arch_model(r_t_garch_x, x=s_act_lag1.reshape(-1, 1), vol='Garch', p=1, q=1, dist='t')
    res_gx = am_gx.fit(disp='off')
    
    garch_x_results = {
        "exogenous_variable": "log1p(AdrActCnt)_lag1",
        "AIC": float(res_gx.aic),
        "BIC": float(res_gx.bic),
        "omega": float(res_gx.params.get('omega', 0)),
        "alpha": float(res_gx.params.get('alpha[1]', 0)),
        "beta": float(res_gx.params.get('beta[1]', 0)),
        "nu_degrees_freedom": float(res_gx.params.get('nu', 0))
    }
    
    # 6. Sub-period Robustness (2013-2017, 2018-2020, 2021-2026)
    periods = {
        "2013-2017": ("2013-01-01", "2017-12-31"),
        "2018-2020": ("2018-01-01", "2020-12-31"),
        "2021-2026": ("2021-01-01", "2026-05-24")
    }
    
    subperiod_results = {}
    for p_name, (start_d, end_d) in periods.items():
        sub = df_clean[(df_clean['time'] >= start_d) & (df_clean['time'] <= end_d)].copy()
        if len(sub) < 100:
            continue
            
        r_sub = sub['r_t'].values * 100.0
        am_sub = arch_model(r_sub, vol='Garch', p=1, q=1, dist='t')
        res_sub = am_sub.fit(disp='off')
        
        h_sub = res_sub.conditional_volatility ** 2
        reg_sub_df = pd.DataFrame({"h_t": h_sub})
        
        for f in active_feats:
            s_val = np.log1p(sub[f].values) if sub[f].min() >= 0 else sub[f].values
            reg_sub_df[f"{f}_lag1"] = pd.Series(s_val).shift(1)
            reg_sub_df[f"{f}_lag7"] = pd.Series(s_val).shift(7)
            
        reg_sub_clean = reg_sub_df.dropna().copy()
        y_sub_reg = reg_sub_clean["h_t"]
        X_sub_mat = sm.add_constant(reg_sub_clean[[c for c in reg_sub_clean.columns if c != "h_t"]])
        
        ols_sub = sm.OLS(y_sub_reg, X_sub_mat).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        
        sub_coefs = {}
        for var_k in [c for c in reg_sub_clean.columns if c != "h_t"]:
            sub_coefs[var_k] = {
                "coef": float(ols_sub.params[var_k]),
                "p_value": float(ols_sub.pvalues[var_k])
            }
            
        subperiod_results[p_name] = {
            "observations": len(sub),
            "garch_params": {
                "alpha": float(res_sub.params.get('alpha[1]', 0)),
                "beta": float(res_sub.params.get('beta[1]', 0)),
                "nu": float(res_sub.params.get('nu', 0))
            },
            "hac_ols_coefficients": sub_coefs
        }
        
    # Check sign stability of AdrActCnt_lag1 and FeeTotUSD_lag1 across sub-periods
    sign_stability = {}
    for check_var in ["AdrActCnt_lag1", "FeeTotUSD_lag1", "HashRate_lag1"]:
        signs = [np.sign(subperiod_results[p]["hac_ols_coefficients"][check_var]["coef"]) for p in subperiod_results]
        sign_stability[check_var] = {
            "signs_by_period": {p: float(signs[idx]) for idx, p in enumerate(subperiod_results)},
            "is_sign_stable": bool(len(set(signs)) == 1)
        }

    # 7. Build Output JSON
    output_json = {
        "paper1_garch_v2": {
            "garch_comparison": garch_comparison,
            "hac_regression_conditional_variance": hac_regression_results,
            "garch_x_results": garch_x_results,
            "subperiod_robustness": subperiod_results,
            "sign_stability": sign_stability,
            "sample_period": {
                "start": times[0],
                "end": times[-1],
                "n_obs": len(df_clean)
            }
        }
    }
    
    out_file = os.path.join(results_dir, "paper1_garch_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        
    # 8. Console Table Output
    print("\n" + "=" * 95)
    print("  PAPER 1 V2 GARCH VOLATILITY & NETWORK ACTIVITY INFERENCE (Wüstenfeld, 2023 Re-estimation)")
    print("=" * 95)
    print(f"Sample: {times[0]} to {times[-1]} (Daily, N={len(df_clean)})")
    print(f"GARCH(1,1) Normal  AIC: {res_norm.aic:.2f} | persistence: {res_norm.params.get('alpha[1]',0)+res_norm.params.get('beta[1]',0):.4f}")
    print(f"GARCH(1,1) Student-t AIC: {res_t.aic:.2f} | persistence: {res_t.params.get('alpha[1]',0)+res_t.params.get('beta[1]',0):.4f} | nu: {res_t.params.get('nu',0):.2f}")
    print("-" * 95)
    print(f"{'Variable':<20} | {'Coef':<9} | {'HAC SE':<9} | {'p-value':<9} | {'p-Bonf':<9} | {'Sig (Bonf)':<10} | {'Sig (BH)'}")
    print("-" * 95)
    
    for var_k, r in hac_regression_results.items():
        print(f"{var_k:<20} | {r['coef']:+9.4f} | {r['HAC_se']:9.4f} | {r['p_value']:9.4f} | {r['p_bonferroni']:9.4f} | {str(r['sig_bonferroni']):<10} | {str(r['sig_bh'])}")
        
    print("-" * 95)
    print("Sub-period Robustness (GARCH(1,1) Student-t & Coef Sign Stability):")
    for p_name, p_res in subperiod_results.items():
        gp = p_res['garch_params']
        print(f"  [{p_name} N={p_res['observations']}] GARCH alpha={gp['alpha']:.4f}, beta={gp['beta']:.4f}, nu={gp['nu']:.2f}")
    print("=" * 95)
    print(f"Results successfully saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper1_garch_v2()
