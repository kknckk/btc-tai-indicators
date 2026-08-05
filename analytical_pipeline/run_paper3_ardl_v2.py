import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, breaks_cusumolsresid
from statsmodels.stats.stattools import jarque_bera
from utils import load_merged_data

def run_paper3_ardl_v2():
    # 1. Load and Resample Data
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    csv_path = os.path.join(results_dir, "derived_metrics.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_merged_data()
        
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df.set_index('time', inplace=True)
    
    # Resample to weekly frequency (W-SUN)
    # PriceUSD: last value of week (end-of-week close)
    # Activity metrics & ratios: weekly mean (smooths weekend noise)
    df_w = pd.DataFrame()
    df_w['PriceUSD'] = df['PriceUSD'].resample('W-SUN').last()
    
    if 'NUPL' in df.columns:
        df_w['NUPL'] = df['NUPL'].resample('W-SUN').mean()
    else:
        df_w['NUPL'] = df['SOPR'].resample('W-SUN').mean()
        
    df_w['AdrActCnt'] = df['AdrActCnt'].resample('W-SUN').mean()
    
    if 'RevUSD' in df.columns and 'IssTotUSD' in df.columns:
        df_w['RevUSD'] = df['RevUSD'].resample('W-SUN').mean()
        df_w['IssTotUSD'] = df['IssTotUSD'].resample('W-SUN').mean()
        df_w['PM_proxy'] = df_w['RevUSD'] / df_w['IssTotUSD']
    elif 'PM_proxy' in df.columns:
        df_w['PM_proxy'] = df['PM_proxy'].resample('W-SUN').mean()
    else:
        df_w['PM_proxy'] = df_w['NUPL'] # fallback
        
    df_w['Y'] = np.log(df_w['PriceUSD'])
    df_w['log_AdrActCnt'] = np.log(df_w['AdrActCnt'])
    
    req_vars = ['Y', 'NUPL', 'PM_proxy', 'log_AdrActCnt']
    
    # Primary sample: Post 2012-01-01 (Kalkan & Tatlı, 2022)
    data = df_w.loc['2012-01-01':].dropna(subset=req_vars).copy()
    
    # 2. Stationarity Tests (ADF & KPSS)
    adf_kpss_res = {}
    for col in req_vars:
        series = data[col]
        diff_series = series.diff().dropna()
        
        adf_lev = adfuller(series, autolag='AIC')
        adf_diff = adfuller(diff_series, autolag='AIC')
        
        kpss_lev = kpss(series, regression='c', nlags='auto')
        kpss_diff = kpss(diff_series, regression='c', nlags='auto')
        
        # Integration order logic
        is_stat_lev = (adf_lev[1] < 0.05) and (kpss_lev[1] >= 0.05)
        is_stat_diff = (adf_diff[1] < 0.05)
        
        if is_stat_lev:
            order = "I(0)"
        elif is_stat_diff:
            order = "I(1)"
        else:
            order = "I(1)" # boundary
            
        adf_kpss_res[col] = {
            "adf_level_p": float(adf_lev[1]),
            "adf_diff_p": float(adf_diff[1]),
            "kpss_level_p": float(kpss_lev[1]),
            "kpss_diff_p": float(kpss_diff[1]),
            "integration_order": order
        }
        
    # 3. ARDL Lag Selection via AIC (max_lag = 4 for weekly grid search)
    best_aic = float('inf')
    best_lags = (1, 1, 1, 1)
    
    for p in range(1, 5):
        for q1 in range(0, 4):
            for q2 in range(0, 4):
                for q3 in range(0, 4):
                    df_reg = pd.DataFrame(index=data.index)
                    df_reg['Y'] = data['Y']
                    
                    for i in range(1, p + 1):
                        df_reg[f'Y_lag{i}'] = data['Y'].shift(i)
                    for j in range(0, q1 + 1):
                        df_reg[f'X1_lag{j}'] = data['NUPL'].shift(j)
                    for j in range(0, q2 + 1):
                        df_reg[f'X2_lag{j}'] = data['PM_proxy'].shift(j)
                    for j in range(0, q3 + 1):
                        df_reg[f'X3_lag{j}'] = data['log_AdrActCnt'].shift(j)
                        
                    df_reg_clean = df_reg.dropna()
                    X_mat = sm.add_constant(df_reg_clean[[c for c in df_reg_clean.columns if c != 'Y']])
                    y_vec = df_reg_clean['Y']
                    
                    res = sm.OLS(y_vec, X_mat).fit()
                    if res.aic < best_aic:
                        best_aic = float(res.aic)
                        best_lags = (p, q1, q2, q3)
                        
    p_opt, q1_opt, q2_opt, q3_opt = best_lags
    
    # 4. Pesaran, Shin, Smith (2001) Bounds Test (Case III)
    # Conditional Unrestricted ECM
    df_uecm = pd.DataFrame(index=data.index)
    df_uecm['dY'] = data['Y'].diff()
    df_uecm['Y_lag1'] = data['Y'].shift(1)
    df_uecm['X1_lag1'] = data['NUPL'].shift(1)
    df_uecm['X2_lag1'] = data['PM_proxy'].shift(1)
    df_uecm['X3_lag1'] = data['log_AdrActCnt'].shift(1)
    
    for i in range(1, p_opt):
        df_uecm[f'dY_lag{i}'] = data['Y'].diff().shift(i)
    for j in range(0, q1_opt):
        df_uecm[f'dX1_lag{j}'] = data['NUPL'].diff().shift(j)
    for j in range(0, q2_opt):
        df_uecm[f'dX2_lag{j}'] = data['PM_proxy'].diff().shift(j)
    for j in range(0, q3_opt):
        df_uecm[f'dX3_lag{j}'] = data['log_AdrActCnt'].diff().shift(j)
        
    df_uecm_clean = df_uecm.dropna()
    y_uecm = df_uecm_clean['dY']
    X_uecm_cols = [c for c in df_uecm_clean.columns if c != 'dY']
    X_uecm_mat = sm.add_constant(df_uecm_clean[X_uecm_cols])
    
    res_uecm = sm.OLS(y_uecm, X_uecm_mat).fit()
    
    # Joint Wald Test H0: lambda_Y = lambda_X1 = lambda_X2 = lambda_X3 = 0
    level_vars = ['Y_lag1', 'X1_lag1', 'X2_lag1', 'X3_lag1']
    r_matrix = np.zeros((4, len(res_uecm.params)))
    for row_idx, var_name in enumerate(level_vars):
        col_idx = X_uecm_mat.columns.get_loc(var_name)
        r_matrix[row_idx, col_idx] = 1.0
        
    f_test_res = res_uecm.f_test(r_matrix)
    f_stat = float(f_test_res.fvalue)
    
    # Critical values for Case III (k=3 regressors) from Pesaran et al. (2001)
    i0_5, i1_5 = 3.23, 4.35
    i0_1, i1_1 = 4.29, 5.61
    
    is_cointegrated = f_stat > i1_5
    decision_str = "COINTEGRATION CONFIRMED (F > I(1) bound at 5%)" if is_cointegrated else "NO COINTEGRATION"
    
    # 5. Long-Run Multipliers & Error Correction Model (ECM)
    lambda_Y = res_uecm.params['Y_lag1']
    lambda_X1 = res_uecm.params['X1_lag1']
    lambda_X2 = res_uecm.params['X2_lag1']
    lambda_X3 = res_uecm.params['X3_lag1']
    alpha_const = res_uecm.params['const']
    
    theta_0 = - alpha_const / lambda_Y
    theta_1 = - lambda_X1 / lambda_Y
    theta_2 = - lambda_X2 / lambda_Y
    theta_3 = - lambda_X3 / lambda_Y
    
    # p-values for long-run multipliers via t-test approximation
    p_X1 = float(res_uecm.pvalues['X1_lag1'])
    p_X2 = float(res_uecm.pvalues['X2_lag1'])
    p_X3 = float(res_uecm.pvalues['X3_lag1'])
    
    long_run_multipliers = {
        "NUPL": {"coef": float(theta_1), "p_value": p_X1, "sign": "+" if theta_1 > 0 else "-"},
        "PM_proxy": {"coef": float(theta_2), "p_value": p_X2, "sign": "+" if theta_2 > 0 else "-"},
        "log_AdrActCnt": {"coef": float(theta_3), "p_value": p_X3, "sign": "+" if theta_3 > 0 else "-"}
    }
    
    # Construct ECT term and fit restricted ECM
    ect_series = data['Y'] - theta_0 - theta_1 * data['NUPL'] - theta_2 * data['PM_proxy'] - theta_3 * data['log_AdrActCnt']
    
    df_ecm_re = pd.DataFrame(index=data.index)
    df_ecm_re['dY'] = data['Y'].diff()
    df_ecm_re['ECT_lag1'] = ect_series.shift(1)
    
    for i in range(1, p_opt):
        df_ecm_re[f'dY_lag{i}'] = data['Y'].diff().shift(i)
    for j in range(0, q1_opt):
        df_ecm_re[f'dX1_lag{j}'] = data['NUPL'].diff().shift(j)
    for j in range(0, q2_opt):
        df_ecm_re[f'dX2_lag{j}'] = data['PM_proxy'].diff().shift(j)
    for j in range(0, q3_opt):
        df_ecm_re[f'dX3_lag{j}'] = data['log_AdrActCnt'].diff().shift(j)
        
    df_ecm_re_clean = df_ecm_re.dropna()
    X_ecm_re_mat = sm.add_constant(df_ecm_re_clean[[c for c in df_ecm_re_clean.columns if c != 'dY']])
    res_ecm = sm.OLS(df_ecm_re_clean['dY'], X_ecm_re_mat).fit()
    
    ect_coef = float(res_ecm.params['ECT_lag1'])
    ect_pval = float(res_ecm.pvalues['ECT_lag1'])
    
    short_run_coefs = {col: float(res_ecm.params[col]) for col in X_ecm_re_mat.columns if col != 'const'}
    
    ecm_results = {
        "ect_coefficient": ect_coef,
        "ect_p_value": ect_pval,
        "speed_of_adjustment_pct": float(abs(ect_coef) * 100.0),
        "r_squared": float(res_ecm.rsquared),
        "adj_r_squared": float(res_ecm.rsquared_adj)
    }
    
    # 6. Toda-Yamamoto Granger Causality Test
    var_data = data[['Y', 'NUPL', 'PM_proxy', 'log_AdrActCnt']].dropna()
    var_model = VAR(var_data)
    var_k = int(var_model.select_order(maxlags=8).selected_orders['aic'])
    ty_lags = var_k + 1
    var_fit = var_model.fit(ty_lags)
    
    toda_yamamoto_res = {}
    for reg in ['NUPL', 'PM_proxy', 'log_AdrActCnt']:
        ctest = var_fit.test_causality(caused='Y', causing=reg, signif=0.05)
        toda_yamamoto_res[reg] = {
            "stat": float(ctest.test_statistic),
            "p_value": float(ctest.pvalue),
            "conclusion": "CAUSES Y" if ctest.pvalue < 0.05 else "DOES NOT CAUSE Y"
        }

    # 7. Diagnostics
    bg_lm, bg_p, _, _ = acorr_breusch_godfrey(res_uecm, nlags=4)
    bp_lm, bp_p, _, _ = het_breuschpagan(res_uecm.resid, res_uecm.model.exog)
    jb_stat, jb_p, _, _ = jarque_bera(res_uecm.resid)
    cusum_res = breaks_cusumolsresid(res_uecm.resid)
    cusum_stat, cusum_p = float(cusum_res[0]), float(cusum_res[1])
    
    diagnostics_res = {
        "breusch_godfrey": {"stat": float(bg_lm), "p_value": float(bg_p), "pass": bool(bg_p > 0.05)},
        "breusch_pagan": {"stat": float(bp_lm), "p_value": float(bp_p), "pass": bool(bp_p > 0.05)},
        "jarque_bera": {"stat": float(jb_stat), "p_value": float(jb_p), "pass": bool(jb_p > 0.05)},
        "cusum_stability": {"stat": cusum_stat, "p_value": cusum_p, "stable": bool(cusum_p > 0.05)}
    }
    
    # 8. Output JSON Structure
    output_json = {
        "paper3_ardl_v2": {
            "sample_period": f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
            "sample_frequency": "Weekly (W-SUN)",
            "adf_kpss": adf_kpss_res,
            "selected_lags": {
                "p": p_opt,
                "q_NUPL": q1_opt,
                "q_PM_proxy": q2_opt,
                "q_log_AdrActCnt": q3_opt,
                "criterion": "AIC",
                "aic": best_aic
            },
            "bounds_test": {
                "F_stat": f_stat,
                "I0_bound_5pct": i0_5,
                "I1_bound_5pct": i1_5,
                "I0_bound_1pct": i0_1,
                "I1_bound_1pct": i1_1,
                "decision": decision_str,
                "cointegration": bool(is_cointegrated)
            },
            "long_run_multipliers": long_run_multipliers,
            "ecm_results": ecm_results,
            "toda_yamamoto": toda_yamamoto_res,
            "diagnostics": diagnostics_res,
            "short_run_coefficients": short_run_coefs
        }
    }
    
    out_file = os.path.join(results_dir, "paper3_ardl_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        
    # 9. Console Output
    print("\n" + "=" * 90)
    print("  PAPER 3 V2 ARDL BOUNDS TEST & COINTEGRATION RESULTS (Kalkan & Tatlı, 2022 Re-estimation)")
    print("=" * 90)
    print(f"Próba: {data.index.min().strftime('%Y-%m-%d')} – {data.index.max().strftime('%Y-%m-%d')} (Tygodniowa, N={len(data)})")
    print(f"Optymalne opóźnienia ARDL(p, q1, q2, q3): ARDL({p_opt}, {q1_opt}, {q2_opt}, {q3_opt}) | AIC: {best_aic:.2f}")
    print("-" * 90)
    print(f"Pesaran Bounds Test F-Stat: {f_stat:.4f}  (Wartości krytyczne 5%: I(0)={i0_5}, I(1)={i1_5})")
    print(f"Kointegracja: {'TAK (Istnieje relacja długookresowa)' if is_cointegrated else 'NIE'}")
    print("-" * 90)
    print("Long-Run Multipliers (Współczynniki długookresowe):")
    for var, m in long_run_multipliers.items():
        sig_stars = "***" if m['p_value'] < 0.01 else ("**" if m['p_value'] < 0.05 else ("*" if m['p_value'] < 0.1 else ""))
        print(f"  - {var:<15}: {m['coef']:+9.4f}  (p-val: {m['p_value']:.4e}) {sig_stars}")
    print("-" * 90)
    print(f"Error Correction Term (ECT_t-1): {ect_coef:+.4f} (p-val: {ect_pval:.4e})")
    print(f"Diagnostyka: Breusch-Godfrey p-val={bg_p:.4f}, Breusch-Pagan p-val={bp_p:.4f}, CUSUM Stable={'TAK' if cusum_p > 0.05 else 'NIE'}")
    print("=" * 90)
    print(f"Wyniki zapisane do: {out_file}\n")

if __name__ == "__main__":
    run_paper3_ardl_v2()
