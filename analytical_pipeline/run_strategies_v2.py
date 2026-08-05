import os
import json
import pandas as pd
import numpy as np
from utils import load_merged_data

def run_strategies_v2():
    # 1. Load Data
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    csv_path = os.path.join(results_dir, "derived_metrics.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_merged_data()
        
    req_cols = ['time', 'PriceUSD', 'r_t', 'MVRV_Z', 'NUPL']
    missing = [col for col in req_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
        
    # Drop rows with NaN in required columns & sort ascending
    df_clean = df.dropna(subset=req_cols).copy()
    df_clean['time'] = pd.to_datetime(df_clean['time'])
    df_clean = df_clean.sort_values('time').reset_index(drop=True)
    
    # Calculate daily simple returns R_t from PriceUSD to avoid log-return compounding discrepancies
    df_clean['R_t'] = df_clean['PriceUSD'].pct_change().fillna(0.0)
    
    n = len(df_clean)
    mvrv = df_clean['MVRV_Z'].values
    nupl = df_clean['NUPL'].values
    r_t = df_clean['R_t'].values
    times = df_clean['time'].dt.strftime('%Y-%m-%d').values
    dates = df_clean['time'].values
    
    # 2. Signal Generation (Zero look-ahead bias)
    # Signal at end of day t is computed using history strictly up to t-1 for quantiles,
    # and evaluated against day t's indicator values.
    
    min_history_days = 365
    rolling_window_days = 1460
    
    def generate_expanding_signals():
        sig = np.zeros(n, dtype=int)
        curr_pos = 0
        for t in range(n):
            if t >= min_history_days:
                h_mvrv = mvrv[:t] # strictly [0..t-1]
                h_nupl = nupl[:t]
                q_mvrv_low, q_mvrv_high = np.quantile(h_mvrv, [0.20, 0.80])
                q_nupl_low, q_nupl_high = np.quantile(h_nupl, [0.20, 0.80])
                
                if mvrv[t] < q_mvrv_low and nupl[t] < q_nupl_low:
                    curr_pos = 1
                elif mvrv[t] > q_mvrv_high or nupl[t] > q_nupl_high:
                    curr_pos = 0
            sig[t] = curr_pos
        return sig

    def generate_rolling_signals():
        sig = np.zeros(n, dtype=int)
        curr_pos = 0
        for t in range(n):
            if t >= rolling_window_days:
                h_mvrv = mvrv[t - rolling_window_days : t] # strictly [t-1460..t-1]
                h_nupl = nupl[t - rolling_window_days : t]
                q_mvrv_low, q_mvrv_high = np.quantile(h_mvrv, [0.20, 0.80])
                q_nupl_low, q_nupl_high = np.quantile(h_nupl, [0.20, 0.80])
                
                if mvrv[t] < q_mvrv_low and nupl[t] < q_nupl_low:
                    curr_pos = 1
                elif mvrv[t] > q_mvrv_high or nupl[t] > q_nupl_high:
                    curr_pos = 0
            sig[t] = curr_pos
        return sig

    def generate_fixed_signals(mvrv_low=0.0, nupl_low=0.0, mvrv_high=5.0, nupl_high=0.75):
        sig = np.zeros(n, dtype=int)
        curr_pos = 0
        for t in range(n):
            if mvrv[t] < mvrv_low and nupl[t] < nupl_low:
                curr_pos = 1
            elif mvrv[t] > mvrv_high or nupl[t] > nupl_high:
                curr_pos = 0
            sig[t] = curr_pos
        return sig

    sig_expanding = generate_expanding_signals()
    sig_rolling = generate_rolling_signals()
    sig_fixed = generate_fixed_signals(0.0, 0.0, 5.0, 0.75)
    sig_fixed_mild = generate_fixed_signals(0.0, 0.0, 3.5, 0.70)
    sig_bh = np.ones(n, dtype=int)
    
    # 3. Position and Evaluation
    # Position held on day t is signal from close of day t-1: pos[t] = sig[t-1]
    # Evaluation period starts after min_history_days (index 365)
    start_idx = min_history_days
    eval_start_date = times[start_idx]
    
    def evaluate(sig_array, cost_bps=20):
        # Shift signal by 1 day to prevent price look-ahead
        pos = np.zeros(n, dtype=int)
        pos[1:] = sig_array[:-1]
        
        sub_pos = pos[start_idx:]
        sub_ret = r_t[start_idx:]
        sub_times = times[start_idx:]
        num_days = len(sub_pos)
        
        # Position changes and cost calculation
        pos_changes = np.zeros(num_days, dtype=int)
        pos_changes[0] = 1 if sub_pos[0] == 1 else 0
        pos_changes[1:] = np.abs(np.diff(sub_pos))
        
        cost = pos_changes * (cost_bps / 10000.0)
        strat_ret = sub_pos * sub_ret - cost
        
        # Equity curve starting at 1.0
        eq = np.cumprod(1.0 + strat_ret)
        bh_eq = np.cumprod(1.0 + sub_ret)
        
        years = num_days / 365.0
        cagr = float((eq[-1]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
        
        ann_ret = float(np.mean(strat_ret) * 365.0)
        ann_vol = float(np.std(strat_ret, ddof=1) * np.sqrt(365.0)) if num_days > 1 else 0.0
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        
        downside = strat_ret[strat_ret < 0]
        down_vol = float(np.std(downside, ddof=1) * np.sqrt(365.0)) if len(downside) > 1 else 1e-6
        sortino = float(ann_ret / down_vol) if down_vol > 0 else 0.0
        
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max
        max_dd = float(np.min(dd))
        
        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0
        transactions = int(np.sum(pos_changes))
        
        # Calculate continuous long holding periods
        long_blocks = []
        curr_len = 0
        for p in sub_pos:
            if p == 1:
                curr_len += 1
            else:
                if curr_len > 0:
                    long_blocks.append(curr_len)
                    curr_len = 0
        if curr_len > 0:
            long_blocks.append(curr_len)
            
        avg_holding_period = float(np.mean(long_blocks)) if len(long_blocks) > 0 else 0.0
        pct_time_in_market = float(np.mean(sub_pos) * 100.0)
        
        metrics = {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "transactions": transactions,
            "avg_holding_period_days": avg_holding_period,
            "pct_time_in_market": pct_time_in_market
        }
        
        equity_curve = []
        for i in range(num_days):
            equity_curve.append({
                "time": sub_times[i],
                "strategy_value": float(eq[i]),
                "bh_value": float(bh_eq[i]),
                "position": int(sub_pos[i])
            })
            
        return metrics, equity_curve

    # Evaluate variants with 20 bps cost and 0 bps cost
    res_exp, eq_exp = evaluate(sig_expanding, cost_bps=20)
    res_exp_nocost, _ = evaluate(sig_expanding, cost_bps=0)
    
    res_roll, eq_roll = evaluate(sig_rolling, cost_bps=20)
    res_roll_nocost, _ = evaluate(sig_rolling, cost_bps=0)
    
    res_fix, eq_fix = evaluate(sig_fixed, cost_bps=20)
    res_fix_nocost, _ = evaluate(sig_fixed, cost_bps=0)
    
    res_fix_mild, eq_fix_mild = evaluate(sig_fixed_mild, cost_bps=20)
    res_fix_mild_nocost, _ = evaluate(sig_fixed_mild, cost_bps=0)
    
    res_bh, _ = evaluate(sig_bh, cost_bps=0)

    # 4. Robustness by period (on Expanding window, the primary out-of-sample strategy)
    pos_exp_full = np.zeros(n, dtype=int)
    pos_exp_full[1:] = sig_expanding[:-1]
    
    periods = {
        "2013-01-01_to_2017-12-31": (dates >= pd.to_datetime('2013-01-01')) & (dates <= pd.to_datetime('2017-12-31')),
        "2018-01-01_to_2020-12-31": (dates >= pd.to_datetime('2018-01-01')) & (dates <= pd.to_datetime('2020-12-31')),
        "2021-01-01_to_end": (dates >= pd.to_datetime('2021-01-01'))
    }
    
    robustness_res = {}
    for name, mask in periods.items():
        sub_p = pos_exp_full[mask]
        sub_r = r_t[mask]
        
        pos_c = np.zeros(len(sub_p), dtype=int)
        pos_c[0] = 1 if sub_p[0] == 1 else 0
        pos_c[1:] = np.abs(np.diff(sub_p))
        
        cost_p = pos_c * 0.002
        strat_r = sub_p * sub_r - cost_p
        
        eq_p = np.cumprod(1.0 + strat_r)
        bh_eq_p = np.cumprod(1.0 + sub_r)
        
        yrs = len(eq_p) / 365.0
        cagr_p = float((eq_p[-1]) ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
        cagr_bh_p = float((bh_eq_p[-1]) ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
        
        ann_r = float(np.mean(strat_r) * 365.0)
        ann_v = float(np.std(strat_r, ddof=1) * np.sqrt(365.0)) if len(strat_r) > 1 else 0.0
        sh = float(ann_r / ann_v) if ann_v > 0 else 0.0
        
        ann_r_bh = float(np.mean(sub_r) * 365.0)
        ann_v_bh = float(np.std(sub_r, ddof=1) * np.sqrt(365.0)) if len(sub_r) > 1 else 0.0
        sh_bh = float(ann_r_bh / ann_v_bh) if ann_v_bh > 0 else 0.0
        
        pk = np.maximum.accumulate(eq_p)
        mdd = float(np.min((eq_p - pk) / pk))
        
        pk_bh = np.maximum.accumulate(bh_eq_p)
        mdd_bh = float(np.min((bh_eq_p - pk_bh) / pk_bh))
        
        robustness_res[name] = {
            "expanding_strategy": {
                "cagr": cagr_p,
                "sharpe": sh,
                "max_drawdown": mdd
            },
            "buy_and_hold": {
                "cagr": cagr_bh_p,
                "sharpe": sh_bh,
                "max_drawdown": mdd_bh
            }
        }

    # 5. Output JSON structure
    output_json = {
        "paper4_v2": {
            "expanding": {
                "metrics": res_exp,
                "metrics_no_cost": res_exp_nocost,
                "equity_curve": eq_exp
            },
            "rolling_4y": {
                "metrics": res_roll,
                "metrics_no_cost": res_roll_nocost,
                "equity_curve": eq_roll
            },
            "fixed_literature": {
                "metrics": res_fix,
                "metrics_no_cost": res_fix_nocost,
                "equity_curve": eq_fix
            },
            "fixed_literature_mild": {
                "metrics": res_fix_mild,
                "metrics_no_cost": res_fix_mild_nocost,
                "equity_curve": eq_fix_mild
            },
            "buy_and_hold": {
                "metrics": res_bh
            },
            "robustness_by_period": robustness_res,
            "parameters": {
                "cost_bps": 20,
                "min_history_days": min_history_days,
                "rolling_window_days": rolling_window_days,
                "eval_start_date": eval_start_date,
                "eval_end_date": times[-1]
            }
        }
    }

    out_file = os.path.join(results_dir, "strategy_results_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    # 6. Summary console printout
    print("\n" + "=" * 90)
    print(f"  PAPER 4 V2 STRATEGY BACKTEST RESULTS (No Look-Ahead Bias)  [{eval_start_date} to {times[-1]}]")
    print("=" * 90)
    header = f"{'Strategy Variant':<25} | {'CAGR':<8} | {'Sharpe':<7} | {'Sortino':<8} | {'MaxDD':<8} | {'Calmar':<7} | {'Trades':<6} | {'Market%':<7}"
    print(header)
    print("-" * 90)
    
    table_rows = [
        ("Buy & Hold", res_bh),
        ("Expanding Quantiles", res_exp),
        ("Rolling 4Y Quantiles", res_roll),
        ("Fixed Lit (5.0/0.75)", res_fix),
        ("Fixed Lit (3.5/0.70)", res_fix_mild),
    ]
    
    for label, m in table_rows:
        row_str = f"{label:<25} | {m['cagr']*100:6.2f}% | {m['sharpe']:7.3f} | {m['sortino']:8.3f} | {m['max_drawdown']*100:7.2f}% | {m['calmar']:7.3f} | {m['transactions']:<6} | {m['pct_time_in_market']:6.1f}%"
        print(row_str)
    print("=" * 90)
    print(f"Results saved to: {out_file}\n")

if __name__ == "__main__":
    run_strategies_v2()
