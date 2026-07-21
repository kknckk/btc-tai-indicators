import os
import json
import pandas as pd
import numpy as np
from utils import load_merged_data, save_results

def run_strategies(df: pd.DataFrame):
    if 'MVRV_Z' not in df.columns or 'NUPL' not in df.columns or 'r_t' not in df.columns:
        print("Brak kolumn MVRV_Z, NUPL lub r_t.")
        return
        
    df_clean = df.dropna(subset=['r_t', 'MVRV_Z', 'NUPL']).copy()
    
    # Progi na podstawie kwantyli
    mvrv_low = df_clean['MVRV_Z'].quantile(0.20)
    mvrv_high = df_clean['MVRV_Z'].quantile(0.80)
    
    nupl_low = df_clean['NUPL'].quantile(0.20)
    nupl_high = df_clean['NUPL'].quantile(0.80)
    
    # Sygnały strategii
    # Wejście gdy wyprzedane, wyjście gdy wykupione. 
    # Stan 1 to Long, 0 to Flat (wyszliśmy z rynku)
    
    signals = []
    transactions = 0
    current_position = 1 # Zakładamy buy and hold na start
    for idx, row in df_clean.iterrows():
        prev_pos = current_position
        if row['MVRV_Z'] < mvrv_low and row['NUPL'] < nupl_low:
            current_position = 1
        elif row['MVRV_Z'] > mvrv_high or row['NUPL'] > nupl_high:
            current_position = 0
            
        if current_position != prev_pos:
            transactions += 1
            
        signals.append(current_position)
        
    df_clean['strategy_pos'] = signals
    df_clean['strategy_return'] = df_clean['strategy_pos'].shift(1) * df_clean['r_t']
    
    # Symulacja equity (log return cumulative sum)
    df_clean['buy_hold_equity'] = df_clean['r_t'].cumsum()
    df_clean['strategy_equity'] = df_clean['strategy_return'].cumsum()
    
    # Przelicz do standardowego equity startującego od 1.0
    df_clean['buy_hold_value'] = np.exp(df_clean['buy_hold_equity'])
    df_clean['strategy_value'] = np.exp(df_clean['strategy_equity'])
    
    def calculate_metrics(returns_series):
        returns = returns_series.dropna()
        if len(returns) == 0:
            return 0, 0, 0, 0
        annual_ret = returns.mean() * 365
        annual_vol = returns.std() * np.sqrt(365)
        sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
        
        # Sortino (tylko ujemne zwroty)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 1e-6
        sortino = annual_ret / downside_vol if downside_vol != 0 else 0
        
        cum_ret = np.exp(returns.cumsum())
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # CAGR (Compound Annual Growth Rate)
        years = len(returns) / 365.0
        cagr = (cum_ret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        
        return float(sharpe), float(max_dd), float(sortino), float(cagr)
        
    sharpe_bh, dd_bh, sortino_bh, cagr_bh = calculate_metrics(df_clean['r_t'])
    sharpe_strat, dd_strat, sortino_strat, cagr_strat = calculate_metrics(df_clean['strategy_return'])
    
    ts_data = df_clean[['time', 'buy_hold_value', 'strategy_value', 'MVRV_Z', 'NUPL', 'strategy_pos']].tail(365*4).copy()
    ts_data['time'] = ts_data['time'].dt.strftime('%Y-%m-%d')
    
    return {
        "metrics": {
            "buy_and_hold": {"sharpe": sharpe_bh, "max_drawdown": dd_bh, "sortino": sortino_bh, "cagr": cagr_bh, "trades": 0},
            "strategy": {"sharpe": sharpe_strat, "max_drawdown": dd_strat, "sortino": sortino_strat, "cagr": cagr_strat, "trades": transactions}
        },
        "thresholds": {
            "mvrv_low": float(mvrv_low), "mvrv_high": float(mvrv_high),
            "nupl_low": float(nupl_low), "nupl_high": float(nupl_high)
        },
        "time_series": ts_data.to_dict(orient='records')
    }

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    df_path = os.path.join(results_dir, "derived_metrics.csv")
    if not os.path.exists(df_path):
        df_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_ingestion", "csv")
        df = load_merged_data() # fallback
    else:
        df = pd.read_csv(df_path)
        
    df['time'] = pd.to_datetime(df['time'])
    
    results = {}
    
    try:
        results['paper4'] = run_strategies(df)
        print("Paper 4 (Strategies) OK")
    except Exception as e:
        print(f"Paper 4 Error: {e}")
        
    with open(os.path.join(results_dir, "strategy_results.json"), "w") as f:
        json.dump(results, f, default=str)
    
    print("Wyniki strategii zapisane.")

if __name__ == "__main__":
    main()
