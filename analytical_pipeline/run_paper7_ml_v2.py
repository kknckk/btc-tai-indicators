import os
import json
import numpy as np

# Patch NumPy int aliases for boruta compatibility
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef
)
from boruta import BorutaPy
import xgboost as xgb
import sys

# Ensure analytical_pipeline directory is in sys.path
sys.path.append(os.path.dirname(__file__))
from utils import load_merged_data

def run_paper7_ml_v2():
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
    
    price = df['PriceUSD']
    
    # 2. Technical Indicators (TA proxies)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    sma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    df['bollinger_pct_b'] = (price - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)
    
    df['r_t_roll7_mean'] = df['r_t'].rolling(7).mean()
    df['r_t_roll30_mean'] = df['r_t'].rolling(30).mean()
    df['r_t_roll7_std'] = df['r_t'].rolling(7).std()
    df['r_t_roll30_std'] = df['r_t'].rolling(30).std()
    
    df['mom_7d'] = (price - price.shift(7)) / (price.shift(7) + 1e-8)
    df['mom_30d'] = (price - price.shift(30)) / (price.shift(30) + 1e-8)
    
    # 3. Base Feature Candidates
    onchain_base = [
        'AdrActCnt', 'TxTfrValUSD', 'FeeTotUSD', 'RevUSD', 'HashRate',
        'NUPL', 'MVRV_Z', 'SOPR', 'NetFlowUSD', 'ShareExUSD',
        'PuellMultiple', 'CDD', 'STH_MVRV', 'LTH_MVRV', 'NVT'
    ]
    ta_base = [
        'r_t', 'rv_7d', 'rv_30d', 'rsi_14', 'macd', 'macd_signal',
        'bollinger_pct_b', 'r_t_roll7_mean', 'r_t_roll30_mean',
        'r_t_roll7_std', 'r_t_roll30_std', 'mom_7d', 'mom_30d'
    ]
    
    base_feats = [c for c in onchain_base + ta_base if c in df.columns and df[c].notnull().mean() > 0.50]
    
    # Add Lags
    for col in ['r_t', 'MVRV_Z', 'NUPL', 'SOPR', 'AdrActCnt', 'HashRate', 'rv_7d']:
        if col in df.columns:
            for lag in [1, 3, 7]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
                
    candidate_cols = [
        c for c in df.columns
        if c not in ['time', 'PriceUSD', 'PriceUSD_log', 'PriceBTC', 'dir_t1']
        and not c.startswith('target_')
    ]
    candidate_cols = [c for c in candidate_cols if df[c].dtype in [np.float64, np.int64] and df[c].notnull().mean() > 0.50]
    
    horizons = [1, 3, 7]
    horizon_results = {}
    
    for h in horizons:
        df[f'target_h{h}'] = (price.shift(-h) > price).astype(int)
        target_col = f'target_h{h}'
        
        clean_df = df.dropna(subset=candidate_cols + [target_col]).copy().reset_index(drop=True)
        
        X = clean_df[candidate_cols].values
        y = clean_df[target_col].values
        returns = clean_df['r_t'].values
        n = len(clean_df)
        
        # Split 80% train, 20% holdout test
        split_idx = int(n * 0.80)
        X_tr, y_tr = X[:split_idx], y[:split_idx]
        X_te, y_te = X[split_idx:], y[split_idx:]
        ret_te = returns[split_idx:]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        # 4. Feature Selection using Boruta
        rf_b = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
        boruta = BorutaPy(rf_b, n_estimators='auto', random_state=42, max_iter=10)
        boruta.fit(X_tr_s, y_tr)

        
        sel_mask = boruta.support_
        selection_method = "Boruta"
        
        if np.sum(sel_mask) < 3:
            selection_method = "Top20_RF_Importance"
            rf_fi = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr_s, y_tr)
            top20_idx = np.argsort(rf_fi.feature_importances_)[::-1][:20]
            sel_mask = np.zeros(len(candidate_cols), dtype=bool)
            sel_mask[top20_idx] = True
            
        selected_feature_names = [candidate_cols[i] for i in range(len(candidate_cols)) if sel_mask[i]]
        
        X_tr_sel = X_tr_s[:, sel_mask]
        X_te_sel = X_te_s[:, sel_mask]
        
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42),
            "SVC_RBF": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42)
        }
        
        model_results = {}
        for m_name, model in models.items():
            # Fit on Full vs Selected Features
            # Full feature evaluation
            model.fit(X_tr_s, y_tr)
            preds_full = model.predict(X_te_s)
            acc_full = accuracy_score(y_te, preds_full)
            f1_full = f1_score(y_te, preds_full, zero_division=0)
            
            # Selected feature evaluation
            model.fit(X_tr_sel, y_tr)
            preds_sel = model.predict(X_te_sel)
            
            acc = accuracy_score(y_te, preds_sel)
            prec = precision_score(y_te, preds_sel, zero_division=0)
            rec = recall_score(y_te, preds_sel, zero_division=0)
            f1 = f1_score(y_te, preds_sel, zero_division=0)
            bacc = balanced_accuracy_score(y_te, preds_sel)
            mcc = matthews_corrcoef(y_te, preds_sel)
            
            # Strategy Economic Return (10 bps cost)
            pos = preds_sel
            pos_diff = np.abs(np.diff(np.insert(pos, 0, 0)))
            cost = pos_diff * 0.0010
            strat_ret = pos * ret_te - cost
            cum_ret = float(np.sum(strat_ret))
            ann_ret = float(np.mean(strat_ret) * 365.0)
            ann_vol = float(np.std(strat_ret, ddof=1) * np.sqrt(365.0)) if len(strat_ret) > 1 else 0.0
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            
            model_results[m_name] = {
                "Accuracy": float(acc),
                "Accuracy_FullSet": float(acc_full),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1": float(f1),
                "F1_FullSet": float(f1_full),
                "BalancedAcc": float(bacc),
                "MCC": float(mcc),
                "StrategyCumReturn": float(cum_ret),
                "StrategySharpe": float(sharpe)
            }
            
        horizon_results[f"h_{h}"] = {
            "selection_method": selection_method,
            "selected_feature_count": len(selected_feature_names),
            "selected_features": selected_feature_names,
            "models": model_results
        }
        
    # Save output JSON
    output_json = {
        "paper7_classification_v2": {
            "dataset_info": {
                "total_candidate_features": len(candidate_cols),
                "horizons": horizons
            },
            "horizon_results": horizon_results
        }
    }
    
    out_file = os.path.join(results_dir, "paper7_classification_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        
    # Console Table Output
    print("\n" + "=" * 105)
    print("  PAPER 7 V2 BITCOIN PRICE DIRECTION & MAGNITUDE CLASSIFICATION (Omole & Enke, 2025)")
    print("=" * 105)
    
    for h in horizons:
        h_data = horizon_results[f"h_{h}"]
        print(f"\n--- Horizon h = {h} days (Feature Selection: {h_data['selection_method']}, Top {h_data['selected_feature_count']} features) ---")
        header = f"{'Model':<18} | {'Acc (Sel)':<9} | {'Acc (Full)':<10} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'MCC':<7} | {'Strat CumRet':<12}"
        print(header)
        print("-" * 105)
        
        for m_name, m in h_data["models"].items():
            row_str = (
                f"{m_name:<18} | {m['Accuracy']:9.4f} | {m['Accuracy_FullSet']:10.4f} | "
                f"{m['Precision']:9.4f} | {m['Recall']:8.4f} | {m['F1']:6.4f} | {m['MCC']:+7.4f} | "
                f"{m['StrategyCumReturn']:+12.4f}"
            )
            print(row_str)
            
    print("=" * 105)
    print(f"Results successfully saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper7_ml_v2()
