import os
import json
import numpy as np
import warnings

# Patch NumPy int aliases for boruta compatibility
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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
    warnings.filterwarnings('ignore')
    
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
    
    # 2. Technical Indicators (TA proxies) & Rolling
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
    
    if 'r_t' not in df.columns:
        df['r_t'] = price.pct_change()
        
    if 'rv_t' not in df.columns:
        df['rv_t'] = df['r_t'].rolling(7).std() # simple fallback
        
    df['r_t_roll7_mean'] = df['r_t'].rolling(7).mean()
    df['r_t_roll30_mean'] = df['r_t'].rolling(30).mean()
    df['r_t_roll7_std'] = df['r_t'].rolling(7).std()
    df['r_t_roll30_std'] = df['r_t'].rolling(30).std()
    
    df['rv_roll7_mean'] = df['rv_t'].rolling(7).mean()
    df['rv_roll30_mean'] = df['rv_t'].rolling(30).mean()
    df['rv_roll7_std'] = df['rv_t'].rolling(7).std()
    df['rv_roll30_std'] = df['rv_t'].rolling(30).std()
    
    df['mom_7d'] = (price - price.shift(7)) / (price.shift(7) + 1e-8)
    df['mom_30d'] = (price - price.shift(30)) / (price.shift(30) + 1e-8)
    
    # 3. Base Feature Candidates
    onchain_base = [
        'AdrActCnt', 'TxTfrValUSD', 'FeeTotUSD', 'RevUSD', 'HashRate',
        'NUPL', 'MVRV_Z', 'SOPR', 'NetFlowUSD', 'ShareExUSD',
        'PuellMultiple', 'CDD'
    ]
    ta_base = [
        'r_t', 'rv_t', 'rsi_14', 'macd', 'macd_signal',
        'bollinger_pct_b', 'r_t_roll7_mean', 'r_t_roll30_mean',
        'r_t_roll7_std', 'r_t_roll30_std', 'mom_7d', 'mom_30d',
        'rv_roll7_mean', 'rv_roll30_mean', 'rv_roll7_std', 'rv_roll30_std'
    ]
    
    # Add Lags for key features
    lag_features = ['r_t', 'rv_t', 'MVRV_Z', 'NUPL', 'AdrActCnt', 'HashRate', 'SOPR', 'PuellMultiple']
    for col in lag_features:
        if col in df.columns:
            for lag in [1, 3, 7]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
                
    candidate_cols = [
        c for c in df.columns
        if c not in ['time', 'PriceUSD', 'PriceUSD_log', 'PriceBTC', 'dir_t1']
        and not c.startswith('target_')
    ]
    # Ensure they are numeric
    candidate_cols = [c for c in candidate_cols if df[c].dtype in [np.float64, np.int64, float, int] and df[c].notnull().mean() > 0.50]
    
    horizons = [1, 3, 7]
    horizon_results = {}
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    for h in horizons:
        print(f"\n--- Processing Horizon {h} ---")
        df[f'target_h{h}'] = (price.shift(-h) > price).astype(int)
        target_col = f'target_h{h}'
        
        clean_df = df.dropna(subset=candidate_cols + [target_col]).copy().reset_index(drop=True)
        
        X = clean_df[candidate_cols].values
        y = clean_df[target_col].values
        returns = clean_df['r_t'].values
        
        # Accumulate predictions across all folds
        models_to_test = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss"),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
            "SVC_RBF": SVC(kernel='rbf', C=1.0, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=150, random_state=42)
        }
        
        feature_sets = ["Full", "Boruta", "Top20"]
        
        y_true_all = []
        returns_te_all = []
        preds_all = {m: {fs: [] for fs in feature_sets} for m in models_to_test}
        
        last_boruta_features = []
        last_boruta_success = False
        last_boruta_error = ""
        last_top20_features = []
        last_method_used = ""
        
        fold_idx = 0
        for train_index, test_index in tscv.split(X):
            fold_idx += 1
            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]
            ret_te = returns[test_index]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            # Top20 Baseline
            rf_fi = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
            top20_idx = np.argsort(rf_fi.feature_importances_)[::-1][:20]
            sel_mask_top20 = np.zeros(len(candidate_cols), dtype=bool)
            sel_mask_top20[top20_idx] = True
            curr_top20_feats = [candidate_cols[i] for i in range(len(candidate_cols)) if sel_mask_top20[i]]
            
            # Boruta
            sel_mask_boruta = np.zeros(len(candidate_cols), dtype=bool)
            boruta_success = False
            boruta_error = ""
            
            try:
                rf_b = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
                boruta = BorutaPy(rf_b, n_estimators='auto', random_state=42, max_iter=20)
                boruta.fit(X_tr_s, y_tr)
                sel_mask_boruta = boruta.support_
                boruta_success = True
                
                # If Boruta selects fewer than 3 features, consider it unsuccessful
                if np.sum(sel_mask_boruta) < 3:
                    boruta_success = False
                    boruta_error = f"Boruta selected only {np.sum(sel_mask_boruta)} features (too few)."
                    
            except Exception as e:
                boruta_success = False
                boruta_error = str(e)
                
            if not boruta_success:
                sel_mask_boruta = sel_mask_top20 # Fallback
                method_used = "Top20_RF_fallback"
                curr_boruta_feats = curr_top20_feats
            else:
                method_used = "Boruta"
                curr_boruta_feats = [candidate_cols[i] for i in range(len(candidate_cols)) if sel_mask_boruta[i]]
                
            # Store info from the last fold
            if fold_idx == 5:
                last_boruta_features = curr_boruta_feats
                last_boruta_success = boruta_success
                last_boruta_error = boruta_error
                last_top20_features = curr_top20_feats
                last_method_used = method_used
                
            X_tr_boruta = X_tr_s[:, sel_mask_boruta]
            X_te_boruta = X_te_s[:, sel_mask_boruta]
            X_tr_top20 = X_tr_s[:, sel_mask_top20]
            X_te_top20 = X_te_s[:, sel_mask_top20]
            
            y_true_all.extend(y_te)
            returns_te_all.extend(ret_te)
            
            for m_name, model in models_to_test.items():
                # Full
                model.fit(X_tr_s, y_tr)
                preds_all[m_name]["Full"].extend(model.predict(X_te_s))
                
                # Boruta
                model.fit(X_tr_boruta, y_tr)
                preds_all[m_name]["Boruta"].extend(model.predict(X_te_boruta))
                
                # Top20
                model.fit(X_tr_top20, y_tr)
                preds_all[m_name]["Top20"].extend(model.predict(X_te_top20))
                
        # Calculate overall metrics
        y_true_all = np.array(y_true_all)
        returns_te_all = np.array(returns_te_all)
        
        model_results = {}
        for m_name in models_to_test:
            model_results[m_name] = {}
            for fs in feature_sets:
                p = np.array(preds_all[m_name][fs])
                acc = accuracy_score(y_true_all, p)
                prec = precision_score(y_true_all, p, zero_division=0)
                rec = recall_score(y_true_all, p, zero_division=0)
                f1 = f1_score(y_true_all, p, zero_division=0)
                bacc = balanced_accuracy_score(y_true_all, p)
                mcc = matthews_corrcoef(y_true_all, p)
                
                # Strategy
                pos_diff = np.abs(np.diff(np.insert(p, 0, 0)))
                cost = pos_diff * 0.0010 # 10 bps
                strat_ret = p * returns_te_all - cost
                cum_ret = float(np.sum(strat_ret))
                ann_ret = float(np.mean(strat_ret) * 365.0)
                ann_vol = float(np.std(strat_ret, ddof=1) * np.sqrt(365.0)) if len(strat_ret) > 1 else 0.0
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
                
                model_results[m_name][f"Accuracy_{fs}"] = float(acc)
                model_results[m_name][f"F1_{fs}"] = float(f1)
                model_results[m_name][f"Precision_{fs}"] = float(prec)
                model_results[m_name][f"Recall_{fs}"] = float(rec)
                model_results[m_name][f"BalancedAcc_{fs}"] = float(bacc)
                model_results[m_name][f"MCC_{fs}"] = float(mcc)
                model_results[m_name][f"StrategyCumReturn_{fs}"] = float(cum_ret)
                model_results[m_name][f"StrategySharpe_{fs}"] = float(sharpe)
                
        horizon_results[f"h_{h}"] = {
            "feature_selection": {
                "method_used": last_method_used,
                "n_selected": len(last_boruta_features),
                "selected_features": last_boruta_features,
                "boruta_success": last_boruta_success,
                "error_if_any": last_boruta_error
            },
            "models": model_results
        }
        
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
        
    print("\n" + "=" * 125)
    print("  PAPER 7 V2 BITCOIN PRICE DIRECTION CLASSIFICATION (Expanding Window 5-Folds)")
    print("=" * 125)
    
    for h in horizons:
        h_data = horizon_results[f"h_{h}"]
        fsel = h_data["feature_selection"]
        print(f"\n--- Horizon h = {h} days ---")
        print(f"Boruta Success: {fsel['boruta_success']} | Method Used: {fsel['method_used']} | N Features: {fsel['n_selected']}")
        if not fsel['boruta_success']:
            print(f"Error/Reason: {fsel['error_if_any']}")
        print(f"Top 15 selected features: {fsel['selected_features'][:15]}")
        
        header = f"{'Model':<18} | {'Acc (Boruta)':<12} | {'Acc (Full)':<10} | {'F1 (Boruta)':<11} | {'Sharpe (Boruta)':<15} | {'Sharpe (Full)':<13}"
        print("-" * 125)
        print(header)
        print("-" * 125)
        
        for m_name, m in h_data["models"].items():
            row_str = (
                f"{m_name:<18} | {m['Accuracy_Boruta']:12.4f} | {m['Accuracy_Full']:10.4f} | "
                f"{m['F1_Boruta']:11.4f} | {m['StrategySharpe_Boruta']:+15.4f} | {m['StrategySharpe_Full']:+13.4f}"
            )
            print(row_str)
            
    print("=" * 125)
    print(f"Results successfully saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper7_ml_v2()
