import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from utils import load_merged_data

def run_paper10_ewi_v2():
    # 1. Load Data
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    csv_path = os.path.join(results_dir, "derived_metrics.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_merged_data()
        
    candidate_features = [
        'TxCnt', 'TxTfrCnt', 'TxTfrValUSD', 'UTXOCnt',
        'AdrActCnt', 'FeeTotUSD', 'BlkSizeMeanByte', 'TxCntSec'
    ]
    
    # Filter features that exist and have >50% non-null values
    active_features = [f for f in candidate_features if f in df.columns and df[f].notnull().mean() > 0.50]
    
    req_cols = ['time', 'rv_30d', 'PriceUSD', 'r_t'] + active_features
    missing = [c for c in ['time', 'rv_30d', 'PriceUSD', 'r_t'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
        
    df_clean = df.dropna(subset=req_cols).copy()
    df_clean['time'] = pd.to_datetime(df_clean['time'])
    df_clean = df_clean.sort_values('time').reset_index(drop=True)
    
    rv = df_clean['rv_30d'].values
    prices = df_clean['PriceUSD'].values
    times = df_clean['time'].dt.strftime('%Y-%m-%d').values
    n = len(df_clean)
    
    # 2. Target Construction
    # Primary: Expanding 95th percentile target (no future look-ahead)
    # Using past history starting after initial warmup (index 365) to avoid 2010 initial extreme outlier distortion
    high_vol_exp = np.zeros(n, dtype=int)
    for t in range(365, n):
        hist_rv = rv[365:t] if t > 365 else rv[:t]
        q95 = np.quantile(hist_rv, 0.95)
        if rv[t] > q95:
            high_vol_exp[t] = 1
            
    # Alternative for comparison: Fixed 95th percentile target on full history
    q95_fixed = np.quantile(rv, 0.95)
    high_vol_fixed = (rv > q95_fixed).astype(int)
    
    X_all = df_clean[active_features].values
    
    # 3. Model & Cross-Validation Configuration
    # Rolling origin / Expanding window setup: Start at index 1000, step 30 days
    start_idx = 1000
    step_days = 30
    
    model_factories = {
        'logit_balanced': lambda w: LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'logit_custom_w': lambda w: LogisticRegression(class_weight={0: 1.0, 1: float(w)}, max_iter=1000, random_state=42),
        'rf_balanced': lambda w: RandomForestClassifier(class_weight='balanced', n_estimators=10, max_depth=5, n_jobs=1, random_state=42)
    }




    # Record overall PCA variance ratios on train folds
    sample_scaler = StandardScaler()
    sample_X_sc = sample_scaler.fit_transform(X_all[:start_idx])
    pca2_sample = PCA(n_components=2).fit(sample_X_sc)
    pca90_sample = PCA(n_components=0.90).fit(sample_X_sc)
    
    pca_info = {
        "features_used": active_features,
        "pca_2": {
            "n_components": 2,
            "explained_variance_ratio": [float(v) for v in pca2_sample.explained_variance_ratio_],
            "total_explained_variance": float(np.sum(pca2_sample.explained_variance_ratio_))
        },
        "pca_90": {
            "n_components": int(pca90_sample.n_components_),
            "explained_variance_ratio": [float(v) for v in pca90_sample.explained_variance_ratio_],
            "total_explained_variance": float(np.sum(pca90_sample.explained_variance_ratio_))
        }
    }
    
    # Evaluate target variants (Primary: Expanding target, Comparison: Fixed target)
    def evaluate_target_variant(target_array, pca_components=0.90):
        models_results = {}
        out_of_sample_probs = {}
        
        for m_name, m_factory in model_factories.items():
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            fold_bal_accs = []
            
            all_y_true = []
            all_y_prob = []
            all_times = []
            all_indices = []
            
            for t in range(start_idx, n - step_days, step_days):
                train_idx = list(range(t))
                test_idx = list(range(t, min(t + step_days, n)))
                
                X_tr, y_tr = X_all[train_idx], target_array[train_idx]
                X_te, y_te = X_all[test_idx], target_array[test_idx]
                
                if len(np.unique(y_tr)) < 2:
                    continue
                    
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)
                
                if pca_components == 2:
                    pca = PCA(n_components=2)
                else:
                    pca = PCA(n_components=0.90)
                    
                X_tr_pca = pca.fit_transform(X_tr_sc)
                X_te_pca = pca.transform(X_te_sc)
                
                n0 = np.sum(y_tr == 0)
                n1 = np.sum(y_tr == 1)
                w = n0 / max(n1, 1)
                
                clf = m_factory(w)
                clf.fit(X_tr_pca, y_tr)
                probs = clf.predict_proba(X_te_pca)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                # Per-fold metrics
                if np.sum(y_te == 1) > 0:
                    fold_precisions.append(precision_score(y_te, preds, zero_division=0))
                    fold_recalls.append(recall_score(y_te, preds, zero_division=0))
                    fold_f1s.append(f1_score(y_te, preds, zero_division=0))
                    fold_bal_accs.append(balanced_accuracy_score(y_te, preds))
                    
                all_y_true.extend(y_te)
                all_y_prob.extend(probs)
                all_times.extend(times[test_idx])
                all_indices.extend(test_idx)
                
            y_t = np.array(all_y_true)
            y_p = np.array(all_y_prob)
            y_pred_default = (y_p >= 0.5).astype(int)
            
            pr_auc = float(average_precision_score(y_t, y_p)) if np.sum(y_t) > 0 else 0.0
            roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0
            
            cm = confusion_matrix(y_t, y_pred_default)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            
            models_results[m_name] = {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": float(precision_score(y_t, y_pred_default, zero_division=0)),
                "precision_std": float(np.std(fold_precisions)) if fold_precisions else 0.0,
                "recall": float(recall_score(y_t, y_pred_default, zero_division=0)),
                "recall_std": float(np.std(fold_recalls)) if fold_recalls else 0.0,
                "f1": float(f1_score(y_t, y_pred_default, zero_division=0)),
                "f1_std": float(np.std(fold_f1s)) if fold_f1s else 0.0,
                "balanced_acc": float(balanced_accuracy_score(y_t, y_pred_default)),
                "balanced_acc_std": float(np.std(fold_bal_accs)) if fold_bal_accs else 0.0,
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
            }
            
            out_of_sample_probs[m_name] = {
                "y_true": y_t,
                "y_prob": y_p,
                "times": all_times,
                "indices": all_indices
            }
            
        return models_results, out_of_sample_probs

    # Run evaluations for expanding target
    exp_models_res, exp_oos = evaluate_target_variant(high_vol_exp, pca_components=0.90)
    fix_models_res, _ = evaluate_target_variant(high_vol_fixed, pca_components=0.90)
    
    # Also evaluate with PCA n_components = 2 for expanding
    exp_models_pca2, _ = evaluate_target_variant(high_vol_exp, pca_components=2)
    
    # Select best model based on PR-AUC
    best_m_name = max(exp_models_res.keys(), key=lambda k: exp_models_res[k]["pr_auc"])
    best_data = exp_oos[best_m_name]
    
    y_true_best = best_data["y_true"]
    y_prob_best = best_data["y_prob"]
    times_best = best_data["times"]
    indices_best = best_data["indices"]
    
    # 4. Threshold Selection on Validation Portion (First 30% of OOS predictions)
    val_len = int(len(y_true_best) * 0.30)
    y_true_val = y_true_best[:val_len]
    y_prob_val = y_prob_best[:val_len]
    
    best_thresh = 0.50
    best_val_f1 = -1.0
    
    candidate_thresholds = np.linspace(0.10, 0.90, 81)
    for p in candidate_thresholds:
        p_preds = (y_prob_val >= p).astype(int)
        score = f1_score(y_true_val, p_preds, zero_division=0)
        if score > best_val_f1:
            best_val_f1 = score
            best_thresh = float(p)
            
    alerts_best = (y_prob_best >= best_thresh).astype(int)
    
    # 5. Event Study Calculation
    def run_event_study(alert_idxs):
        ret_7, ret_14, ret_30 = [], [], []
        mdd_7, mdd_14, mdd_30 = [], [], []
        
        for idx in alert_idxs:
            p0 = prices[idx]
            if idx + 30 >= n:
                continue
            p7 = prices[idx + 7]
            p14 = prices[idx + 14]
            p30 = prices[idx + 30]
            
            ret_7.append((p7 - p0) / p0)
            ret_14.append((p14 - p0) / p0)
            ret_30.append((p30 - p0) / p0)
            
            mdd_7.append(np.min((prices[idx:idx+8] - p0) / p0))
            mdd_14.append(np.min((prices[idx:idx+15] - p0) / p0))
            mdd_30.append(np.min((prices[idx:idx+31] - p0) / p0))
            
        if not ret_7:
            return {
                "avg_return_7d": 0.0, "avg_return_14d": 0.0, "avg_return_30d": 0.0,
                "avg_maxdd_7d": 0.0, "avg_maxdd_14d": 0.0, "avg_maxdd_30d": 0.0
            }
            
        return {
            "avg_return_7d": float(np.mean(ret_7)),
            "avg_return_14d": float(np.mean(ret_14)),
            "avg_return_30d": float(np.mean(ret_30)),
            "avg_maxdd_7d": float(np.mean(mdd_7)),
            "avg_maxdd_14d": float(np.mean(mdd_14)),
            "avg_maxdd_30d": float(np.mean(mdd_30))
        }

    alert_indices = [indices_best[i] for i in range(len(alerts_best)) if alerts_best[i] == 1]
    event_study_res = run_event_study(alert_indices)
    event_study_res["alerts_count"] = len(alert_indices)
    
    # Monte Carlo 1000 iterations for random alert baseline
    valid_test_indices = [idx for idx in indices_best if idx + 30 < n]
    mc_results = []
    n_alerts = max(len(alert_indices), 10)
    
    for seed in range(1000):
        np.random.seed(seed)
        rand_idxs = np.random.choice(valid_test_indices, size=n_alerts, replace=False)
        mc_results.append(run_event_study(rand_idxs))
        
    mc_df = pd.DataFrame(mc_results)
    vs_random = {
        "random_avg_return_7d": float(mc_df["avg_return_7d"].mean()),
        "random_avg_return_14d": float(mc_df["avg_return_14d"].mean()),
        "random_avg_return_30d": float(mc_df["avg_return_30d"].mean()),
        "random_avg_maxdd_7d": float(mc_df["avg_maxdd_7d"].mean()),
        "random_avg_maxdd_14d": float(mc_df["avg_maxdd_14d"].mean()),
        "random_avg_maxdd_30d": float(mc_df["avg_maxdd_30d"].mean())
    }
    event_study_res["vs_random"] = vs_random
    
    # 6. Time Series Output Series
    ts_list = []
    for i in range(len(y_prob_best)):
        idx = indices_best[i]
        ts_list.append({
            "time": times_best[i],
            "ewi": float(y_prob_best[i]),
            "alert": int(alerts_best[i]),
            "rv_30d": float(rv[idx]),
            "high_vol": int(y_true_best[i])
        })

    # 7. Construct Final JSON Structure
    output_json = {
        "pca": pca_info,
        "models": exp_models_res,
        "models_pca2": exp_models_pca2,
        "models_fixed_target_comparison": fix_models_res,
        "best_model": best_m_name,
        "threshold": best_thresh,
        "threshold_val_f1": float(best_val_f1),
        "event_study": event_study_res,
        "time_series": ts_list
    }
    
    out_file = os.path.join(results_dir, "paper10_ewi_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        
    # 8. Console Table Output
    print("\n" + "=" * 95)
    print(f"  PAPER 10 V2 EARLY WARNING INDICATOR RESULTS  [Target: Expanding 95th Percentile RV]")
    print("=" * 95)
    header = f"{'Model':<22} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'F1':<6} | {'Recall':<7} | {'Precision':<9} | {'BalAcc':<7}"
    print(header)
    print("-" * 95)
    
    for m_name, m in exp_models_res.items():
        row_str = (
            f"{m_name:<22} | {m['pr_auc']:7.4f} | {m['roc_auc']:7.4f} | "
            f"{m['f1']:6.4f} | {m['recall']:7.4f} | {m['precision']:9.4f} | {m['balanced_acc']:7.4f}"
        )
        print(row_str)
    print("=" * 95)
    print(f"Best Model Selected: {best_m_name} (Optimal Validation Threshold: {best_thresh:.2f})")
    print(f"Alerts Generated: {len(alert_indices)} | Event Study Avg MaxDD 14d: {event_study_res['avg_maxdd_14d']*100:.2f}% (vs Random: {vs_random['random_avg_maxdd_14d']*100:.2f}%)")
    print(f"Results saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper10_ewi_v2()
