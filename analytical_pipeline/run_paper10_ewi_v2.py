import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
import sys

# Ensure analytical_pipeline directory is in sys.path
sys.path.append(os.path.dirname(__file__))
from utils import load_merged_data

def run_paper10_ewi_v2():
    warnings.filterwarnings('ignore')
    
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
    
    active_features = [f for f in candidate_features if f in df.columns and df[f].notnull().mean() > 0.50]
    
    req_cols = ['time', 'rv_30d', 'PriceUSD', 'r_t'] + active_features
    df_clean = df.dropna(subset=req_cols).copy()
    df_clean['time'] = pd.to_datetime(df_clean['time'])
    df_clean = df_clean.sort_values('time').reset_index(drop=True)
    
    rv = df_clean['rv_30d'].values
    prices = df_clean['PriceUSD'].values
    times = df_clean['time'].dt.strftime('%Y-%m-%d').values
    n = len(df_clean)
    
    # 2. Target Construction: Expanding 95th percentile
    high_vol_exp = np.zeros(n, dtype=int)
    for t in range(365, n):
        hist_rv = rv[365:t] if t > 365 else rv[:t]
        q95 = np.quantile(hist_rv, 0.95)
        if rv[t] > q95:
            high_vol_exp[t] = 1
            
    X_all = df_clean[active_features].values
    
    # 3. Model & Cross-Validation Configuration
    start_idx = 1000
    step_days = 30
    
    def get_hgb():
        try:
            return HistGradientBoostingClassifier(max_iter=100, max_depth=5, class_weight='balanced', random_state=42)
        except TypeError:
            return HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)

    model_factories = {
        'logit_balanced': lambda: LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'rf_balanced': lambda: RandomForestClassifier(class_weight='balanced', n_estimators=50, max_depth=5, n_jobs=-1, random_state=42),
        'hgb_balanced': get_hgb
    }
    
    pca90_sample = PCA(n_components=0.90).fit(StandardScaler().fit_transform(X_all[:start_idx]))

    models_results = {}
    out_of_sample_probs = {}
    
    print("Running Expanding Window CV + Calibration...")
    for m_name, m_factory in model_factories.items():
        all_y_true = []
        all_y_prob = []
        all_times = []
        all_indices = []
        
        for t in range(start_idx, n - step_days, step_days):
            train_idx = list(range(t))
            test_idx = list(range(t, min(t + step_days, n)))
            
            X_tr, y_tr = X_all[train_idx], high_vol_exp[train_idx]
            X_te, y_te = X_all[test_idx], high_vol_exp[test_idx]
            
            if len(np.unique(y_tr)) < 2:
                continue
                
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            pca = PCA(n_components=0.90)
            X_tr_pca = pca.fit_transform(X_tr_sc)
            X_te_pca = pca.transform(X_te_sc)
            
            base_clf = m_factory()
            
            # Handling sample weights for older HGB
            fit_params = {}
            if m_name == 'hgb_balanced' and not hasattr(base_clf, 'class_weight'):
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
                # CalibratedClassifierCV needs sample_weight passed via fit_params for internal CV
                fit_params['sample_weight'] = sample_weights
                
            # Calibrate on validation folds internally (cv=5)
            calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=5)
            try:
                calibrated_clf.fit(X_tr_pca, y_tr, **fit_params)
            except Exception: # fallback if isotonic fails due to old sklearn or params
                calibrated_clf = CalibratedClassifierCV(base_estimator=base_clf, method='sigmoid', cv=5)
                calibrated_clf.fit(X_tr_pca, y_tr)
                
            probs = calibrated_clf.predict_proba(X_te_pca)[:, 1]
            
            all_y_true.extend(y_te)
            all_y_prob.extend(probs)
            all_times.extend(times[test_idx])
            all_indices.extend(test_idx)
            
        y_t = np.array(all_y_true)
        y_p = np.array(all_y_prob)
        y_pred_default = (y_p >= 0.5).astype(int)
        
        pr_auc = float(average_precision_score(y_t, y_p)) if np.sum(y_t) > 0 else 0.0
        roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0
        
        models_results[m_name] = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1_default_0.5": float(f1_score(y_t, y_pred_default, zero_division=0)),
            "recall_default_0.5": float(recall_score(y_t, y_pred_default, zero_division=0)),
            "precision_default_0.5": float(precision_score(y_t, y_pred_default, zero_division=0)),
            "balanced_acc_default_0.5": float(balanced_accuracy_score(y_t, y_pred_default))
        }
        
        out_of_sample_probs[m_name] = {
            "y_true": y_t,
            "y_prob": y_p,
            "times": all_times,
            "indices": all_indices
        }
        
    best_m_name = max(models_results.keys(), key=lambda k: models_results[k]["pr_auc"])
    best_data = out_of_sample_probs[best_m_name]
    
    y_true_best = best_data["y_true"]
    y_prob_best = best_data["y_prob"]
    times_best = best_data["times"]
    indices_best = best_data["indices"]
    
    # 4. Threshold Selection on Validation Portion (First 30% of OOS predictions)
    val_len = int(len(y_true_best) * 0.30)
    y_true_val = y_true_best[:val_len]
    y_prob_val = y_prob_best[:val_len]
    
    y_true_test = y_true_best[val_len:]
    y_prob_test = y_prob_best[val_len:]
    indices_test = indices_best[val_len:]
    times_test = times_best[val_len:]
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, th_f1 = -1.0, 0.5
    best_prec25, th_prec25 = -1.0, 0.5
    
    for th in thresholds:
        p_val = (y_prob_val >= th).astype(int)
        f1 = f1_score(y_true_val, p_val, zero_division=0)
        rec = recall_score(y_true_val, p_val, zero_division=0)
        prec = precision_score(y_true_val, p_val, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            th_f1 = float(th)
            
        if rec >= 0.25 and prec > best_prec25:
            best_prec25 = prec
            th_prec25 = float(th)
            
    if best_prec25 == -1.0:
        th_prec25 = th_f1
        
    def evaluate_threshold(th):
        preds = (y_prob_test >= th).astype(int)
        return {
            "threshold": float(th),
            "precision": float(precision_score(y_true_test, preds, zero_division=0)),
            "recall": float(recall_score(y_true_test, preds, zero_division=0)),
            "f1": float(f1_score(y_true_test, preds, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_true_test, preds))
        }
        
    res_max_f1 = evaluate_threshold(th_f1)
    res_prec25 = evaluate_threshold(th_prec25)
    
    # We will use the max_precision_at_recall_0.25 threshold for the Event Study
    alerts_test = (y_prob_test >= th_prec25).astype(int)
    alert_indices = [indices_test[i] for i in range(len(alerts_test)) if alerts_test[i] == 1]
    
    # 5. Event Study Calculation
    def run_event_study(alert_idxs):
        ret_7, ret_14, ret_30 = [], [], []
        mdd_7, mdd_14, mdd_30 = [], [], []
        
        for idx in alert_idxs:
            p0 = prices[idx]
            if idx + 30 >= n:
                continue
            
            p7, p14, p30 = prices[idx + 7], prices[idx + 14], prices[idx + 30]
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
            "avg_return_7d": float(np.mean(ret_7)), "avg_return_14d": float(np.mean(ret_14)), "avg_return_30d": float(np.mean(ret_30)),
            "avg_maxdd_7d": float(np.mean(mdd_7)), "avg_maxdd_14d": float(np.mean(mdd_14)), "avg_maxdd_30d": float(np.mean(mdd_30))
        }

    event_study_res = run_event_study(alert_indices)
    event_study_res["alerts_count_in_test"] = len(alert_indices)
    test_years = len(y_true_test) / 365.25
    event_study_res["alerts_per_year"] = float(len(alert_indices) / test_years) if test_years > 0 else 0.0
    
    # Monte Carlo 1000 iterations for random alert baseline
    valid_test_indices = [idx for idx in indices_test if idx + 30 < n]
    mc_results = []
    n_alerts = max(len(alert_indices), 10)
    
    for seed in range(1000):
        np.random.seed(seed)
        if len(valid_test_indices) >= n_alerts:
            rand_idxs = np.random.choice(valid_test_indices, size=n_alerts, replace=False)
            mc_results.append(run_event_study(rand_idxs))
        
    if mc_results:
        mc_df = pd.DataFrame(mc_results)
        vs_random = {
            "random_avg_return_7d": float(mc_df["avg_return_7d"].mean()),
            "random_avg_return_14d": float(mc_df["avg_return_14d"].mean()),
            "random_avg_return_30d": float(mc_df["avg_return_30d"].mean()),
            "random_avg_maxdd_7d": float(mc_df["avg_maxdd_7d"].mean()),
            "random_avg_maxdd_14d": float(mc_df["avg_maxdd_14d"].mean()),
            "random_avg_maxdd_30d": float(mc_df["avg_maxdd_30d"].mean())
        }
    else:
        vs_random = {k: 0.0 for k in ["random_avg_return_7d", "random_avg_return_14d", "random_avg_return_30d", "random_avg_maxdd_7d", "random_avg_maxdd_14d", "random_avg_maxdd_30d"]}
        
    event_study_res["vs_random"] = vs_random
    
    # Alert Quality Score
    num = event_study_res["avg_return_14d"] - vs_random["random_avg_return_14d"]
    den = abs(event_study_res["avg_maxdd_14d"]) + 1e-6
    alert_quality_score = float(num / den)
    
    # 7. Construct Final JSON Structure
    ts_list = []
    for i in range(len(y_prob_best)):
        idx = indices_best[i]
        ts_list.append({
            "time": times_best[i],
            "ewi": float(y_prob_best[i]),
            "alert_max_f1": int(y_prob_best[i] >= th_f1),
            "alert_prec25": int(y_prob_best[i] >= th_prec25),
            "rv_30d": float(rv[idx]),
            "high_vol": int(y_true_best[i])
        })

    output_json = {
        "paper10_ewi_v2": {
            "models": models_results,
            "calibration": {
                "method": "CalibratedClassifierCV (isotonic/sigmoid)",
                "cv": 5
            },
            "best_model": best_m_name,
            "threshold_optimization": {
                "max_f1": res_max_f1,
                "min_precision_at_recall_0.25": res_prec25
            },
            "event_study_v2": event_study_res,
            "alert_quality_score": alert_quality_score,
            "time_series": ts_list
        }
    }
    
    out_file = os.path.join(results_dir, "paper10_ewi_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        
    # 8. Console Table Output
    print("\n" + "=" * 95)
    print(f"  PAPER 10 V2 EWI CALIBRATED RESULTS  [Target: Expanding 95th Percentile RV]")
    print("=" * 95)
    print(f"Best Model Selected: {best_m_name} (PR-AUC: {models_results[best_m_name]['pr_auc']:.4f})")
    
    print("\n--- Threshold Optimization (Tested on 70% OOS Test Set) ---")
    print(f"1) Max F1 Threshold: {th_f1:.2f} -> Precision: {res_max_f1['precision']:.3f}, Recall: {res_max_f1['recall']:.3f}, F1: {res_max_f1['f1']:.3f}")
    print(f"2) Min Prec@Rec=0.25 Threshold: {th_prec25:.2f} -> Precision: {res_prec25['precision']:.3f}, Recall: {res_prec25['recall']:.3f}, F1: {res_prec25['f1']:.3f}")
    print(f"   (Rekomendowany dla dashboardu ze wzgledu na mniejsza liczbe False Positives)")
    
    print("\n--- Event Study (Using Threshold 2) ---")
    print(f"Alerts in Test Set: {len(alert_indices)} ({event_study_res['alerts_per_year']:.1f} per year)")
    print(f"Alert 14d Avg Return: {event_study_res['avg_return_14d']*100:.2f}% (vs Random: {vs_random['random_avg_return_14d']*100:.2f}%)")
    print(f"Alert 14d Avg MaxDD:  {event_study_res['avg_maxdd_14d']*100:.2f}% (vs Random: {vs_random['random_avg_maxdd_14d']*100:.2f}%)")
    print(f"Alert Quality Score:  {alert_quality_score:.3f} (Wyższy = Lepszy sygnał vs ryzyko)")
    
    print("=" * 95)
    print(f"Results saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper10_ewi_v2()
