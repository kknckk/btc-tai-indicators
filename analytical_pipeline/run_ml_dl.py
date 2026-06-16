import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from prophet import Prophet
from utils import load_merged_data, save_results

def run_paper5_prophet(df: pd.DataFrame):
    df_clean = df.dropna(subset=['PriceUSD', 'RevUSD', 'AdrActCnt', 'FeeTotUSD']).copy()
    
    # Przygotowanie danych dla Propheta
    p_df = pd.DataFrame()
    p_df['ds'] = df_clean['time']
    p_df['y'] = df_clean['PriceUSD']
    p_df['RevUSD'] = df_clean['RevUSD']
    p_df['AdrActCnt'] = df_clean['AdrActCnt']
    p_df['FeeTotUSD'] = df_clean['FeeTotUSD']
    
    # Train/Test split (ostatnie 30 dni jako test)
    train = p_df.iloc[:-30]
    test = p_df.iloc[-30:]
    
    m = Prophet(daily_seasonality=True)
    m.add_regressor('RevUSD')
    m.add_regressor('AdrActCnt')
    m.add_regressor('FeeTotUSD')
    
    m.fit(train)
    
    forecast = m.predict(test.drop(columns=['y']))
    
    # Metryki
    y_true = test['y'].values
    y_pred = forecast['yhat'].values
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Predykcje do wyświetlenia
    predictions = []
    for i in range(len(test)):
        predictions.append({
            "time": str(test.iloc[i]['ds'].date()),
            "actual": float(y_true[i]),
            "predicted": float(y_pred[i])
        })
        
    return {
        "metrics": {"MAE": float(mae), "RMSE": float(rmse)},
        "predictions": predictions
    }

def run_paper7_classification(df: pd.DataFrame):
    if 'dir_t1' not in df.columns:
        return {}
        
    features_onchain = ['AdrActCnt', 'TxTfrValUSD', 'FeeTotUSD', 'RevUSD', 'HashRate', 'NUPL', 'MVRV_Z']
    features_ta = ['r_t', 'rv_7d'] # uproszczone proxy dla TA
    
    available_features = [f for f in features_onchain + features_ta if f in df.columns]
    
    df_clean = df.dropna(subset=available_features + ['dir_t1'])
    X = df_clean[available_features]
    y = df_clean['dir_t1']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    # Feature importance z Random Forest
    importances = rf.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)
    
    return {
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, y_pred_rf)),
            "f1": float(f1_score(y_test, y_pred_rf)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
        },
        "svm": {
            "accuracy": float(accuracy_score(y_test, y_pred_svm)),
            "f1": float(f1_score(y_test, y_pred_svm))
        },
        "feature_importance": [{"feature": f, "importance": float(imp)} for f, imp in feat_imp]
    }

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    df_path = os.path.join(results_dir, "derived_metrics.csv")
    if not os.path.exists(df_path):
        print("Brak derived_metrics.csv")
        return
        
    df = pd.read_csv(df_path)
    df['time'] = pd.to_datetime(df['time'])
    
    results = {}
    
    try:
        results['paper5'] = run_paper5_prophet(df)
        print("Paper 5 (Prophet) OK")
    except Exception as e:
        print(f"Paper 5 Error: {e}")
        
    try:
        results['paper7'] = run_paper7_classification(df)
        print("Paper 7 (Classification) OK")
    except Exception as e:
        print(f"Paper 7 Error: {e}")
        
    with open(os.path.join(results_dir, "ml_results.json"), "w") as f:
        json.dump(results, f, default=str)
    
    print("Wyniki ML zapisane.")

if __name__ == "__main__":
    main()
