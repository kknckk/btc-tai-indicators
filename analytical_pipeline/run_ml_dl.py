import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
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
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train) # RF doesn't strictly need scaling, but we can pass X_train
    y_pred_rf = rf.predict(X_test)
    
    svm = LinearSVC(dual=False, random_state=42, max_iter=2000)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    
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
        "feature_importance": [{"feature": f, "importance": float(imp)} for f, imp in feat_imp],
        "roc_data": [
            {"model": "Random Forest", "accuracy": float(accuracy_score(y_test, y_pred_rf))},
            {"model": "SVM", "accuracy": float(accuracy_score(y_test, y_pred_svm))}
        ]
    }

class VolatilityTransformer(nn.Module):
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]
        emb = self.embedding(src)
        out = self.transformer_encoder(emb)
        # Take the last sequence output
        out = out[:, -1, :]
        return self.fc_out(out).squeeze(-1)

class VolatilityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc_out(out).squeeze(-1)

def run_paper6_dl_volatility(df: pd.DataFrame):
    features = ['AdrActCnt', 'FeeTotUSD', 'RevUSD', 'TxTfrValUSD', 'rv_7d']
    available_features = [f for f in features if f in df.columns]
    
    if 'rv_7d' not in available_features:
        return {}
        
    df_clean = df.dropna(subset=available_features).copy()
    
    # Scale features
    data = df_clean[available_features].values
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    
    seq_len = 14
    X_seq = []
    y_seq = []
    
    # Target is rv_7d
    target_idx = available_features.index('rv_7d')
    
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i+seq_len])
        y_seq.append(data[i+seq_len, target_idx])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    if len(X_seq) < 100:
        return {}
        
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = torch.tensor(X_seq[:train_size], dtype=torch.float32), torch.tensor(X_seq[train_size:], dtype=torch.float32)
    y_train, y_test = torch.tensor(y_seq[:train_size], dtype=torch.float32), torch.tensor(y_seq[train_size:], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize models
    input_dim = len(available_features)
    lstm_model = VolatilityLSTM(input_dim=input_dim)
    transformer_model = VolatilityTransformer(input_dim=input_dim)
    
    criterion = nn.MSELoss()
    optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    optimizer_tf = torch.optim.Adam(transformer_model.parameters(), lr=0.01)
    
    # Fast training for PoC (5 epochs)
    epochs = 5
    for epoch in range(epochs):
        lstm_model.train()
        transformer_model.train()
        for batch_x, batch_y in train_loader:
            optimizer_lstm.zero_grad()
            out_lstm = lstm_model(batch_x)
            loss_lstm = criterion(out_lstm, batch_y)
            loss_lstm.backward()
            optimizer_lstm.step()
            
            optimizer_tf.zero_grad()
            out_tf = transformer_model(batch_x)
            loss_tf = criterion(out_tf, batch_y)
            loss_tf.backward()
            optimizer_tf.step()
            
    # Eval
    lstm_model.eval()
    transformer_model.eval()
    with torch.no_grad():
        pred_lstm = lstm_model(X_test)
        pred_tf = transformer_model(X_test)
        mse_lstm = criterion(pred_lstm, y_test).item()
        mse_tf = criterion(pred_tf, y_test).item()
        
    # Time series of actual vs pred for last N points
    n_plot = min(100, len(y_test))
    ts_actual = y_test[-n_plot:].numpy().tolist()
    ts_lstm = pred_lstm[-n_plot:].numpy().tolist()
    ts_tf = pred_tf[-n_plot:].numpy().tolist()
    
    ts_data = []
    # Dates for these last n_plot
    # The original dataframe has 'time' column. We lost it during scaling.
    # Let's just create an index for simplicity
    for i in range(n_plot):
        ts_data.append({
            "index": i,
            "actual": float(ts_actual[i]),
            "lstm": float(ts_lstm[i]),
            "transformer": float(ts_tf[i])
        })
        
    return {
        "sequence_length": seq_len,
        "epochs_trained": epochs,
        "test_mse": {
            "LSTM": float(mse_lstm),
            "Transformer": float(mse_tf)
        },
        "time_series": ts_data
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
        
    try:
        results['paper6'] = run_paper6_dl_volatility(df)
        print("Paper 6 (Transformers vs LSTM) OK")
    except Exception as e:
        print(f"Paper 6 Error: {e}")
        
    with open(os.path.join(results_dir, "ml_results.json"), "w") as f:
        json.dump(results, f, default=str)
    
    print("Wyniki ML zapisane.")

if __name__ == "__main__":
    main()
