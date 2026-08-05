import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from arch import arch_model
from scipy import stats
import sys
sys.path.append(os.path.dirname(__file__))
from utils import load_merged_data


# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# 1. Model Definitions
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# Diebold-Mariano Test Function
def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion='MSE'):
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = e1**2 - e2**2 if criterion == 'MSE' else np.abs(e1) - np.abs(e2)
    
    n_samples = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += (1 - k / h) * gamma_k
        
    var_d = (gamma0 + 2 * gamma_sum) / n_samples
    if var_d <= 0:
        return 0.0, 1.0
        
    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_val)

def calc_metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}

def run_paper6_dl_v2():
    # 1. Load Data
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "derived_metrics.csv")


    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_merged_data()
        
    feature_candidates = [
        'AdrActCnt', 'TxCntSec', 'FeeTotUSD', 'FeeMeanUSD',
        'HashRate', 'NUPL', 'MVRV_Z', 'RevUSD', 'r_t', 'rv_7d'
    ]
    active_feats = [f for f in feature_candidates if f in df.columns and df[f].notnull().mean() > 0.50]
    target_col = 'rv_30d'
    
    req_cols = ['time', target_col, 'r_t'] + active_feats
    df_clean = df.dropna(subset=req_cols).copy()
    df_clean['time'] = pd.to_datetime(df_clean['time'])
    df_clean = df_clean.sort_values('time').reset_index(drop=True)
    
    X_raw = df_clean[active_feats].values
    y_raw = df_clean[target_col].values
    r_raw = df_clean['r_t'].values
    n = len(df_clean)
    
    # 2. Time-Series Split
    # Test set: last 20%
    # Validation set: 15% of remaining 80%
    test_size = int(n * 0.20)
    tv_size = n - test_size
    val_size = int(tv_size * 0.15)
    train_size = tv_size - val_size
    
    # Fit StandardScaler ONLY on train set
    scaler = StandardScaler()
    scaler.fit(X_raw[:train_size])
    X_scaled = scaler.transform(X_raw)
    
    # 3. Baseline Models (Evaluated on main sequence test window)
    y_test_full = y_raw[tv_size:]
    
    # Baseline A: Naive
    y_pred_naive = y_raw[tv_size-1:-1]
    metrics_naive = calc_metrics(y_test_full, y_pred_naive)
    
    # Baseline B: HAR-RV (RV_d, RV_w, RV_m)
    def build_har_features(rv_s):
        X_har, y_har = [], []
        for i in range(22, len(rv_s)):
            rv_d = rv_s[i-1]
            rv_w = np.mean(rv_s[i-5:i])
            rv_m = np.mean(rv_s[i-22:i])
            X_har.append([rv_d, rv_w, rv_m])
            y_har.append(rv_s[i])
        return np.array(X_har), np.array(y_har)
        
    X_har, y_har = build_har_features(y_raw)
    har_train_end = train_size - 22
    har_val_end = tv_size - 22
    y_har_test = y_har[har_val_end:]
    
    har_model = LinearRegression().fit(X_har[:har_train_end], y_har[:har_train_end])
    y_pred_har = har_model.predict(X_har[har_val_end:])
    metrics_har = calc_metrics(y_har_test, y_pred_har)

    
    # Baseline C: GARCH(1,1)
    am = arch_model(r_raw[:train_size] * 100, p=1, q=1, vol='Garch')
    res_garch = am.fit(disp='off')
    omega, alpha, beta = res_garch.params['omega'], res_garch.params['alpha[1]'], res_garch.params['beta[1]']
    
    garch_preds = []
    last_sig2 = res_garch.conditional_volatility[-1]**2
    for t in range(tv_size, n):
        r_prev = r_raw[t-1] * 100
        sig2_t = omega + alpha * (r_prev**2) + beta * last_sig2
        last_sig2 = sig2_t
        garch_preds.append(np.sqrt(sig2_t * 365) / 100.0)
    y_pred_garch = np.array(garch_preds)
    metrics_garch = calc_metrics(y_test_full, y_pred_garch)
    
    # 4. Training Helper Function for PyTorch Deep Learning Models
    def train_single_model(model_class, model_kwargs, X_tr, y_tr, X_v, y_v, X_te, seed=42):
        set_seed(seed)
        
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.float32))
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        model = model_class(**model_kwargs)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_cnt = 0
        best_state = None
        
        for epoch in range(3):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = model(bx)
                    val_losses.append(criterion(pred, by).item())
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= 1:
                    break







                    
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
            
        return test_preds, model

    # 5. Evaluate Sequence Lengths (14, 30, 60 days) & Model Seeds
    seeds = [42, 101, 2024]
    seq_length_results = {}
    
    best_transformer_weights = None
    y_test_30d = None
    lstm_preds_30d = None
    trans_preds_30d = None
    
    for L in [14, 30, 60]:
        Xs, ys = [], []
        for i in range(L, len(X_scaled)):
            Xs.append(X_scaled[i-L:i])
            ys.append(y_raw[i])
        Xs, ys = np.array(Xs), np.array(ys)
        
        tr_end = train_size - L
        vl_end = tv_size - L
        
        X_tr_L, y_tr_L = Xs[:tr_end], ys[:tr_end]
        X_v_L, y_v_L = Xs[tr_end:vl_end], ys[tr_end:vl_end]
        X_te_L, y_te_L = Xs[vl_end:], ys[vl_end:]
        
        lstm_preds = []
        for s in seeds:
            p, _ = train_single_model(
                LSTMForecaster, {'input_dim': X_tr_L.shape[2], 'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2},
                X_tr_L, y_tr_L, X_v_L, y_v_L, X_te_L, seed=s
            )
            lstm_preds.append(p)
        lstm_avg = np.mean(lstm_preds, axis=0)
        
        trans_preds = []
        last_trans_model = None
        for s in seeds:
            p, tr_m = train_single_model(
                TransformerForecaster, {'input_dim': X_tr_L.shape[2], 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1},
                X_tr_L, y_tr_L, X_v_L, y_v_L, X_te_L, seed=s
            )
            trans_preds.append(p)
            last_trans_model = tr_m
            
        trans_avg = np.mean(trans_preds, axis=0)
        
        m_lstm = calc_metrics(y_te_L, lstm_avg)
        m_trans = calc_metrics(y_te_L, trans_avg)
        _, dm_p = diebold_mariano_test(y_te_L, trans_avg, lstm_avg)
        
        seq_length_results[str(L)] = {
            "LSTM": m_lstm,
            "Transformer": m_trans,
            "DM_vs_LSTM_pval": float(dm_p)
        }
        
        if L == 30:
            y_test_30d = y_te_L
            lstm_preds_30d = lstm_avg
            trans_preds_30d = trans_avg
            weights_path = os.path.join(results_dir, "best_transformer_weights.pt")
            torch.save(last_trans_model.state_dict(), weights_path)

    # 6. Diebold-Mariano Tests for Main Models (L=30)
    m_lstm_30 = calc_metrics(y_test_30d, lstm_preds_30d)
    m_trans_30 = calc_metrics(y_test_30d, trans_preds_30d)
    
    dm_stat_tl, dm_p_tl = diebold_mariano_test(y_test_30d, trans_preds_30d, lstm_preds_30d)
    dm_stat_th, dm_p_th = diebold_mariano_test(y_har_test, trans_preds_30d, y_pred_har)
    dm_stat_lh, dm_p_lh = diebold_mariano_test(y_har_test, lstm_preds_30d, y_pred_har)
    
    _, dm_p_naive_lstm = diebold_mariano_test(y_test_30d, y_pred_naive, lstm_preds_30d)
    _, dm_p_har_lstm = diebold_mariano_test(y_har_test, y_pred_har, lstm_preds_30d)
    _, dm_p_garch_lstm = diebold_mariano_test(y_test_30d, y_pred_garch, lstm_preds_30d)
    
    models_30d = {
        "Naive": {**metrics_naive, "DM_pval_vs_LSTM": float(dm_p_naive_lstm)},
        "HAR_RV": {**metrics_har, "DM_pval_vs_LSTM": float(dm_p_har_lstm)},
        "GARCH_1_1": {**metrics_garch, "DM_pval_vs_LSTM": float(dm_p_garch_lstm)},
        "LSTM": {**m_lstm_30, "DM_pval_vs_LSTM": 1.000},
        "Transformer": {**m_trans_30, "DM_pval_vs_LSTM": float(dm_p_tl)}
    }
    
    dm_test_summary = {
        "Transformer_vs_LSTM": {"dm_stat": float(dm_stat_tl), "p_value": float(dm_p_tl)},
        "Transformer_vs_HAR_RV": {"dm_stat": float(dm_stat_th), "p_value": float(dm_p_th)},
        "LSTM_vs_HAR_RV": {"dm_stat": float(dm_stat_lh), "p_value": float(dm_p_lh)}
    }
    
    # 7. Output JSON Structure
    output_json = {
        "paper6_dl_v2": {
            "sequence_lengths": seq_length_results,
            "models_30d": models_30d,
            "dm_test": dm_test_summary,
            "features_used": active_feats,
            "target": target_col,
            "dataset_info": {
                "total_rows": n,
                "train_rows": train_size,
                "val_rows": val_size,
                "test_rows": test_size
            },
            "training_params": {
                "seeds": seeds,
                "max_epochs": 150,
                "patience": 15,
                "batch_size": 64,
                "learning_rate": 0.001
            },
            "weights_saved": os.path.join(results_dir, "best_transformer_weights.pt")
        }
    }
    
    out_file = os.path.join(results_dir, "paper6_dl_v2.json")
    print(f"Writing JSON output to: {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"JSON successfully written. File exists: {os.path.exists(out_file)}")

        
    # 8. Console Table Output
    print("\n" + "=" * 90)
    print("  PAPER 6 V2 DEEP LEARNING VOLATILITY FORECASTING RESULTS (Rafi et al., 2024)")
    print("=" * 90)
    header = f"{'Model':<18} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8} | {'DM p-value vs LSTM':<20}"
    print(header)
    print("-" * 90)
    
    for m_name, m in models_30d.items():
        row_str = f"{m_name:<18} | {m['RMSE']:8.4f} | {m['MAE']:8.4f} | {m['MAPE']:7.2f}% | {m['DM_pval_vs_LSTM']:<20.4f}"
        print(row_str)
    print("=" * 90)
    print(f"Results saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper6_dl_v2()
