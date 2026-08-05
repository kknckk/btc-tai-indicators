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
import traceback
import sys
sys.path.append(os.path.dirname(__file__))
from utils import load_merged_data

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def run_paper6_dl_v2():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "derived_metrics.csv")
    out_file = os.path.join(results_dir, "paper6_dl_v2.json")
    
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
    
    test_size = int(n * 0.20)
    tv_size = n - test_size
    val_size = int(tv_size * 0.15)
    train_size = tv_size - val_size
    
    scaler = StandardScaler()
    scaler.fit(X_raw[:train_size])
    X_scaled = scaler.transform(X_raw)
    
    y_test_full = y_raw[tv_size:]
    
    # Baseline A: Naive
    y_pred_naive = y_raw[tv_size-1:-1]
    metrics_naive = calc_metrics(y_test_full, y_pred_naive)
    
    # Baseline B: HAR-RV
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
    
    MAX_EPOCHS = 80
    PATIENCE = 10
    
    def train_single_model(model_class, model_kwargs, X_tr, y_tr, X_v, y_v, X_te, seed=42):
        set_seed(seed)
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.float32))
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        model = model_class(**model_kwargs)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        patience_cnt = 0
        best_state = None
        epochs_used = 0
        
        for epoch in range(MAX_EPOCHS):
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
                if patience_cnt >= PATIENCE:
                    epochs_used = epoch + 1
                    break
            epochs_used = epoch + 1
                    
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
            
        return test_preds, model, epochs_used

    seeds = [42, 101, 2024]
    seq_length_results = {}
    
    y_test_30d = None
    lstm_preds_30d = None
    trans_preds_30d = None
    
    for L in [14, 30, 60]:
        try:
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
            lstm_eps = []
            for s in seeds:
                p, _, ep = train_single_model(
                    LSTMForecaster, {'input_dim': X_tr_L.shape[2], 'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2},
                    X_tr_L, y_tr_L, X_v_L, y_v_L, X_te_L, seed=s
                )
                lstm_preds.append(p)
                lstm_eps.append(ep)
            lstm_avg = np.mean(lstm_preds, axis=0)
            
            trans_preds = []
            trans_eps = []
            last_trans_model = None
            for s in seeds:
                p, tr_m, ep = train_single_model(
                    TransformerForecaster, {'input_dim': X_tr_L.shape[2], 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1},
                    X_tr_L, y_tr_L, X_v_L, y_v_L, X_te_L, seed=s
                )
                trans_preds.append(p)
                trans_eps.append(ep)
                last_trans_model = tr_m
                
            trans_avg = np.mean(trans_preds, axis=0)
            
            m_lstm = calc_metrics(y_te_L, lstm_avg)
            m_trans = calc_metrics(y_te_L, trans_avg)
            
            seq_length_results[str(L)] = {
                "LSTM": m_lstm,
                "Transformer": m_trans,
                "epochs_used": {"LSTM": int(np.mean(lstm_eps)), "Transformer": int(np.mean(trans_eps))}
            }
            
            if L == 30:
                y_test_30d = y_te_L
                lstm_preds_30d = lstm_avg
                trans_preds_30d = trans_avg
                weights_path = os.path.join(results_dir, "best_transformer_weights.pt")
                if last_trans_model:
                    torch.save(last_trans_model.state_dict(), weights_path)
        except Exception as e:
            print(f"Error training sequence length {L}: {e}")
            traceback.print_exc()
            seq_length_results[str(L)] = {"error": str(e)}

    # Fallbacks in case 30 failed
    if y_test_30d is None:
        y_test_30d = y_test_full
        lstm_preds_30d = np.zeros_like(y_test_full)
        trans_preds_30d = np.zeros_like(y_test_full)

    dm_stat_tl, dm_p_tl = diebold_mariano_test(y_test_30d, trans_preds_30d, lstm_preds_30d)
    
    # Need to match lengths for HAR/GARCH tests
    min_len = min(len(y_test_30d), len(y_har_test), len(y_pred_garch))
    y_true_dm = y_test_30d[-min_len:]
    pred_trans = trans_preds_30d[-min_len:]
    pred_lstm = lstm_preds_30d[-min_len:]
    pred_har = y_pred_har[-min_len:]
    pred_garch = y_pred_garch[-min_len:]

    dm_stat_th, dm_p_th = diebold_mariano_test(y_true_dm, pred_trans, pred_har)
    dm_stat_lh, dm_p_lh = diebold_mariano_test(y_true_dm, pred_lstm, pred_har)
    dm_stat_tg, dm_p_tg = diebold_mariano_test(y_true_dm, pred_trans, pred_garch)
    
    dm_test_summary = {
        "Transformer_vs_LSTM": {"dm_stat": float(dm_stat_tl), "p_value": float(dm_p_tl)},
        "Transformer_vs_HAR_RV": {"dm_stat": float(dm_stat_th), "p_value": float(dm_p_th)},
        "LSTM_vs_HAR_RV": {"dm_stat": float(dm_stat_lh), "p_value": float(dm_p_lh)},
        "Transformer_vs_GARCH": {"dm_stat": float(dm_stat_tg), "p_value": float(dm_p_tg)}
    }
    
    baselines = {
        "naive": metrics_naive,
        "GARCH_1_1": metrics_garch,
        "HAR_RV": metrics_har
    }
    
    # find best model based on RMSE of seq_len 30 vs baselines
    best_rmse = float('inf')
    best_model = "Unknown"
    
    if "30" in seq_length_results and "Transformer" in seq_length_results["30"]:
        t_rmse = seq_length_results["30"]["Transformer"]["RMSE"]
        if t_rmse < best_rmse:
            best_rmse = t_rmse
            best_model = "Transformer"
    
    if "30" in seq_length_results and "LSTM" in seq_length_results["30"]:
        l_rmse = seq_length_results["30"]["LSTM"]["RMSE"]
        if l_rmse < best_rmse:
            best_rmse = l_rmse
            best_model = "LSTM"

    for b_name, b_metrics in baselines.items():
        if b_metrics["RMSE"] < best_rmse:
            best_rmse = b_metrics["RMSE"]
            best_model = b_name

    output_json = {
        "paper6_dl_v2": {
            "sequence_lengths": seq_length_results,
            "main_sequence_length": 30,
            "baselines": baselines,
            "diebold_mariano": dm_test_summary,
            "training_config": {
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "d_model": 64,
                "seeds": seeds
            },
            "best_model": best_model,
            "notes": "Automated run with error handling and early stopping."
        }
    }
    
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
    header = f"{'Model':<18} | {'SeqLen':<6} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8} | {'DM p-value vs HAR-RV':<20}"
    print(header)
    print("-" * 90)
    
    # Baselines
    for b_name, b_metrics in baselines.items():
        if b_name == "HAR_RV":
            dm_p = 1.0
        elif b_name == "naive":
            _, dm_p = diebold_mariano_test(y_true_dm, y_pred_naive[-min_len:], pred_har)
        elif b_name == "GARCH_1_1":
            _, dm_p = diebold_mariano_test(y_true_dm, pred_garch, pred_har)
        else:
            dm_p = 0.0
            
        row_str = f"{b_name:<18} | {'-':<6} | {b_metrics['RMSE']:8.4f} | {b_metrics['MAE']:8.4f} | {b_metrics['MAPE']:7.2f}% | {dm_p:<20.4f}"
        print(row_str)

    # DL Models
    for L, res in seq_length_results.items():
        if "error" in res:
            continue
        
        for m_name in ["LSTM", "Transformer"]:
            if m_name in res:
                m_metrics = res[m_name]
                if L == "30":
                    if m_name == "LSTM":
                        dm_p = dm_p_lh
                    else:
                        dm_p = dm_p_th
                else:
                    dm_p = float('nan') 
                    
                row_str = f"{m_name:<18} | {L:<6} | {m_metrics['RMSE']:8.4f} | {m_metrics['MAE']:8.4f} | {m_metrics['MAPE']:7.2f}% | {dm_p:<20.4f}"
                print(row_str)

    print("=" * 90)
    print(f"Results saved to: {out_file}\n")

if __name__ == "__main__":
    run_paper6_dl_v2()
