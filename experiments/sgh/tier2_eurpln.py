"""Tier-II EUR/PLN forecasters: VMD-CNN-BiLSTM, anomaly transformer, TFT (TRD-47)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from experiments.sgh.tier1_bitcoin import _split_train_val
from models._common import TrainConfig, make_windows, set_seed, timed_train_loop


class _ConvBiLSTMNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int, conv_channels: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, conv_channels, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel, padding=pad),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.conv(z).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :]).squeeze(-1)


class _AnomalyTransformerNet(nn.Module):
    def __init__(self, n_feat: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, anomaly_w: torch.Tensor | None = None) -> torch.Tensor:
        z = self.in_proj(x)
        z = self.encoder(z)
        if anomaly_w is not None:
            w = anomaly_w.unsqueeze(-1)
            z = z * (1.0 + w)
        return self.head(z[:, -1, :]).squeeze(-1)


class _SimpleTFTNet(nn.Module):
    """Lightweight TFT-style block: variable selection + BiLSTM + temporal attention."""

    def __init__(self, n_feat: int, hidden: int, nhead: int, dropout: float):
        super().__init__()
        self.var_gate = nn.Sequential(nn.Linear(n_feat, n_feat), nn.Softmax(dim=-1))
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden * 2, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.var_gate(x)
        x = x * gates
        h, _ = self.lstm(x)
        q = h[:, -1:, :]
        ctx, _ = self.attn(q, h, h)
        return self.head(ctx.squeeze(1)).squeeze(-1)


@dataclass
class _WindowTrainer:
    window: int
    hidden: int
    lr: float
    max_epochs: int
    patience: int
    seed: int
    net_factory: Callable[[int], nn.Module]
    _model: nn.Module | None = field(default=None, init=False, repr=False)
    _x_scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _y_mu: float = field(default=0.0, init=False)
    _y_sd: float = field(default=1.0, init=False)
    _extra: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit_arrays(self, X_arr: np.ndarray, y_arr: np.ndarray, extra: np.ndarray | None = None) -> None:
        set_seed(self.seed)
        self._x_scaler = StandardScaler().fit(X_arr)
        xs = self._x_scaler.transform(X_arr)
        self._y_mu = float(np.nanmean(y_arr))
        self._y_sd = float(np.nanstd(y_arr)) or 1.0
        ys = (y_arr - self._y_mu) / self._y_sd
        Xw, yw, _ = make_windows(xs, ys, self.window)
        mask = np.isfinite(yw)
        Xw, yw = Xw[mask], yw[mask]
        if len(Xw) < 8:
            raise ValueError("not enough windows")
        ew = None
        if extra is not None:
            _, ew, _ = make_windows(extra, None, self.window)
            ew = ew[mask]
        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]
        model = self.net_factory(X_arr.shape[1])
        if ew is not None:
            self._extra = ew
            self._train_anomaly(model, X_tr, y_tr, X_val, y_val, ew[:-val_n], ew[-val_n:])
        else:
            cfg = TrainConfig(
                lr=self.lr,
                batch_size=min(64, max(1, len(X_tr))),
                max_epochs=self.max_epochs,
                patience=self.patience,
            )
            self._model, _, _ = timed_train_loop(model, X_tr, y_tr, X_val, y_val, cfg)

    def _train_anomaly(
        self,
        model: _AnomalyTransformerNet,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ew_tr: np.ndarray,
        ew_val: np.ndarray,
    ) -> None:
        device = torch.device("cpu")
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        best = float("inf")
        best_state = None
        patience = 0
        for _ in range(self.max_epochs):
            model.train()
            bx = torch.tensor(X_tr, dtype=torch.float32, device=device)
            by = torch.tensor(y_tr, dtype=torch.float32, device=device)
            w = torch.tensor(ew_tr, dtype=torch.float32, device=device)
            opt.zero_grad()
            pred = model(bx, w)
            loss = nn.functional.mse_loss(pred, by)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(
                    torch.tensor(X_val, dtype=torch.float32, device=device),
                    torch.tensor(ew_val, dtype=torch.float32, device=device),
                )
                vloss = float(nn.functional.mse_loss(pv, torch.tensor(y_val, device=device)))
            if vloss < best:
                best = vloss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        model.to("cpu")
        self._model = model

    def predict_arrays(self, X_arr: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        if self._model is None or self._x_scaler is None:
            raise RuntimeError("model not fitted")
        xs = self._x_scaler.transform(X_arr)
        Xw, _, idx = make_windows(xs, None, self.window)
        out = np.full(len(X_arr), np.nan, dtype=float)
        if not len(idx):
            return out
        self._model.eval()
        with torch.no_grad():
            if extra is not None:
                ew, _, _ = make_windows(extra, None, self.window)
                pred = self._model(
                    torch.tensor(Xw, dtype=torch.float32),
                    torch.tensor(ew, dtype=torch.float32),
                ).numpy()
            else:
                pred = self._model(torch.tensor(Xw, dtype=torch.float32)).numpy()
        out[np.array(idx)] = pred * self._y_sd + self._y_mu
        return out


def _vmd_cnn_columns(cols: list[str]) -> list[str]:
    prefer = [c for c in cols if c.startswith("vmd_")]
    rest = [c for c in cols if c not in prefer]
    return prefer + rest


class VMDCNNBiLSTMForecaster:
    """Causal VMD modes (in features) + 1D-CNN + BiLSTM level forecaster."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.window = int(cfg.get("window", 42))
        self.hidden = int(cfg.get("hidden", 32))
        self.conv_channels = int(cfg.get("conv_channels", 16))
        self.kernel = int(cfg.get("kernel", 3))
        self.lr = float(cfg.get("lr", 1e-3))
        self.max_epochs = int(cfg.get("max_epochs", 40))
        self.patience = int(cfg.get("patience", 8))
        self._trainer: _WindowTrainer | None = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VMDCNNBiLSTMForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = _vmd_cnn_columns(list(Xf.columns))
        tr = _WindowTrainer(
            window=self.window,
            hidden=self.hidden,
            lr=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
            net_factory=lambda n: _ConvBiLSTMNet(n, self.hidden, self.conv_channels, self.kernel),
        )
        tr.fit_arrays(Xf[self._cols].to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._trainer = tr
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._trainer is not None
        pred = self._trainer.predict_arrays(X[self._cols].to_numpy(dtype=float))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "vmd_cnn_bilstm", "window": self.window, "hidden": self.hidden}


class AnomalyTransformerForecaster:
    """IsolationForest anomaly scores + transformer encoder (anomaly-weighted states)."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.window = int(cfg.get("window", 42))
        self.d_model = int(cfg.get("d_model", 32))
        self.nhead = int(cfg.get("nhead", 2))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.max_epochs = int(cfg.get("max_epochs", 40))
        self.patience = int(cfg.get("patience", 8))
        self.if_contamination = float(cfg.get("if_contamination", 0.05))
        self._trainer: _WindowTrainer | None = None
        self._cols: list[str] = []
        self._iforest: IsolationForest | None = None

    def _anomaly_weights(self, X: pd.DataFrame) -> np.ndarray:
        assert self._iforest is not None
        scores = -self._iforest.decision_function(X[self._cols].to_numpy(dtype=float))
        scores = np.clip(scores, 0, None)
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AnomalyTransformerForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = list(Xf.columns)
        tr_sl, _ = _split_train_val(len(Xf), self.val_frac)
        self._iforest = IsolationForest(
            contamination=self.if_contamination,
            random_state=self.seed,
            n_estimators=int(self.cfg.get("if_estimators", 100)),
        )
        self._iforest.fit(Xf.iloc[tr_sl][self._cols].to_numpy(dtype=float))
        weights = self._anomaly_weights(Xf)
        tr = _WindowTrainer(
            window=self.window,
            hidden=self.d_model,
            lr=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
            net_factory=lambda n: _AnomalyTransformerNet(
                n, self.d_model, self.nhead, self.num_layers, self.dropout
            ),
        )
        tr.fit_arrays(
            Xf[self._cols].to_numpy(dtype=float),
            yf.to_numpy(dtype=float),
            extra=weights.reshape(-1, 1),
        )
        self._trainer = tr
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._trainer is not None and self._iforest is not None
        weights = self._anomaly_weights(X)
        pred = self._trainer.predict_arrays(
            X[self._cols].to_numpy(dtype=float),
            extra=weights.reshape(-1, 1),
        )
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "anomaly_transformer", "window": self.window, "d_model": self.d_model}


class TFTForecaster:
    """Compact temporal fusion transformer for multivariate FX windows."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.window = int(cfg.get("window", 63))
        self.hidden = int(cfg.get("hidden", 32))
        self.nhead = int(cfg.get("nhead", 2))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.max_epochs = int(cfg.get("max_epochs", 45))
        self.patience = int(cfg.get("patience", 8))
        self._trainer: _WindowTrainer | None = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TFTForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = list(Xf.columns)
        tr = _WindowTrainer(
            window=self.window,
            hidden=self.hidden,
            lr=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
            net_factory=lambda n: _SimpleTFTNet(n, self.hidden, self.nhead, self.dropout),
        )
        tr.fit_arrays(Xf[self._cols].to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._trainer = tr
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._trainer is not None
        pred = self._trainer.predict_arrays(X[self._cols].to_numpy(dtype=float))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "tft", "window": self.window, "hidden": self.hidden}


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    del horizon
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "vmd_cnn_bilstm":
        bc = dict(cfg.get("vmd_cnn_bilstm", {}))
        bc["val_frac"] = vf
        return lambda: VMDCNNBiLSTMForecaster(bc, seed, price_col)
    if name == "anomaly_transformer":
        ac = dict(cfg.get("anomaly_transformer", {}))
        ac["val_frac"] = vf
        return lambda: AnomalyTransformerForecaster(ac, seed, price_col)
    if name == "tft":
        tc = dict(cfg.get("tft", {}))
        return lambda: TFTForecaster(tc, seed, price_col)
    raise KeyError(f"unknown tier-II EUR/PLN model {name!r}")
