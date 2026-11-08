"""Tier-I Bitcoin forecasters: BiLSTM, XGBoost (TRD-58)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.sgh.tier1_wheat import XGBLevelForecaster, _split_train_val, _val_mse
from models._common import TrainConfig, make_windows, set_seed, timed_train_loop


class _BiLSTMNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True, bidirectional=True
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class _SeqBiLSTM:
    window: int
    hidden: int
    layers: int
    lr: float
    max_epochs: int
    patience: int
    seed: int
    _model: _BiLSTMNet | None = field(default=None, init=False, repr=False)
    _x_scaler: object | None = field(default=None, init=False, repr=False)
    _y_mu: float = field(default=0.0, init=False)
    _y_sd: float = field(default=1.0, init=False)

    def fit_arrays(self, X_arr: np.ndarray, y_arr: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler

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
            raise ValueError("not enough BiLSTM windows")
        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]
        model = _BiLSTMNet(X_arr.shape[1], self.hidden, self.layers)
        cfg = TrainConfig(
            lr=self.lr,
            batch_size=min(64, max(1, len(X_tr))),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        self._model, _, _ = timed_train_loop(model, X_tr, y_tr, X_val, y_val, cfg)

    def predict_arrays(self, X_arr: np.ndarray) -> np.ndarray:
        if self._model is None or self._x_scaler is None:
            raise RuntimeError("BiLSTM not fitted")
        xs = self._x_scaler.transform(X_arr)
        Xw, _, idx = make_windows(xs, None, self.window)
        out = np.full(len(X_arr), np.nan, dtype=float)
        if not len(idx):
            return out
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.tensor(Xw, dtype=torch.float32)).numpy()
        out[np.array(idx)] = pred * self._y_sd + self._y_mu
        return out


class BiLSTMLevelForecaster:
    """BiLSTM with (window, hidden, lr) chosen on the validation tail of each train fold."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.window_grid = tuple(cfg.get("window_grid", (30, 60, 90)))
        self.hidden_grid = tuple(cfg.get("hidden_grid", (16, 32, 64)))
        self.lr_grid = tuple(cfg.get("lr_grid", (5e-4, 1e-3, 2e-3)))
        self.max_epochs = int(cfg.get("max_epochs", 40))
        self.patience = int(cfg.get("patience", 8))
        self.layers = int(cfg.get("layers", 1))
        self._best: _SeqBiLSTM | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BiLSTMLevelForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl]
        y_va = yf.iloc[va_sl]

        best_score = float("inf")
        best_cfg: _SeqBiLSTM | None = None
        for window in self.window_grid:
            for hidden in self.hidden_grid:
                for lr in self.lr_grid:
                    try:
                        lstm = _SeqBiLSTM(
                            window=int(window),
                            hidden=int(hidden),
                            layers=self.layers,
                            lr=float(lr),
                            max_epochs=self.max_epochs,
                            patience=self.patience,
                            seed=self.seed,
                        )
                        lstm.fit_arrays(X_tr, y_tr)
                        pred = lstm.predict_arrays(X_va.to_numpy(dtype=float))
                        score = _val_mse(y_va, pred)
                        if score < best_score:
                            best_score = score
                            best_cfg = lstm
                    except (ValueError, RuntimeError):
                        continue

        if best_cfg is None:
            best_cfg = _SeqBiLSTM(
                window=int(self.window_grid[0]),
                hidden=int(self.hidden_grid[0]),
                layers=self.layers,
                lr=float(self.lr_grid[0]),
                max_epochs=self.max_epochs,
                patience=self.patience,
                seed=self.seed,
            )
            best_cfg.fit_arrays(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))
        else:
            best_cfg.fit_arrays(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._best = best_cfg
        self._diag = {"val_mse": best_score}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._best is not None
        pred = self._best.predict_arrays(X.to_numpy(dtype=float))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "bilstm", **self._diag}


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    del horizon
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "bilstm":
        bc = dict(cfg.get("bilstm", {}))
        bc["val_frac"] = vf
        return lambda: BiLSTMLevelForecaster(bc, seed, price_col)
    if name == "xgboost":
        xc = dict(cfg.get("xgboost", {}))
        xc["val_frac"] = vf
        return lambda: XGBLevelForecaster(xc, seed, price_col)
    raise KeyError(f"unknown tier-I bitcoin model {name!r}")
