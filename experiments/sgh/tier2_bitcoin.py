"""Tier-II Bitcoin forecasters: LSTM, TFT, SVM, HellFormer (TRD-59)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from experiments.sgh.tier1_wheat import LSTMLevelForecaster, _split_train_val, _val_mse
from experiments.sgh.tier2_eurpln import TFTForecaster
from models._common import TrainConfig, make_windows, set_seed, timed_train_loop


def _feature_cols(X: pd.DataFrame, price_col: str) -> list[str]:
    return [c for c in X.columns if c != price_col and X[c].dtype != object]


class _HellFormerNet(nn.Module):
    """Lightweight HellFormer-style block: heat-smoothed multi-scale lags + hierarchical transformer."""

    def __init__(
        self,
        n_feat: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        kernel_sizes: tuple[int, ...],
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_feat, d_model, k, padding=k // 2),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )
        self.fuse = nn.Linear(d_model * len(kernel_sizes), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zt = x.transpose(1, 2)
        parts = [branch(zt).transpose(1, 2) for branch in self.branches]
        z = torch.cat(parts, dim=-1)
        z = self.fuse(z)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)


@dataclass
class _SeqHellFormer:
    window: int
    d_model: int
    nhead: int
    num_layers: int
    dropout: float
    kernel_sizes: tuple[int, ...]
    lr: float
    max_epochs: int
    patience: int
    seed: int
    _model: _HellFormerNet | None = field(default=None, init=False, repr=False)
    _x_scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _y_mu: float = field(default=0.0, init=False)
    _y_sd: float = field(default=1.0, init=False)

    def fit_arrays(self, X_arr: np.ndarray, y_arr: np.ndarray) -> None:
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
            raise ValueError("not enough HellFormer windows")
        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]
        model = _HellFormerNet(
            X_arr.shape[1],
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dropout,
            self.kernel_sizes,
        )
        cfg = TrainConfig(
            lr=self.lr,
            batch_size=min(64, max(1, len(X_tr))),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        self._model, _, _ = timed_train_loop(model, X_tr, y_tr, X_val, y_val, cfg)

    def predict_arrays(self, X_arr: np.ndarray) -> np.ndarray:
        if self._model is None or self._x_scaler is None:
            raise RuntimeError("HellFormer not fitted")
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


class HellFormerForecaster:
    """Multi-scale heat convolutions + transformer; window chosen on fold validation tail."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.window_grid = tuple(cfg.get("window_grid", (30, 60, 90)))
        self.d_model = int(cfg.get("d_model", 32))
        self.nhead = int(cfg.get("nhead", 2))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.dropout = float(cfg.get("dropout", 0.1))
        ks = cfg.get("kernel_sizes", (3, 7, 15))
        self.kernel_sizes = tuple(int(k) for k in ks)
        self.lr = float(cfg.get("lr", 1e-3))
        self.max_epochs = int(cfg.get("max_epochs", 45))
        self.patience = int(cfg.get("patience", 8))
        self._best: _SeqHellFormer | None = None
        self._cols: list[str] = []
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HellFormerForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = _feature_cols(Xf, self.price_col)
        use = self._cols + [self.price_col]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl][use].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl][use]
        y_va = yf.iloc[va_sl]

        best_score = float("inf")
        best_cfg: _SeqHellFormer | None = None
        for window in self.window_grid:
            try:
                hf = _SeqHellFormer(
                    window=int(window),
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    kernel_sizes=self.kernel_sizes,
                    lr=self.lr,
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    seed=self.seed,
                )
                hf.fit_arrays(X_tr, y_tr)
                pred = hf.predict_arrays(X_va.to_numpy(dtype=float))
                score = _val_mse(y_va, pred)
                if score < best_score:
                    best_score = score
                    best_cfg = hf
            except (ValueError, RuntimeError):
                continue

        if best_cfg is None:
            best_cfg = _SeqHellFormer(
                window=int(self.window_grid[0]),
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.dropout,
                kernel_sizes=self.kernel_sizes,
                lr=self.lr,
                max_epochs=self.max_epochs,
                patience=self.patience,
                seed=self.seed,
            )
            best_cfg.fit_arrays(Xf[use].to_numpy(dtype=float), yf.to_numpy(dtype=float))
        else:
            best_cfg.fit_arrays(Xf[use].to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._best = best_cfg
        self._diag = {"val_mse": best_score}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._best is not None
        use = self._cols + [self.price_col]
        pred = self._best.predict_arrays(X[use].to_numpy(dtype=float))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "hellformer", **self._diag}


class SVMLevelForecaster:
    """RBF SVR on tabular on-chain features; C chosen on validation tail."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.c_grid = tuple(cfg.get("c_grid", (0.1, 1.0, 10.0, 100.0)))
        self.epsilon = float(cfg.get("epsilon", 0.01))
        self._model: SVR | None = None
        self._scaler: StandardScaler | None = None
        self._cols: list[str] = []
        self._c: float = float(self.c_grid[0])
        self._diag: dict = {}

    def _gamma(self, n_feat: int) -> float:
        return 1.0 / max(n_feat, 1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SVMLevelForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = _feature_cols(Xf, self.price_col)
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl][self._cols].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl][self._cols].to_numpy(dtype=float)
        y_va = yf.iloc[va_sl].to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        gamma = self._gamma(len(self._cols))

        best_score = float("inf")
        best_c = self.c_grid[0]
        for c in self.c_grid:
            model = SVR(C=float(c), gamma=gamma, epsilon=self.epsilon)
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_va_s)
            score = _val_mse(pd.Series(y_va), pred)
            if score < best_score:
                best_score = score
                best_c = float(c)

        self._scaler = scaler
        self._c = best_c
        self._model = SVR(C=self._c, gamma=gamma, epsilon=self.epsilon)
        self._model.fit(scaler.transform(Xf[self._cols].to_numpy(dtype=float)), yf.to_numpy(dtype=float))
        self._diag = {"val_mse": best_score, "C": self._c}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None and self._scaler is not None
        pred = self._model.predict(self._scaler.transform(X[self._cols].to_numpy(dtype=float)))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "svm", **self._diag}


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    del horizon
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "lstm":
        lc = dict(cfg.get("lstm", {}))
        lc["val_frac"] = vf
        return lambda: LSTMLevelForecaster(lc, seed, price_col)
    if name == "tft":
        tc = dict(cfg.get("tft", {}))
        return lambda: TFTForecaster(tc, seed, price_col)
    if name == "svm":
        sc = dict(cfg.get("svm", {}))
        sc["val_frac"] = vf
        return lambda: SVMLevelForecaster(sc, seed, price_col)
    if name == "hellformer":
        hc = dict(cfg.get("hellformer", {}))
        hc["val_frac"] = vf
        return lambda: HellFormerForecaster(hc, seed, price_col)
    raise KeyError(f"unknown tier-II bitcoin model {name!r}")

