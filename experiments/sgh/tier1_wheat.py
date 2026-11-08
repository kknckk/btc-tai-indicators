"""Tier-I wheat futures forecasters: LSTM, ARIMA, XGBoost (TRD-52)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from models._common import TrainConfig, make_windows, set_seed, timed_train_loop


def _split_train_val(n: int, val_frac: float) -> tuple[slice, slice]:
    cut = max(30, int(n * (1.0 - val_frac)))
    if cut >= n - 10:
        cut = n // 2
    return slice(0, cut), slice(cut, n)


def _val_mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    m = y_true.notna().to_numpy() & np.isfinite(y_pred)
    if not m.any():
        return float("inf")
    return float(np.mean((y_true.to_numpy()[m] - y_pred[m]) ** 2))


class _LSTMNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class _SeqLSTM:
    window: int
    hidden: int
    layers: int
    lr: float
    max_epochs: int
    patience: int
    seed: int
    _model: _LSTMNet | None = field(default=None, init=False, repr=False)
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
            raise ValueError("not enough LSTM windows")
        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]
        model = _LSTMNet(X_arr.shape[1], self.hidden, self.layers)
        cfg = TrainConfig(
            lr=self.lr,
            batch_size=min(64, max(1, len(X_tr))),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        self._model, _, _ = timed_train_loop(model, X_tr, y_tr, X_val, y_val, cfg)

    def predict_arrays(self, X_arr: np.ndarray) -> np.ndarray:
        if self._model is None or self._x_scaler is None:
            raise RuntimeError("LSTM not fitted")
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


class LSTMLevelForecaster:
    """LSTM with (window, hidden, lr) chosen on the validation tail of each train fold."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.window_grid = tuple(cfg.get("window_grid", (21, 42, 63)))
        self.hidden_grid = tuple(cfg.get("hidden_grid", (16, 32, 64)))
        self.lr_grid = tuple(cfg.get("lr_grid", (5e-4, 1e-3, 2e-3)))
        self.max_epochs = int(cfg.get("max_epochs", 40))
        self.patience = int(cfg.get("patience", 8))
        self.layers = int(cfg.get("layers", 1))
        self._best: _SeqLSTM | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMLevelForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl]
        y_va = yf.iloc[va_sl]

        best_score = float("inf")
        best_cfg: _SeqLSTM | None = None
        for window in self.window_grid:
            for hidden in self.hidden_grid:
                for lr in self.lr_grid:
                    try:
                        lstm = _SeqLSTM(
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
            best_cfg = _SeqLSTM(
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
        return {"model": "lstm", **self._diag}


class ArimaLevelForecaster:
    """ARIMA on spot price; order chosen on validation; h-step level forecast per origin."""

    def __init__(self, price_col: str, horizon: int, cfg: dict) -> None:
        self.price_col = price_col
        self.h = horizon
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.min_hist = int(cfg.get("min_hist", 40))
        raw = cfg.get("order_grid", [(1, 0, 1), (2, 0, 1), (1, 0, 2)])
        self.order_grid = [tuple(o) for o in raw]
        self._order: tuple[int, int, int] = (1, 0, 1)
        self._series: pd.Series | None = None
        self._diag: dict = {}

    def _pick_order(self, series: pd.Series) -> tuple[int, int, int]:
        n = len(series)
        cut = max(self.min_hist, int(n * (1.0 - self.val_frac)))
        if cut >= n - 5:
            cut = n // 2
        hist = series.iloc[:cut]
        va = series.iloc[cut:]
        best = (float("inf"), self.order_grid[0])
        for order in self.order_grid:
            try:
                res = ARIMA(hist, order=order).fit()
                fc = res.forecast(steps=len(va))
                mse = float(np.mean((va.to_numpy() - fc.to_numpy()) ** 2))
                if mse < best[0]:
                    best = (mse, order)
            except Exception:  # noqa: BLE001
                continue
        return best[1]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ArimaLevelForecaster":
        series = X[self.price_col].reindex(y.index).astype(float)
        self._order = self._pick_order(series.dropna())
        self._series = series
        self._diag = {"order": list(self._order)}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._series is not None
        combined = self._series.copy()
        out = np.full(len(X), np.nan, dtype=float)
        for j, t in enumerate(X.index):
            combined.loc[t] = float(X[self.price_col].loc[t])
            hist = combined.loc[:t].dropna()
            if len(hist) < self.min_hist:
                out[j] = float(hist.iloc[-1]) if len(hist) else np.nan
                continue
            try:
                res = ARIMA(hist, order=self._order).fit()
                fc = res.forecast(steps=self.h)
                out[j] = float(fc.iloc[-1])
            except Exception:  # noqa: BLE001
                out[j] = float(hist.iloc[-1])
        return out

    def diagnostics(self) -> dict:
        return {"model": "arima", **self._diag}


class XGBLevelForecaster:
    """XGBoost on tabular features; depth and learning rate from validation MSE."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.max_depth_grid = tuple(cfg.get("max_depth_grid", (3, 4, 6)))
        self.lr_grid = tuple(cfg.get("learning_rate_grid", (0.03, 0.05, 0.1)))
        self.n_estimators = int(cfg.get("n_estimators", 200))
        self.subsample = float(cfg.get("subsample", 0.8))
        self.colsample = float(cfg.get("colsample_bytree", 0.8))
        self.early_stopping = int(cfg.get("early_stopping_rounds", 20))
        self.n_jobs = int(cfg.get("n_jobs", 4))
        self._model: xgb.XGBRegressor | None = None
        self._cols: list[str] = []
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBLevelForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = list(Xf.columns)
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr, y_tr = Xf.iloc[tr_sl], yf.iloc[tr_sl]
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        best = (float("inf"), self.max_depth_grid[0], self.lr_grid[0])
        for depth in self.max_depth_grid:
            for lr in self.lr_grid:
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=int(depth),
                    learning_rate=float(lr),
                    subsample=self.subsample,
                    colsample_bytree=self.colsample,
                    tree_method="hist",
                    random_state=self.seed,
                    early_stopping_rounds=self.early_stopping,
                    n_jobs=self.n_jobs,
                )
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
                pred = model.predict(X_va)
                mse = _val_mse(y_va, pred)
                if mse < best[0]:
                    best = (mse, depth, lr)

        _, depth, lr = best
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=int(depth),
            learning_rate=float(lr),
            subsample=self.subsample,
            colsample_bytree=self.colsample,
            tree_method="hist",
            random_state=self.seed,
            early_stopping_rounds=self.early_stopping,
            n_jobs=self.n_jobs,
        )
        cut = max(1, int(len(Xf) * (1 - 0.15)))
        self._model.fit(
            Xf.iloc[:cut],
            yf.iloc[:cut],
            eval_set=[(Xf.iloc[cut:], yf.iloc[cut:])],
            verbose=False,
        )
        self._diag = {"max_depth": int(depth), "learning_rate": float(lr), "val_mse": best[0]}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        pred = self._model.predict(X[self._cols])
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "xgboost", **self._diag}


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "lstm":
        lc = dict(cfg.get("lstm", {}))
        lc["val_frac"] = vf
        return lambda: LSTMLevelForecaster(lc, seed, price_col)
    if name == "arima":
        ac = dict(cfg.get("arima", {}))
        ac["val_frac"] = vf
        return lambda: ArimaLevelForecaster(price_col, horizon, ac)
    if name == "xgboost":
        xc = dict(cfg.get("xgboost", {}))
        xc["val_frac"] = vf
        return lambda: XGBLevelForecaster(xc, seed, price_col)
    raise KeyError(f"unknown tier-I wheat model {name!r}")
