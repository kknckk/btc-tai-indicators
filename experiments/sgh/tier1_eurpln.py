"""Tier-I EUR/PLN forecasters: PatchTST, stacking (XGB+ARIMA+GRU), PSO-LSTM (TRD-46)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from models._common import TrainConfig, make_windows, set_seed, timed_train_loop
from models.patchtst import PatchTST, PatchTSTConfig

logger = logging.getLogger(__name__)


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


class _GRUNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class _SeqGRU:
    window: int
    hidden: int
    layers: int
    lr: float
    max_epochs: int
    patience: int
    seed: int
    _model: _GRUNet | None = field(default=None, init=False, repr=False)
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
            raise ValueError("not enough GRU windows")
        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]
        model = _GRUNet(X_arr.shape[1], self.hidden, self.layers)
        cfg = TrainConfig(
            lr=self.lr,
            batch_size=min(64, max(1, len(X_tr))),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        self._model, _, _ = timed_train_loop(model, X_tr, y_tr, X_val, y_val, cfg)

    def predict_arrays(self, X_arr: np.ndarray) -> np.ndarray:
        if self._model is None or self._x_scaler is None:
            raise RuntimeError("GRU not fitted")
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


class PatchTSTLevelForecaster:
    """PatchTST with config from YAML; internal val split for early stopping."""

    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = dict(cfg)
        self.seed = seed
        self._inner: PatchTST | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PatchTSTLevelForecaster":
        p = {**self.cfg, "seed": self.seed}
        self._inner = PatchTST(PatchTSTConfig(**p))
        self._inner.fit(X, y)
        self._diag = {
            "epochs": self._inner.epochs_used_,
            "train_s": self._inner.train_seconds_,
            "n_params": self._inner.n_params,
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._inner is not None
        pred = self._inner.predict(X)
        last = X.iloc[:, 0]
        fallback = last.to_numpy(dtype=float)
        out = np.where(np.isfinite(pred), pred, fallback)
        return out

    def diagnostics(self) -> dict:
        return {"model": "patchtst", **self._diag}


class ArimaLevelForecaster:
    """ARIMA(1,0,1) on the spot rate; h-step level forecast per origin."""

    def __init__(self, price_col: str, horizon: int, min_hist: int = 40) -> None:
        self.price_col = price_col
        self.h = horizon
        self.min_hist = min_hist
        self._series: pd.Series | None = None
        self._order = (1, 0, 1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ArimaLevelForecaster":
        self._series = X[self.price_col].reindex(y.index).astype(float)
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


class StackingEnsembleForecaster:
    """Level stacking: XGBoost + ARIMA + GRU → Ridge meta-learner (val fold only)."""

    def __init__(self, cfg: dict, seed: int, horizon: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.h = horizon
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        xgb_cfg = dict(cfg.get("xgb", {}))
        gru_cfg = dict(cfg.get("gru", {}))
        self._xgb_p = {
            "n_estimators": xgb_cfg.get("n_estimators", 200),
            "max_depth": xgb_cfg.get("max_depth", 4),
            "learning_rate": xgb_cfg.get("learning_rate", 0.05),
            "subsample": xgb_cfg.get("subsample", 0.8),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
            "val_fraction": xgb_cfg.get("val_fraction", 0.15),
            "early_stopping_rounds": xgb_cfg.get("early_stopping_rounds", 20),
        }
        self._gru = _SeqGRU(
            window=int(gru_cfg.get("window", 42)),
            hidden=int(gru_cfg.get("hidden", 32)),
            layers=int(gru_cfg.get("layers", 1)),
            lr=float(gru_cfg.get("lr", 1e-3)),
            max_epochs=int(gru_cfg.get("max_epochs", 40)),
            patience=int(gru_cfg.get("patience", 8)),
            seed=seed,
        )
        self._meta: Ridge | None = None
        self._xgb: xgb.XGBRegressor | None = None
        self._arima = ArimaLevelForecaster(price_col, horizon)
        self._cols: list[str] = []
        self._diag: dict = {}

    def _fit_xgb(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        p = dict(self._xgb_p)
        vf = p.pop("val_fraction")
        es = p.pop("early_stopping_rounds")
        cut = max(1, int(len(X) * (1 - vf)))
        model = xgb.XGBRegressor(
            tree_method="hist", random_state=self.seed, early_stopping_rounds=es, **p
        )
        model.fit(
            X.iloc[:cut],
            y.iloc[:cut],
            eval_set=[(X.iloc[cut:], y.iloc[cut:])],
            verbose=False,
        )
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsembleForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = list(Xf.columns)
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        self._xgb = self._fit_xgb(Xf, yf)
        self._arima.fit(Xf, yf)
        self._gru.fit_arrays(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))

        stack_va = np.column_stack(
            [
                self._xgb.predict(X_va[self._cols]),
                self._arima.predict(X_va),
                self._gru.predict_arrays(X_va.to_numpy(dtype=float)),
            ]
        )
        ok = np.isfinite(stack_va).all(axis=1) & y_va.notna().to_numpy()
        if ok.sum() < 5:
            self._meta = Ridge(alpha=1.0)
            self._meta.fit(np.ones((len(y_va), 1)), y_va.to_numpy())
            self._diag = {"meta_fallback": True}
            return self
        self._meta = Ridge(alpha=float(self.cfg.get("ridge_alpha", 1.0)))
        self._meta.fit(stack_va[ok], y_va.to_numpy()[ok])
        self._diag = {
            "meta_coef": self._meta.coef_.tolist(),
            "val_mse": _val_mse(y_va, self._meta.predict(stack_va)),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._xgb is not None and self._meta is not None
        stack = np.column_stack(
            [
                self._xgb.predict(X[self._cols]),
                self._arima.predict(X),
                self._gru.predict_arrays(X.to_numpy(dtype=float)),
            ]
        )
        pred = self._meta.predict(stack)
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "stacking", **self._diag}


class PSOLSTMForecaster:
    """PSO over GRU hyperparameters (hidden, lr, window); best config refit on full train."""

    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.n_particles = int(cfg.get("n_particles", 8))
        self.n_iters = int(cfg.get("n_iters", 4))
        self.gru_defaults = dict(cfg.get("gru", {}))
        self._best: _SeqGRU | None = None
        self._diag: dict = {}

    def _decode(self, vec: np.ndarray) -> _SeqGRU:
        hidden_opts = (16, 32, 64)
        lr_opts = (5e-4, 1e-3, 2e-3)
        win_opts = (21, 42, 63)
        hi = int(hidden_opts[int(vec[0] * len(hidden_opts)) % len(hidden_opts)])
        lr = float(lr_opts[int(vec[1] * len(lr_opts)) % len(lr_opts)])
        win = int(win_opts[int(vec[2] * len(win_opts)) % len(win_opts)])
        g = self.gru_defaults
        return _SeqGRU(
            window=win,
            hidden=hi,
            layers=int(g.get("layers", 1)),
            lr=lr,
            max_epochs=int(g.get("max_epochs", 35)),
            patience=int(g.get("patience", 6)),
            seed=self.seed,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PSOLSTMForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl]
        y_va = yf.iloc[va_sl]

        rng = np.random.default_rng(self.seed)
        particles = rng.uniform(0, 1, size=(self.n_particles, 3))
        velocities = rng.normal(0, 0.05, size=particles.shape)
        pbest = particles.copy()
        pbest_scores = np.full(self.n_particles, np.inf)
        gbest = particles[0]
        gbest_score = np.inf

        def score_particle(vec: np.ndarray) -> float:
            try:
                gru = self._decode(vec)
                gru.fit_arrays(X_tr, y_tr)
                pred = gru.predict_arrays(X_va.to_numpy(dtype=float))
                return _val_mse(y_va, pred)
            except Exception:  # noqa: BLE001
                return float("inf")

        for _ in range(self.n_iters):
            for i in range(self.n_particles):
                s = score_particle(particles[i])
                if s < pbest_scores[i]:
                    pbest_scores[i] = s
                    pbest[i] = particles[i].copy()
                if s < gbest_score:
                    gbest_score = s
                    gbest = particles[i].copy()
            w = 0.6
            c1, c2 = 1.2, 1.2
            r1, r2 = rng.uniform(0, 1, particles.shape), rng.uniform(0, 1, particles.shape)
            velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
            particles = np.clip(particles + velocities, 0, 1)

        self._best = self._decode(gbest)
        self._best.fit_arrays(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._diag = {"best_val_mse": gbest_score, "gbest": gbest.tolist()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._best is not None
        pred = self._best.predict_arrays(X.to_numpy(dtype=float))
        spot = X.iloc[:, 0].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "pso_lstm", **self._diag}


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    if name == "patchtst":
        pt = dict(cfg.get("patchtst", {}))
        return lambda: PatchTSTLevelForecaster(pt, seed)
    if name == "stacking":
        st = dict(cfg.get("stacking", {}))
        st["val_frac"] = float(cfg.get("val_frac", 0.2))
        return lambda: StackingEnsembleForecaster(st, seed, horizon, price_col)
    if name == "pso_lstm":
        pl = dict(cfg.get("pso_lstm", {}))
        pl["val_frac"] = float(cfg.get("val_frac", 0.2))
        return lambda: PSOLSTMForecaster(pl, seed)
    raise KeyError(f"unknown tier-I EUR/PLN model {name!r}")
