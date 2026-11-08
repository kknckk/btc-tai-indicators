"""Tier-I WTI oil forecasters: LSTM-SVR, XGBoost, GARCH-MIDAS (TRD-54)."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from experiments.sgh.tier1_wheat import XGBLevelForecaster, _split_train_val, _val_mse
from models._common import TrainConfig, make_windows, set_seed, timed_train_loop

logger = logging.getLogger(__name__)


class _LSTMEncoder(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]


class _LSTMSVRNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int = 1):
        super().__init__()
        self.encoder = _LSTMEncoder(n_feat, hidden, layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


@dataclass
class _LSTMSVRBundle:
    window: int
    hidden: int
    layers: int
    lr: float
    max_epochs: int
    patience: int
    seed: int
    svr_c: float
    svr_gamma: str | float
    _encoder: _LSTMEncoder | None = field(default=None, init=False, repr=False)
    _svr: SVR | None = field(default=None, init=False, repr=False)
    _x_scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _y_mu: float = field(default=0.0, init=False)
    _y_sd: float = field(default=1.0, init=False)

    def _gamma(self, n_feat: int) -> float | str:
        if self.svr_gamma == "scale":
            return "scale"
        if self.svr_gamma == "auto":
            return "auto"
        return float(self.svr_gamma)

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
        if len(Xw) < 12:
            raise ValueError("not enough LSTM-SVR windows")

        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]

        model = _LSTMSVRNet(X_arr.shape[1], self.hidden, self.layers)

        cfg = TrainConfig(
            lr=self.lr,
            batch_size=min(64, max(1, len(X_tr))),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        trained, _, _ = timed_train_loop(model, X_tr, y_tr, X_val, y_val, cfg)
        self._encoder = trained.encoder

        def embed(Xw_batch: np.ndarray) -> np.ndarray:
            self._encoder.eval()
            with torch.no_grad():
                t = torch.tensor(Xw_batch, dtype=torch.float32)
                return self._encoder(t).numpy()

        z_tr = embed(X_tr)
        self._svr = SVR(C=self.svr_c, gamma=self._gamma(X_arr.shape[1]), epsilon=0.01)
        self._svr.fit(z_tr, y_tr * self._y_sd + self._y_mu)

    def predict_arrays(self, X_arr: np.ndarray) -> np.ndarray:
        if self._encoder is None or self._svr is None or self._x_scaler is None:
            raise RuntimeError("LSTM-SVR not fitted")
        xs = self._x_scaler.transform(X_arr)
        Xw, _, idx = make_windows(xs, None, self.window)
        out = np.full(len(X_arr), np.nan, dtype=float)
        if not len(idx):
            return out
        self._encoder.eval()
        with torch.no_grad():
            z = self._encoder(torch.tensor(Xw, dtype=torch.float32)).numpy()
        pred = self._svr.predict(z)
        out[np.array(idx)] = pred
        return out


class LSTMSVRLevelForecaster:
    """Two-stage LSTM representation + SVR; grids chosen on validation tail of each fold."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.window_grid = tuple(cfg.get("window_grid", (21, 42, 63)))
        self.hidden_grid = tuple(cfg.get("hidden_grid", (16, 32)))
        self.lr_grid = tuple(cfg.get("lr_grid", (5e-4, 1e-3)))
        self.max_epochs = int(cfg.get("max_epochs", 40))
        self.patience = int(cfg.get("patience", 8))
        self.layers = int(cfg.get("layers", 1))
        raw_c = cfg.get("svr_c_grid", (0.1, 1.0, 10.0))
        self.svr_c_grid = tuple(raw_c)
        raw_g = cfg.get("svr_gamma_grid", ("scale", "auto"))
        self.svr_gamma_grid = tuple(raw_g)
        self._best: _LSTMSVRBundle | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMSVRLevelForecaster":
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl]
        y_va = yf.iloc[va_sl]

        best_score = float("inf")
        best_cfg: _LSTMSVRBundle | None = None
        for window in self.window_grid:
            for hidden in self.hidden_grid:
                for lr in self.lr_grid:
                    for c in self.svr_c_grid:
                        for gamma in self.svr_gamma_grid:
                            try:
                                bundle = _LSTMSVRBundle(
                                    window=int(window),
                                    hidden=int(hidden),
                                    layers=self.layers,
                                    lr=float(lr),
                                    max_epochs=self.max_epochs,
                                    patience=self.patience,
                                    seed=self.seed,
                                    svr_c=float(c),
                                    svr_gamma=gamma,
                                )
                                bundle.fit_arrays(X_tr, y_tr)
                                pred = bundle.predict_arrays(X_va.to_numpy(dtype=float))
                                score = _val_mse(y_va, pred)
                                if score < best_score:
                                    best_score = score
                                    best_cfg = bundle
                            except (ValueError, RuntimeError):
                                continue

        if best_cfg is None:
            best_cfg = _LSTMSVRBundle(
                window=int(self.window_grid[0]),
                hidden=int(self.hidden_grid[0]),
                layers=self.layers,
                lr=float(self.lr_grid[0]),
                max_epochs=self.max_epochs,
                patience=self.patience,
                seed=self.seed,
                svr_c=float(self.svr_c_grid[0]),
                svr_gamma=self.svr_gamma_grid[0],
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
        return {"model": "lstm_svr", **self._diag}


def _midas_weights(k_max: int, theta1: float, theta2: float) -> np.ndarray:
    k = np.arange(1, k_max + 1, dtype=float)
    x = (k - 0.5) / k_max
    w = np.power(x, theta1 - 1.0) * np.power(1.0 - x, theta2 - 1.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(k_max) / k_max


def _midas_tau(ret: pd.Series, k_max: int, theta1: float, theta2: float, rv_window: int) -> pd.Series:
    rv = ret.pow(2).rolling(rv_window, min_periods=rv_window).sum()
    w = _midas_weights(k_max, theta1, theta2)
    tau = pd.Series(0.0, index=ret.index)
    for i, wi in enumerate(w):
        lag = rv_window * (i + 1)
        tau = tau + wi * rv.shift(lag)
    return tau.clip(lower=1e-12)


@dataclass
class GarchMidasLevelForecaster:
    """GARCH mean on MIDAS-scaled returns (22d RV weights) → h-day WTI level."""

    returns: pd.Series
    price_col: str
    horizon: int
    val_frac: float = 0.2
    k_grid: tuple[int, ...] = (3, 4, 6)
    theta1_grid: tuple[float, ...] = (1.0, 2.0)
    theta2_grid: tuple[float, ...] = (5.0, 10.0)
    mean_lags_grid: tuple[int, ...] = (0, 1)
    dist_grid: tuple[str, ...] = ("normal", "t")
    rv_window: int = 22
    state: dict = field(default_factory=dict)
    _k: int = field(default=3, init=False)
    _t1: float = field(default=1.0, init=False)
    _t2: float = field(default=5.0, init=False)
    _lags: int = field(default=0, init=False)
    _dist: str = field(default="normal", init=False)
    _params: pd.Series | None = field(default=None, init=False)

    def _scaled_returns(self, ret: pd.Series, k: int, t1: float, t2: float) -> pd.Series:
        tau = _midas_tau(ret, k, t1, t2, self.rv_window)
        return (ret / np.sqrt(tau)).replace([np.inf, -np.inf], np.nan)

    def _arch(self, series: pd.Series, lags: int, dist: str):
        scaled = series.astype(float) * 100.0
        mean = "Constant" if lags == 0 else "AR"
        lags_arg = None if lags == 0 else lags
        return arch_model(
            scaled,
            mean=mean,
            lags=lags_arg,
            vol="GARCH",
            p=1,
            q=1,
            dist=dist,
            rescale=False,
        )

    def _fit_params(self, ret: pd.Series, k: int, t1: float, t2: float, lags: int, dist: str) -> pd.Series:
        sr = self._scaled_returns(ret, k, t1, t2).dropna()
        if len(sr) < 40:
            raise ValueError("too few MIDAS-scaled returns")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._arch(sr, lags, dist).fit(disp="off", options={"maxiter": 500})
        if res.convergence_flag != 0 and "params" in self.state:
            raise RuntimeError(f"no convergence (flag {res.convergence_flag})")
        return res.params

    def _level_at(
        self,
        price: float,
        ret_upto: pd.Series,
        params: pd.Series,
        k: int,
        t1: float,
        t2: float,
        lags: int,
        dist: str,
    ) -> float:
        sr = self._scaled_returns(ret_upto, k, t1, t2).dropna()
        if len(sr) < max(10, lags + 5):
            return float(price)
        tau_last = float(_midas_tau(ret_upto, k, t1, t2, self.rv_window).dropna().iloc[-1])
        scaled = sr * 100.0
        pos = len(scaled) - 1
        res = self._arch(scaled, lags, dist).fix(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = res.forecast(horizon=self.horizon, start=pos, reindex=False, method="analytic")
        mu_scaled = float(f.mean.iloc[-1].sum()) / 100.0
        mu_h = mu_scaled * np.sqrt(max(tau_last, 1e-12))
        return float(price * np.exp(mu_h))

    def _predict_slice(
        self,
        X: pd.DataFrame,
        params: pd.Series,
        k: int,
        t1: float,
        t2: float,
        lags: int,
        dist: str,
    ) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=float)
        for j, t in enumerate(X.index):
            hist = self.returns.loc[:t].dropna()
            price = float(X[self.price_col].loc[t])
            try:
                out[j] = self._level_at(price, hist, params, k, t1, t2, lags, dist)
            except Exception:  # noqa: BLE001
                out[j] = price
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GarchMidasLevelForecaster":
        ret = self.returns.reindex(X.index)
        n = len(X)
        tr_sl, va_sl = _split_train_val(n, self.val_frac)
        X_va, y_va = X.iloc[va_sl], y.iloc[va_sl]
        best = (float("inf"), self.k_grid[0], self.theta1_grid[0], self.theta2_grid[0], 0, "normal")
        for k in self.k_grid:
            for t1 in self.theta1_grid:
                for t2 in self.theta2_grid:
                    for lags in self.mean_lags_grid:
                        for dist in self.dist_grid:
                            try:
                                params = self._fit_params(ret.iloc[tr_sl], k, t1, t2, lags, dist)
                                preds = self._predict_slice(X_va, params, k, t1, t2, lags, dist)
                                mse = _val_mse(y_va, preds)
                                if mse < best[0]:
                                    best = (mse, k, t1, t2, lags, dist)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("garch-midas grid failed k=%s: %s", k, exc)
        _, self._k, self._t1, self._t2, self._lags, self._dist = best
        try:
            self._params = self._fit_params(ret, self._k, self._t1, self._t2, self._lags, self._dist)
            self.state["params"] = self._params
        except Exception as exc:  # noqa: BLE001
            self.state["failures"] = self.state.get("failures", 0) + 1
            logger.warning("garch-midas fit failed (%s); reusing previous params", exc)
            if "params" not in self.state:
                raise
            self._params = self.state["params"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        return self._predict_slice(X, self._params, self._k, self._t1, self._t2, self._lags, self._dist)

    def diagnostics(self) -> dict:
        return {
            "model": "garch_midas",
            "k": self._k,
            "theta1": self._t1,
            "theta2": self._t2,
            "mean_lags": self._lags,
            "dist": self._dist,
            "fit_failures": self.state.get("failures", 0),
        }


def model_factory(
    name: str,
    cfg: dict,
    *,
    horizon: int,
    price_col: str,
    returns: pd.Series | None = None,
    states: dict[str, dict] | None = None,
) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    states = states if states is not None else {}
    if name == "lstm_svr":
        lc = dict(cfg.get("lstm_svr", {}))
        lc["val_frac"] = vf
        return lambda: LSTMSVRLevelForecaster(lc, seed, price_col)
    if name == "xgboost":
        xc = dict(cfg.get("xgboost", {}))
        xc["val_frac"] = vf
        return lambda: XGBLevelForecaster(xc, seed, price_col)
    if name == "garch_midas":
        if returns is None:
            raise ValueError("garch_midas requires returns series")
        gm = dict(cfg.get("garch_midas", {}))
        return lambda: GarchMidasLevelForecaster(
            returns,
            price_col,
            horizon,
            val_frac=vf,
            k_grid=tuple(gm.get("k_grid", (3, 4, 6))),
            theta1_grid=tuple(gm.get("theta1_grid", (1.0, 2.0))),
            theta2_grid=tuple(gm.get("theta2_grid", (5.0, 10.0))),
            mean_lags_grid=tuple(gm.get("mean_lags_grid", (0, 1))),
            dist_grid=tuple(gm.get("dist_grid", ("normal", "t"))),
            rv_window=int(gm.get("rv_window", 22)),
            state=states.setdefault("garch_midas", {}),
        )
    raise KeyError(f"unknown tier-I oil model {name!r}")
