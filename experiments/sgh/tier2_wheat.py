"""Tier-II wheat futures forecasters: SARIMA-LSTM, FEDformer, GWRFR (TRD-53)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from experiments.sgh.tier1_wheat import LSTMLevelForecaster, _split_train_val, _val_mse
from experiments.sgh.tier2_eurpln import _WindowTrainer
from experiments.sgh.wheat_panel import PRICE_COL


def _ml_feature_cols(X: pd.DataFrame) -> list[str]:
    skip = {PRICE_COL}
    return [c for c in X.columns if c not in skip and X[c].dtype != object]


class _FEDformerNet(nn.Module):
    """Lightweight FEDformer-style block: trend/seasonal split + Fourier mixing + transformer."""

    def __init__(self, n_feat: int, d_model: int, nhead: int, num_layers: int, dropout: float, n_freq: int):
        super().__init__()
        self.n_freq = n_freq
        self.trend_in = nn.Linear(n_feat, d_model)
        self.season_in = nn.Linear(n_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def _fourier_mix(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x, dim=1)
        k = min(self.n_freq, xf.shape[1])
        xf = xf.clone()
        if xf.shape[1] > k:
            xf[:, k:] = 0
        return torch.fft.irfft(xf, n=x.shape[1], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = x.mean(dim=1, keepdim=True).expand_as(x)
        seasonal = x - trend
        seasonal = self._fourier_mix(seasonal)
        z = self.trend_in(trend) + self.season_in(seasonal)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)


@dataclass
class SarimaLevelCore:
    price_col: str
    horizon: int
    val_frac: float
    min_hist: int
    order_grid: list[tuple[int, int, int]]
    seasonal_grid: list[tuple[int, int, int, int]]
    _order: tuple[int, int, int] = field(default=(1, 0, 1), init=False)
    _seasonal: tuple[int, int, int, int] = field(default=(0, 0, 0, 0), init=False)
    _series: pd.Series | None = field(default=None, init=False)

    def _pick_spec(self, series: pd.Series) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
        n = len(series)
        cut = max(self.min_hist, int(n * (1.0 - self.val_frac)))
        if cut >= n - 5:
            cut = n // 2
        hist = series.iloc[:cut]
        va = series.iloc[cut:]
        best = (float("inf"), self.order_grid[0], self.seasonal_grid[0])
        for order in self.order_grid:
            for seasonal in self.seasonal_grid:
                try:
                    if seasonal == (0, 0, 0, 0):
                        res = ARIMA(hist, order=order).fit()
                    else:
                        res = SARIMAX(hist, order=order, seasonal_order=seasonal).fit(disp=False)
                    fc = res.forecast(steps=len(va))
                    mse = float(np.mean((va.to_numpy() - fc.to_numpy()) ** 2))
                    if mse < best[0]:
                        best = (mse, order, seasonal)
                except Exception:  # noqa: BLE001
                    continue
        return best[1], best[2]

    def _forecast(self, hist: pd.Series) -> float:
        if len(hist) < self.min_hist:
            return float(hist.iloc[-1]) if len(hist) else float("nan")
        try:
            if self._seasonal == (0, 0, 0, 0):
                res = ARIMA(hist, order=self._order).fit()
            else:
                res = SARIMAX(hist, order=self._order, seasonal_order=self._seasonal).fit(disp=False)
            fc = res.forecast(steps=self.horizon)
            return float(fc.iloc[-1])
        except Exception:  # noqa: BLE001
            return float(hist.iloc[-1])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SarimaLevelCore:
        series = X[self.price_col].reindex(y.index).astype(float)
        self._order, self._seasonal = self._pick_spec(series.dropna())
        self._series = series
        return self

    def predict_slice(self, X: pd.DataFrame) -> np.ndarray:
        assert self._series is not None
        combined = self._series.copy()
        out = np.full(len(X), np.nan, dtype=float)
        for j, t in enumerate(X.index):
            combined.loc[t] = float(X[self.price_col].loc[t])
            hist = combined.loc[:t].dropna()
            out[j] = self._forecast(hist)
        return out


class SarimaLSTMForecaster:
    """SARIMA h-step level + LSTM on tabular features → Ridge stack (validation only)."""

    def __init__(self, cfg: dict, seed: int, horizon: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.h = horizon
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        lstm_cfg = dict(cfg.get("lstm", {}))
        lstm_cfg["val_frac"] = self.val_frac
        self._lstm = LSTMLevelForecaster(lstm_cfg, seed, price_col)
        raw_o = cfg.get("order_grid", [(1, 0, 1), (2, 0, 1), (1, 0, 2)])
        raw_s = cfg.get("seasonal_grid", [(0, 0, 0, 0), (1, 0, 1, 5), (1, 0, 1, 21)])
        self._sarima = SarimaLevelCore(
            price_col=price_col,
            horizon=horizon,
            val_frac=self.val_frac,
            min_hist=int(cfg.get("min_hist", 60)),
            order_grid=[tuple(o) for o in raw_o],
            seasonal_grid=[tuple(s) for s in raw_s],
        )
        self._meta: Ridge | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SarimaLSTMForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        self._sarima.fit(Xf, yf)
        self._lstm.fit(Xf, yf)

        stack_va = np.column_stack([self._sarima.predict_slice(X_va), self._lstm.predict(X_va)])
        ok = np.isfinite(stack_va).all(axis=1) & y_va.notna().to_numpy()
        self._meta = Ridge(alpha=float(self.cfg.get("ridge_alpha", 1.0)))
        if ok.sum() < 5:
            self._meta.fit(np.ones((len(y_va), 1)), y_va.to_numpy())
            self._diag = {"meta_fallback": True}
            return self
        self._meta.fit(stack_va[ok], y_va.to_numpy()[ok])
        self._diag = {
            "order": list(self._sarima._order),
            "seasonal_order": list(self._sarima._seasonal),
            "val_mse": _val_mse(y_va, self._meta.predict(stack_va)),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._meta is not None
        stack = np.column_stack([self._sarima.predict_slice(X), self._lstm.predict(X)])
        pred = self._meta.predict(stack)
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "sarima_lstm", **self._diag}


class FEDformerForecaster:
    """Fourier-enhanced decomposed transformer on multivariate wheat windows."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.window = int(cfg.get("window", 42))
        self.d_model = int(cfg.get("d_model", 32))
        self.nhead = int(cfg.get("nhead", 2))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.n_freq = int(cfg.get("n_freq", 4))
        self.lr = float(cfg.get("lr", 1e-3))
        self.max_epochs = int(cfg.get("max_epochs", 45))
        self.patience = int(cfg.get("patience", 8))
        self._trainer: _WindowTrainer | None = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FEDformerForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        prefer = [c for c in Xf.columns if c.startswith(("fourier_", "emd_"))]
        rest = [c for c in Xf.columns if c not in prefer]
        self._cols = prefer + rest
        tr = _WindowTrainer(
            window=self.window,
            hidden=self.d_model,
            lr=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
            net_factory=lambda n: _FEDformerNet(
                n, self.d_model, self.nhead, self.num_layers, self.dropout, self.n_freq
            ),
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
        return {"model": "fedformer", "window": self.window, "d_model": self.d_model}


class GWRFRForecaster:
    """Time-local random forest: sample weights from Gaussian kernel over recency."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.bw_grid = tuple(cfg.get("bandwidth_grid", (21.0, 63.0, 126.0)))
        self.n_estimators_grid = tuple(cfg.get("n_estimators_grid", (100, 200)))
        self.max_depth_grid = tuple(cfg.get("max_depth_grid", (4, 8, None)))
        self._model: RandomForestRegressor | None = None
        self._cols: list[str] = []
        self._bw: float = 63.0
        self._params: dict = {}

    def _temporal_weights(self, n: int, bandwidth: float) -> np.ndarray:
        age = np.arange(n, dtype=float)
        age = (n - 1) - age
        return np.exp(-0.5 * (age / bandwidth) ** 2)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GWRFRForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        self._cols = _ml_feature_cols(Xf)
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        best = (float("inf"), self.bw_grid[0], self.n_estimators_grid[0], self.max_depth_grid[0])
        for bw in self.bw_grid:
            w_full = self._temporal_weights(len(Xf), bw)
            for n_est in self.n_estimators_grid:
                for depth in self.max_depth_grid:
                    model = RandomForestRegressor(
                        n_estimators=int(n_est),
                        max_depth=depth,
                        min_samples_leaf=3,
                        random_state=self.seed,
                        n_jobs=1,
                    )
                    model.fit(
                        Xf.iloc[tr_sl][self._cols].fillna(0.0),
                        yf.iloc[tr_sl],
                        sample_weight=w_full[tr_sl],
                    )
                    pred = model.predict(X_va[self._cols].fillna(0.0))
                    mse = _val_mse(y_va, pred)
                    if mse < best[0]:
                        best = (mse, bw, n_est, depth)

        self._bw = float(best[1])
        self._params = {"n_estimators": int(best[2]), "max_depth": best[3]}
        w = self._temporal_weights(len(Xf), self._bw)
        self._model = RandomForestRegressor(
            n_estimators=self._params["n_estimators"],
            max_depth=self._params["max_depth"],
            min_samples_leaf=3,
            random_state=self.seed,
            n_jobs=1,
        )
        self._model.fit(Xf[self._cols].fillna(0.0), yf, sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        pred = self._model.predict(X[self._cols].fillna(0.0))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "gwrfr", "bandwidth": self._bw, **self._params}


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "sarima_lstm":
        sc = dict(cfg.get("sarima_lstm", {}))
        sc["val_frac"] = vf
        return lambda: SarimaLSTMForecaster(sc, seed, horizon, price_col)
    if name == "fedformer":
        fc = dict(cfg.get("fedformer", {}))
        return lambda: FEDformerForecaster(fc, seed, price_col)
    if name == "gwrfr":
        gc = dict(cfg.get("gwrfr", {}))
        gc["val_frac"] = vf
        return lambda: GWRFRForecaster(gc, seed, price_col)
    raise KeyError(f"unknown tier-II wheat model {name!r}")
