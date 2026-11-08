"""Tier-II gold futures forecasters: TimeGPT, VAR-LSTM, TVP-VAR (TRD-57)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from experiments.sgh.gold_panel import PRICE_COL
from experiments.sgh.tier1_wheat import LSTMLevelForecaster, _split_train_val, _val_mse

ENDOG_COLS = (PRICE_COL, "gold_lr", "gold_vol21")


def _endog_from_X(X: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ENDOG_COLS if c not in X.columns]
    if missing:
        raise KeyError(f"tier-II gold features missing columns: {missing}")
    return X.loc[:, list(ENDOG_COLS)].astype(float)


def _stack_lags(endog: pd.DataFrame, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = endog.to_numpy()
    n, k = vals.shape
    if n <= p:
        raise ValueError("not enough rows for VAR lags")
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for t in range(p, n):
        block = vals[t - p : t].reshape(-1)
        rows.append(block)
        targets.append(vals[t])
    return np.asarray(rows), np.asarray(targets), vals


def _var_forecast_level(endog: pd.DataFrame, p: int, horizon: int) -> float | None:
    if len(endog) < p + 8:
        return None
    try:
        mod = VAR(endog)
        res = mod.fit(maxlags=p, ic=None, trend="c")
        fc = res.forecast(endog.values[-p:], steps=max(1, horizon))
        row = fc[-1] if fc.ndim == 2 else fc
        return float(row[0])
    except Exception:
        return None


def _ridge_var_forecast_level(
    endog: pd.DataFrame,
    p: int,
    horizon: int,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> float | None:
    if len(endog) < p + 8:
        return None
    try:
        Z, Y, vals = _stack_lags(endog, p)
        k = endog.shape[1]
        coefs, intercepts = [], []
        for j in range(k):
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Z, Y[:, j], sample_weight=sample_weight)
            coefs.append(ridge.coef_)
            intercepts.append(float(ridge.intercept_))
        state = vals[-p:].reshape(-1)
        path = vals.copy()
        for _ in range(max(1, horizon)):
            nxt = np.zeros(k)
            for j in range(k):
                nxt[j] = intercepts[j] + float(np.dot(coefs[j], state))
            path = np.vstack([path, nxt])
            state = path[-p:].reshape(-1)
        return float(path[-1, 0])
    except Exception:
        return None


def _grid_search_var(
    endog: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    val_frac: float,
    p_grid: tuple[int, ...],
    forecast_fn,
) -> tuple[int, float]:
    cut = max(12, int(len(endog) * (1.0 - val_frac)))
    if cut >= len(endog) - 5:
        cut = len(endog) // 2
    val_idx = endog.index[cut:]
    best = (float("inf"), p_grid[0])
    for p in p_grid:
        errs: list[float] = []
        for idx in val_idx:
            hist = endog.loc[endog.index <= idx]
            pred = forecast_fn(hist, p, horizon)
            yt = y.loc[idx]
            if pred is not None and np.isfinite(yt):
                errs.append((pred - yt) ** 2)
        if errs:
            mse = float(np.mean(errs))
            if mse < best[0]:
                best = (mse, p)
    return best[1], best[0]


def _nixtla_forecast(series: pd.Series, horizon: int) -> float | None:
    key = os.environ.get("NIXTLA_API_KEY") or os.environ.get("TIMEGPT_TOKEN")
    if not key:
        return None
    try:
        from nixtla import NixtlaClient  # type: ignore[import-untyped]

        client = NixtlaClient(api_key=key)
        df = pd.DataFrame(
            {
                "unique_id": "gold",
                "ds": pd.to_datetime(series.index),
                "y": series.astype(float).values,
            }
        )
        fc = client.forecast(df, h=horizon, freq="B")
        col = "TimeGPT" if "TimeGPT" in fc.columns else fc.columns[-1]
        return float(fc[col].iloc[horizon - 1])
    except Exception:
        return None


def _ets_forecast(series: pd.Series, horizon: int, trend: str | None, seasonal: str | None) -> float | None:
    s = series.dropna().astype(float)
    if len(s) < 30:
        return None
    try:
        model = ExponentialSmoothing(
            s,
            trend=trend,
            seasonal=seasonal,
            initialization_method="estimated",
        )
        res = model.fit(optimized=True, use_brute=False)
        fc = res.forecast(horizon)
        return float(fc.iloc[-1])
    except Exception:
        return None


@dataclass
class VARLevelCore:
    horizon: int
    val_frac: float
    min_hist: int
    p_grid: tuple[int, ...]
    _p: int = field(default=2, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VARLevelCore:
        endog = _endog_from_X(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        self._p, _ = _grid_search_var(endog, y, self.horizon, self.val_frac, self.p_grid, _var_forecast_level)
        self._hist = endog
        return self

    def predict_slice(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=float)
        for j, idx in enumerate(X.index):
            ex = _endog_from_X(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            pred = _var_forecast_level(hist, self._p, self.horizon)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, PRICE_COL])
            out[j] = pred
        return out


class TimeGPTForecaster:
    """Nixtla TimeGPT when API key present; otherwise validation-tuned Holt–Winters (ETS) fallback."""

    def __init__(self, cfg: dict, horizon: int, price_col: str) -> None:
        self.cfg = cfg
        self.h = horizon
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.min_hist = int(cfg.get("min_hist", 80))
        self.prefer_nixtla = bool(cfg.get("prefer_nixtla", True))
        trend_grid = cfg.get("ets_trend_grid", ("add", None))
        seasonal_grid = cfg.get("ets_seasonal_grid", (None,))
        self._trend_grid = tuple(None if t in (None, "null") else t for t in trend_grid)
        self._seasonal_grid = tuple(None if s in (None, "null") else s for s in seasonal_grid)
        self._backend = "ets"
        self._trend: str | None = "add"
        self._seasonal: str | None = None
        self._series: pd.Series | None = None
        self._diag: dict = {}

    def _price_series(self, X: pd.DataFrame) -> pd.Series:
        return X[self.price_col].astype(float)

    def _pick_ets(self, series: pd.Series, y: pd.Series) -> tuple[str | None, str | None]:
        n = len(series)
        cut = max(self.min_hist, int(n * (1.0 - self.val_frac)))
        if cut >= n - 5:
            cut = n // 2
        val_idx = series.index[cut:]
        best = (float("inf"), self._trend_grid[0], self._seasonal_grid[0])
        for trend in self._trend_grid:
            for seasonal in self._seasonal_grid:
                errs: list[float] = []
                for idx in val_idx:
                    hist = series.loc[series.index <= idx].dropna()
                    if len(hist) < self.min_hist:
                        continue
                    pred = _ets_forecast(hist, self.h, trend, seasonal)
                    yt = y.loc[idx]
                    if pred is not None and np.isfinite(yt):
                        errs.append((pred - yt) ** 2)
                if errs:
                    mse = float(np.mean(errs))
                    if mse < best[0]:
                        best = (mse, trend, seasonal)
        return best[1], best[2]

    def _nixtla_val_mse(self, series: pd.Series, y: pd.Series) -> float | None:
        n = len(series)
        cut = max(self.min_hist, int(n * (1.0 - self.val_frac)))
        if cut >= n - 5:
            cut = n // 2
        val_idx = series.index[cut:]
        errs: list[float] = []
        for idx in val_idx:
            hist = series.loc[series.index <= idx].dropna()
            if len(hist) < self.min_hist:
                continue
            pred = _nixtla_forecast(hist, self.h)
            yt = y.loc[idx]
            if pred is not None and np.isfinite(yt):
                errs.append((pred - yt) ** 2)
        if not errs:
            return None
        return float(np.mean(errs))

    def _ets_val_mse(
        self, series: pd.Series, y: pd.Series, trend: str | None, seasonal: str | None
    ) -> float | None:
        n = len(series)
        cut = max(self.min_hist, int(n * (1.0 - self.val_frac)))
        if cut >= n - 5:
            cut = n // 2
        val_idx = series.index[cut:]
        errs: list[float] = []
        for idx in val_idx:
            hist = series.loc[series.index <= idx].dropna()
            if len(hist) < self.min_hist:
                continue
            pred = _ets_forecast(hist, self.h, trend, seasonal)
            yt = y.loc[idx]
            if pred is not None and np.isfinite(yt):
                errs.append((pred - yt) ** 2)
        if not errs:
            return None
        return float(np.mean(errs))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TimeGPTForecaster:
        series = self._price_series(X).reindex(y.index)
        m = y.notna() & series.notna()
        series, y = series.loc[m], y.loc[m]
        self._trend, self._seasonal = self._pick_ets(series, y)
        self._backend = "ets"
        if self.prefer_nixtla:
            mse_n = self._nixtla_val_mse(series, y)
            mse_e = self._ets_val_mse(series, y, self._trend, self._seasonal)
            if mse_n is not None and mse_e is not None and mse_n <= mse_e * 1.05:
                self._backend = "nixtla"
        self._series = series
        self._diag = {
            "backend": self._backend,
            "ets_trend": self._trend,
            "ets_seasonal": self._seasonal,
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._series is not None
        combined = self._series.copy()
        out = np.full(len(X), np.nan, dtype=float)
        for j, idx in enumerate(X.index):
            combined.loc[idx] = float(X[self.price_col].loc[idx])
            hist = combined.loc[:idx].dropna()
            if len(hist) < self.min_hist:
                out[j] = float(X.loc[idx, self.price_col])
                continue
            if self._backend == "nixtla":
                pred = _nixtla_forecast(hist, self.h)
            else:
                pred = _ets_forecast(hist, self.h, self._trend, self._seasonal)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, self.price_col])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"model": "timegpt", **self._diag}


class VARLSTMForecaster:
    """Multivariate VAR h-step level + tabular LSTM → Ridge stack (validation only)."""

    def __init__(self, cfg: dict, seed: int, horizon: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.h = horizon
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        lstm_cfg = dict(cfg.get("lstm", {}))
        lstm_cfg["val_frac"] = self.val_frac
        self._lstm = LSTMLevelForecaster(lstm_cfg, seed, price_col)
        p_grid = tuple(cfg.get("p_grid", (1, 2, 3, 4)))
        self._var = VARLevelCore(
            horizon=horizon,
            val_frac=self.val_frac,
            min_hist=int(cfg.get("min_hist", 60)),
            p_grid=p_grid,
        )
        self._meta: Ridge | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VARLSTMForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        self._var.fit(Xf, yf)
        self._lstm.fit(Xf, yf)

        stack_va = np.column_stack([self._var.predict_slice(X_va), self._lstm.predict(X_va)])
        ok = np.isfinite(stack_va).all(axis=1) & y_va.notna().to_numpy()
        self._meta = Ridge(alpha=float(self.cfg.get("ridge_alpha", 1.0)))
        if ok.sum() < 5:
            self._meta.fit(np.ones((len(y_va), 1)), y_va.to_numpy())
            self._diag = {"meta_fallback": True, "p": self._var._p}
            return self
        self._meta.fit(stack_va[ok], y_va.to_numpy()[ok])
        self._diag = {
            "p": self._var._p,
            "val_mse": _val_mse(y_va, self._meta.predict(stack_va)),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._meta is not None
        stack = np.column_stack([self._var.predict_slice(X), self._lstm.predict(X)])
        pred = self._meta.predict(stack)
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "var_lstm", **self._diag}


@dataclass
class TVPVARForecaster:
    """Time-varying-parameter VAR (exp-weighted ridge on lag stacks; halflife from validation)."""

    horizon: int = 1
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3)
    halflife_grid: tuple[float, ...] = (21.0, 63.0, 126.0)
    alpha: float = 1.0
    _p: int = field(default=2, init=False)
    _halflife: float = field(default=63.0, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def _weights(self, n: int, halflife: float) -> np.ndarray:
        age = np.arange(n, dtype=float)
        age = (n - 1) - age
        return np.exp(-np.log(2) * age / halflife)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TVPVARForecaster:
        endog = _endog_from_X(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        best = (float("inf"), self.p_grid[0], self.halflife_grid[0])
        for halflife in self.halflife_grid:
            for p in self.p_grid:

                def fn(e: pd.DataFrame, lag: int, h: int, hl=halflife) -> float | None:
                    if len(e) < lag + 8:
                        return None
                    try:
                        Z, Y, _ = _stack_lags(e, lag)
                        w = self._weights(len(Z), hl)
                        return _ridge_var_forecast_level(e, lag, h, self.alpha, sample_weight=w)
                    except Exception:
                        return None

                _, mse = _grid_search_var(endog, y, self.horizon, self.val_frac, (p,), fn)
                if np.isfinite(mse) and mse < best[0]:
                    best = (mse, p, halflife)
        if not np.isfinite(best[0]):
            self._p, self._halflife = int(self.p_grid[0]), float(self.halflife_grid[0])
        else:
            self._p, self._halflife = int(best[1]), float(best[2])
        self._hist = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=float)
        for j, idx in enumerate(X.index):
            ex = _endog_from_X(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            if len(hist) < self._p + 8:
                out[j] = float(X.loc[idx, PRICE_COL])
                continue
            try:
                Z, Y, _ = _stack_lags(hist, self._p)
                w = self._weights(len(Z), self._halflife)
                pred = _ridge_var_forecast_level(hist, self._p, self.horizon, self.alpha, sample_weight=w)
            except Exception:
                pred = None
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, PRICE_COL])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {
            "model": "tvp_var",
            "p": self._p,
            "halflife": self._halflife,
            "alpha": self.alpha,
            "horizon": self.horizon,
        }


def model_factory(name: str, cfg: dict, *, horizon: int, price_col: str) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "timegpt":
        tc = dict(cfg.get("timegpt", {}))
        tc["val_frac"] = vf
        return lambda: TimeGPTForecaster(tc, horizon, price_col)
    if name == "var_lstm":
        vc = dict(cfg.get("var_lstm", {}))
        vc["val_frac"] = vf
        return lambda: VARLSTMForecaster(vc, seed, horizon, price_col)
    if name == "tvp_var":
        tv = dict(cfg.get("tvp_var", {}))
        return lambda: TVPVARForecaster(
            horizon=horizon,
            val_frac=vf,
            p_grid=tuple(tv.get("p_grid", (1, 2, 3))),
            halflife_grid=tuple(float(x) for x in tv.get("halflife_grid", (21, 63, 126))),
            alpha=float(tv.get("alpha", 1.0)),
        )
    raise KeyError(f"unknown tier-II gold model {name!r}")
