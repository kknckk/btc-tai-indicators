"""Tier-I TFI net-assets forecasters: flow-performance, panel OLS, dynamic regression (TRD-44)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge


def _split_train_val(X: pd.DataFrame, y: pd.Series, val_frac: float) -> tuple[tuple, tuple]:
    n = len(X)
    cut = max(12, int(n * (1.0 - val_frac)))
    if cut >= n - 5:
        cut = n // 2
    return (X.iloc[:cut], y.iloc[:cut]), (X.iloc[cut:], y.iloc[cut:])


def _level_mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    m = y_true.notna().to_numpy() & np.isfinite(y_pred)
    if not m.any():
        return float("inf")
    return float(np.mean((y_true.to_numpy()[m] - y_pred[m]) ** 2))


def _ols_predict(design: pd.DataFrame, params: pd.Series) -> np.ndarray:
    X = sm.add_constant(design, has_constant="add")
    cols = [c for c in params.index if c in X.columns]
    return (X[cols] @ params[cols]).to_numpy()


FLOW_EXOG = ("tfi_l1", "perf_l1", "perf_l3", "flow_l1", "flow_l2", "nbp_rate_l1", "cpi_l1")
PANEL_EXOG = (
    "tfi_l1",
    "tfi_l12",
    "nbp_rate_l1",
    "cpi_yoy",
    "cpi_l1",
    "deposits_l1",
    "deposits_mom_l1",
    "consumer_l1",
    "bond_10y_l1",
)


def _cols_present(X: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    return [c for c in names if c in X.columns]


@dataclass
class FlowPerformanceForecaster:
    """Level forecast from lagged performance and implied net-flow proxies (Sirri–Tufano style)."""

    val_frac: float = 0.2
    _params: pd.Series | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FlowPerformanceForecaster:
        cols = _cols_present(X, FLOW_EXOG)
        if "tfi_l1" not in cols:
            cols = ["tfi_l1"] + [c for c in cols if c != "tfi_l1"]
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        Df, Dv = Xf[cols], Xv[cols]
        m = yf.notna() & Df.notna().all(axis=1)
        mv = yv.notna() & Dv.notna().all(axis=1)
        if int(m.sum()) < 8:
            m = y.notna() & X[cols].notna().all(axis=1)
            fit = sm.OLS(y[m], sm.add_constant(X.loc[m, cols])).fit()
        else:
            fit = sm.OLS(yf[m], sm.add_constant(Df[m])).fit()
            pred = _ols_predict(Dv[mv], fit.params)
            _ = _level_mse(yv[mv], pred)
        self._params = fit.params
        self._cols = cols
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        D = X[self._cols]
        m = D.notna().all(axis=1)
        out = np.full(len(X), np.nan)
        if m.any():
            out[m] = _ols_predict(D[m], self._params)
        fallback = X["tfi_now"].ffill()
        return np.where(np.isfinite(out), out, fallback.to_numpy())

    def diagnostics(self) -> dict:
        return {"n_regressors": len(self._cols), "flow_coef": _coef(self._params, "flow_l1")}


def _coef(params: pd.Series | None, name: str) -> float | None:
    if params is None or name not in params.index:
        return None
    return float(params[name])


@dataclass
class PanelRegressionForecaster:
    """Pooled macro-panel regressors + TFI persistence (OLS, exog subset via validation MSE)."""

    val_frac: float = 0.2
    ar_lags: tuple[int, ...] = (1, 2, 3, 12)
    _params: pd.Series | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _ar: tuple[int, ...] = field(default=(1, 12), init=False)

    def _design(self, X: pd.DataFrame, ar: tuple[int, ...]) -> pd.DataFrame:
        parts: list[pd.Series] = []
        for lag in ar:
            col = f"tfi_l{lag}"
            if col in X.columns:
                parts.append(X[col])
        for col in _cols_present(X, PANEL_EXOG):
            if col.startswith("tfi_"):
                continue
            parts.append(X[col])
        return pd.concat(parts, axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PanelRegressionForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        best = (float("inf"), self.ar_lags, None, [])
        for ar in ((1,), (1, 12), (1, 2, 3), self.ar_lags):
            ar = tuple(l for l in ar if f"tfi_l{l}" in X.columns) or (1,)
            Df = self._design(Xf, ar)
            Dv = self._design(Xv, ar)
            if Df.empty:
                continue
            m = yf.notna() & Df.notna().all(axis=1)
            mv = yv.notna() & Dv.notna().all(axis=1)
            if int(m.sum()) < 10 or int(mv.sum()) < 3:
                continue
            fit = sm.OLS(yf[m], sm.add_constant(Df[m])).fit()
            pred = _ols_predict(Dv[mv], fit.params)
            mse = _level_mse(yv[mv], pred)
            if mse < best[0]:
                best = (mse, ar, fit.params, list(Df.columns))
        self._params = best[2]
        self._cols = best[3]
        self._ar = best[1]
        if self._params is None:
            ar = (1, 12) if "tfi_l12" in X.columns else (1,)
            D = self._design(X, ar)
            m = y.notna() & D.notna().all(axis=1)
            fit = sm.OLS(y[m], sm.add_constant(D[m])).fit()
            self._params = fit.params
            self._cols = list(D.columns)
            self._ar = ar
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        D = self._design(X, self._ar)
        m = D.notna().all(axis=1)
        out = np.full(len(X), np.nan)
        if m.any():
            out[m] = _ols_predict(D[m], self._params)
        fallback = X["tfi_now"].ffill()
        return np.where(np.isfinite(out), out, fallback.to_numpy())

    def diagnostics(self) -> dict:
        return {"ar_lags": list(self._ar), "n_regressors": len(self._cols)}


@dataclass
class DynamicRegressionForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DynamicRegressionForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        cols = [c for c in X.columns if c != "tfi_now"]
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            m = Ridge(alpha=alpha)
            m.fit(Xf[cols].fillna(0.0), yf)
            pred = m.predict(Xv[cols].fillna(0.0))
            mse = _level_mse(yv, pred)
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self._alpha = best_alpha
        self._cols = cols
        self._model = Ridge(alpha=best_alpha)
        self._model.fit(X[cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X[self._cols].fillna(0.0))

    def diagnostics(self) -> dict:
        return {"alpha": self._alpha, "n_features": len(self._cols)}


def model_factory(name: str, cfg: dict):
    val_frac = float(cfg.get("val_frac", 0.2))
    if name == "flow_performance":
        return lambda: FlowPerformanceForecaster(val_frac=val_frac)
    if name == "panel_regression":
        return lambda: PanelRegressionForecaster(
            val_frac=val_frac,
            ar_lags=tuple(cfg.get("panel", {}).get("ar_lags", (1, 2, 3, 12))),
        )
    if name == "dynamic_regression":
        return lambda: DynamicRegressionForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    raise KeyError(f"unknown tier-I tfi_assets model {name!r}")
