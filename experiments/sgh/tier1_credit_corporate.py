"""Tier-I corporate-credit forecasters: ARDL, dynamic regression, bank lending channel (TRD-40)."""
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


EXOG_BASE = (
    "nbp_rate_l1",
    "nbp_drate_l1",
    "cpi_yoy",
    "cpi_l1",
    "consumer_l1",
    "mortgage_l1",
    "standards_l1",
)


def _ardl_block(X: pd.DataFrame, p: int) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for lag in range(1, p + 1):
        parts.append(X["corporate_now"].shift(lag).rename(f"ardl_corporate_l{lag}"))
    for col in EXOG_BASE:
        if col not in X.columns:
            continue
        parts.append(X[col].rename(f"ardl_{col}"))
        for lag in range(2, min(p, 3) + 1):
            parts.append(X[col].shift(lag - 1).rename(f"ardl_{col}_l{lag}"))
    return pd.concat(parts, axis=1)


def _blc_block(X: pd.DataFrame, n_drate_lags: int) -> pd.DataFrame:
    """Reduced-form bank lending channel: policy rate changes, lagged credit growth, optional standards."""
    parts: dict[str, pd.Series] = {
        "blc_corporate_dlog_l1": X["corporate_dlog_l1"],
        "blc_corporate_dlog_l2": X["corporate_dlog_l2"],
        "blc_nbp_rate_l1": X["nbp_rate_l1"],
        "blc_cpi_l1": X["cpi_l1"],
    }
    for lag in range(0, n_drate_lags):
        col = "nbp_drate" if lag == 0 else f"nbp_drate_l{lag}"
        if col in X.columns:
            parts[f"blc_{col}"] = X[col]
    if "standards_l1" in X.columns:
        parts["blc_standards_l1"] = X["standards_l1"]
    if "standards_l2" in X.columns:
        parts["blc_standards_l2"] = X["standards_l2"]
    return pd.DataFrame(parts, index=X.index)


@dataclass
class ARDLForecaster:
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3, 4)
    _p: int = field(default=2, init=False)
    _params: pd.Series | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ARDLForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        best = (float("inf"), self.p_grid[0], None, [])
        for p in self.p_grid:
            Df = _ardl_block(Xf, p)
            Dv = _ardl_block(Xv, p)
            m = yf.notna() & Df.notna().all(axis=1)
            mv = yv.notna() & Dv.notna().all(axis=1)
            if int(m.sum()) < 10 or int(mv.sum()) < 3:
                continue
            fit = sm.OLS(yf[m], sm.add_constant(Df[m])).fit()
            pred = _ols_predict(Dv[mv], fit.params)
            mse = _level_mse(yv[mv], pred)
            if mse < best[0]:
                best = (mse, p, fit.params, list(Df.columns))
        self._p = best[1]
        self._params = best[2]
        self._cols = best[3]
        if self._params is None:
            self._p = 2
            D = _ardl_block(X, self._p)
            m = y.notna() & D.notna().all(axis=1)
            fit = sm.OLS(y[m], sm.add_constant(D[m])).fit()
            self._params = fit.params
            self._cols = list(D.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        D = _ardl_block(X, self._p)
        m = D.notna().all(axis=1)
        out = np.full(len(X), np.nan)
        if m.any():
            out[m] = _ols_predict(D[m], self._params)
        fallback = X["corporate_now"].ffill()
        out = np.where(np.isfinite(out), out, fallback.to_numpy())
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "n_regressors": len(self._cols)}


@dataclass
class DynamicRegressionForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DynamicRegressionForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        cols = [c for c in X.columns if c != "corporate_now"]
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


@dataclass
class BankLendingChannelForecaster:
    """Level forecast via reduced-form BLC (Δrate, lagged Δlog credit, standards) — OLS on forward level."""

    val_frac: float = 0.2
    drate_lags_grid: tuple[int, ...] = (1, 2, 3)
    _n_drate_lags: int = field(default=2, init=False)
    _params: pd.Series | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BankLendingChannelForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        best = (float("inf"), self.drate_lags_grid[0], None, [])
        for n_lags in self.drate_lags_grid:
            Df = _blc_block(Xf, n_lags)
            Dv = _blc_block(Xv, n_lags)
            m = yf.notna() & Df.notna().all(axis=1)
            mv = yv.notna() & Dv.notna().all(axis=1)
            if int(m.sum()) < 10 or int(mv.sum()) < 3:
                continue
            fit = sm.OLS(yf[m], sm.add_constant(Df[m])).fit()
            pred = _ols_predict(Dv[mv], fit.params)
            mse = _level_mse(yv[mv], pred)
            if mse < best[0]:
                best = (mse, n_lags, fit.params, list(Df.columns))
        self._n_drate_lags = best[1]
        self._params = best[2]
        self._cols = best[3]
        if self._params is None:
            self._n_drate_lags = 2
            D = _blc_block(X, self._n_drate_lags)
            m = y.notna() & D.notna().all(axis=1)
            fit = sm.OLS(y[m], sm.add_constant(D[m])).fit()
            self._params = fit.params
            self._cols = list(D.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        D = _blc_block(X, self._n_drate_lags)
        m = D.notna().all(axis=1)
        out = np.full(len(X), np.nan)
        if m.any():
            out[m] = _ols_predict(D[m], self._params)
        fallback = X["corporate_now"].ffill()
        out = np.where(np.isfinite(out), out, fallback.to_numpy())
        return out

    def diagnostics(self) -> dict:
        return {"n_drate_lags": self._n_drate_lags, "n_regressors": len(self._cols)}


def model_factory(name: str, cfg: dict):
    val_frac = float(cfg.get("val_frac", 0.2))
    if name == "ardl":
        return lambda: ARDLForecaster(
            val_frac=val_frac,
            p_grid=tuple(cfg.get("ardl", {}).get("p", (1, 2, 3, 4))),
        )
    if name == "dynamic_regression":
        return lambda: DynamicRegressionForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "bank_lending_channel":
        return lambda: BankLendingChannelForecaster(
            val_frac=val_frac,
            drate_lags_grid=tuple(cfg.get("blc", {}).get("drate_lags", (1, 2, 3))),
        )
    raise KeyError(f"unknown tier-I corporate credit model {name!r}")
