"""Tier-I household-deposits forecasters: ARDL, ECM, dynamic regression (TRD-42)."""
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


EXOG_BASE = ("nbp_rate_l1", "cpi_yoy", "cpi_l1", "consumer_l1", "mortgage_l1")


def _ardl_block(X: pd.DataFrame, p: int) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for lag in range(1, p + 1):
        parts.append(X["deposits_now"].shift(lag).rename(f"ardl_deposits_l{lag}"))
    for col in EXOG_BASE:
        if col not in X.columns:
            continue
        parts.append(X[col].rename(f"ardl_{col}"))
        for lag in range(2, min(p, 3) + 1):
            parts.append(X[col].shift(lag - 1).rename(f"ardl_{col}_l{lag}"))
    return pd.concat(parts, axis=1)


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
        fallback = X["deposits_now"].ffill()
        out = np.where(np.isfinite(out), out, fallback.to_numpy())
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "n_regressors": len(self._cols)}


@dataclass
class ECMForecaster:
    val_frac: float = 0.2
    _lr_cols: list[str] = field(default_factory=list, init=False)
    _lr_params: pd.Series | None = field(default=None, init=False)
    _sr_params: pd.Series | None = field(default=None, init=False)
    _sr_cols: list[str] = field(default_factory=list, init=False)

    def _long_run_cols(self, X: pd.DataFrame) -> list[str]:
        cols = [c for c in ("nbp_rate_l1", "cpi_yoy", "consumer_l1", "mortgage_l1") if c in X.columns]
        return cols or ["nbp_rate_l1", "cpi_yoy"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ECMForecaster:
        self._lr_cols = self._long_run_cols(X)
        lr_x = X[self._lr_cols]
        m_lr = X["deposits_now"].notna() & lr_x.notna().all(axis=1)
        lr_fit = sm.OLS(X.loc[m_lr, "deposits_now"], sm.add_constant(lr_x[m_lr])).fit()
        self._lr_params = lr_fit.params
        ect = X["deposits_now"] - _ols_predict(lr_x, lr_fit.params)
        d_dep = X["deposits_now"] - X["deposits_l1"]
        sr_parts: dict[str, pd.Series] = {
            "ect_l1": ect.shift(1),
            "d_deposits_l1": d_dep.shift(1),
        }
        if "nbp_rate_l1" in X.columns:
            sr_parts["nbp_rate_l1"] = X["nbp_rate_l1"]
        if "cpi_l1" in X.columns:
            sr_parts["cpi_l1"] = X["cpi_l1"]
        sr = pd.DataFrame(sr_parts)
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        idx_f = Xf.index
        idx_v = Xv.index
        Df, Dv = sr.loc[idx_f], sr.loc[idx_v]
        mf = yf.notna() & Df.notna().all(axis=1)
        mv = yv.notna() & Dv.notna().all(axis=1)
        if int(mf.sum()) < 8:
            mf = y.notna() & sr.notna().all(axis=1)
            fit = sm.OLS(y[mf], sm.add_constant(sr[mf])).fit()
        else:
            fit = sm.OLS(yf[mf], sm.add_constant(Df[mf])).fit()
        self._sr_params = fit.params
        self._sr_cols = list(sr.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._lr_params is not None and self._sr_params is not None
        lr_x = X[self._lr_cols]
        ect = X["deposits_now"] - _ols_predict(lr_x, self._lr_params)
        d_dep = X["deposits_now"] - X["deposits_l1"]
        sr_parts: dict[str, pd.Series] = {
            "ect_l1": ect.shift(1),
            "d_deposits_l1": d_dep.shift(1),
        }
        if "nbp_rate_l1" in X.columns:
            sr_parts["nbp_rate_l1"] = X["nbp_rate_l1"]
        if "cpi_l1" in X.columns:
            sr_parts["cpi_l1"] = X["cpi_l1"]
        sr = pd.DataFrame(sr_parts)
        m = sr.notna().all(axis=1)
        out = np.full(len(X), np.nan)
        if m.any():
            out[m] = _ols_predict(sr[m], self._sr_params)
        fallback = X["deposits_now"].ffill()
        return np.where(np.isfinite(out), out, fallback.to_numpy())

    def diagnostics(self) -> dict:
        ect_coef = None
        if self._sr_params is not None and "ect_l1" in self._sr_params.index:
            ect_coef = float(self._sr_params["ect_l1"])
        return {"long_run": self._lr_cols, "ect_coef": ect_coef}


@dataclass
class DynamicRegressionForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DynamicRegressionForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        cols = [c for c in X.columns if c != "deposits_now"]
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
    if name == "ardl":
        return lambda: ARDLForecaster(
            val_frac=val_frac,
            p_grid=tuple(cfg.get("ardl", {}).get("p", (1, 2, 3, 4))),
        )
    if name == "ecm":
        return lambda: ECMForecaster(val_frac=val_frac)
    if name == "dynamic_regression":
        return lambda: DynamicRegressionForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    raise KeyError(f"unknown tier-I deposits model {name!r}")
