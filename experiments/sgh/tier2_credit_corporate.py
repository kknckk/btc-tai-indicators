"""Tier-II corporate-credit forecasters: VAR/BVAR, standards survey, FCI, bank–firm, LASSO, ML (TRD-41)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from statsmodels.tsa.api import VAR

from experiments.sgh.tier1_bonds import _level_mse

ENDOG_COLS = ("credit_corporate", "nbp_rate", "gus_cpi_yoy", "credit_consumer")


def _split_train_val_xy(X: pd.DataFrame, y: pd.Series, val_frac: float) -> tuple[tuple, tuple]:
    n = len(X)
    cut = max(12, int(n * (1.0 - val_frac)))
    if cut >= n - 5:
        cut = n // 2
    return (X.iloc[:cut], y.iloc[:cut]), (X.iloc[cut:], y.iloc[cut:])


def _endog_from_X(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ENDOG_COLS if c in X.columns]
    if len(cols) < 3:
        cols = [c for c in ("credit_corporate", "nbp_rate", "gus_cpi_yoy") if c in X.columns]
    missing = [c for c in ("credit_corporate", "nbp_rate", "gus_cpi_yoy") if c not in cols]
    if missing:
        raise KeyError(f"tier-II corporate features missing columns: {missing}")
    return X.loc[:, cols].astype(float)


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


def _var_forecast(endog: pd.DataFrame, p: int, horizon: int) -> float | None:
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


def _ridge_var_forecast(endog: pd.DataFrame, p: int, horizon: int, alpha: float) -> float | None:
    if len(endog) < p + 8:
        return None
    try:
        Z, Y, vals = _stack_lags(endog, p)
        k = endog.shape[1]
        coefs = []
        intercepts = []
        for j in range(k):
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Z, Y[:, j])
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


def _ml_feature_cols(X: pd.DataFrame) -> list[str]:
    skip = {
        "credit_corporate",
        "credit_consumer",
        "credit_mortgage",
        "nbp_rate",
        "gus_cpi_yoy",
    }
    return [c for c in X.columns if c not in skip and X[c].dtype != object]


def _standards_cols(X: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in (
            "corporate_now",
            "corporate_l1",
            "standards_l1",
            "standards_l2",
            "nbp_rate_l1",
            "nbp_drate_l1",
            "cpi_yoy",
            "cpi_l1",
        )
        if c in X.columns
    ]
    return cols or [c for c in ("corporate_now", "corporate_l1", "nbp_rate_l1", "cpi_yoy") if c in X.columns]


def _bank_firm_cols(X: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in (
            "corporate_now",
            "corporate_l1",
            "corporate_dlog_l1",
            "corporate_dlog_l2",
            "nbp_rate_l1",
            "nbp_drate",
            "nbp_drate_l1",
            "nbp_drate_l2",
            "standards_l1",
            "standards_l2",
            "consumer_l1",
            "consumer_mom_l1",
            "mortgage_l1",
            "cpi_yoy",
            "fci_l1",
        )
        if c in X.columns
    ]
    return cols or [c for c in ("corporate_now", "corporate_l1", "nbp_rate_l1", "cpi_yoy") if c in X.columns]


@dataclass
class VARForecaster:
    horizon: int = 1
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3, 4)
    _p: int = field(default=2, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VARForecaster:
        endog = _endog_from_X(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        self._p, _ = _grid_search_var(endog, y, self.horizon, self.val_frac, self.p_grid, _var_forecast)
        self._hist = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        for j, idx in enumerate(X.index):
            ex = _endog_from_X(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            pred = _var_forecast(hist, self._p, self.horizon)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, "corporate_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "horizon": self.horizon, "model": "var"}


@dataclass
class BVARForecaster:
    horizon: int = 1
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3)
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    _p: int = field(default=2, init=False)
    _alpha: float = field(default=1.0, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BVARForecaster:
        endog = _endog_from_X(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        best = (float("inf"), self.p_grid[0], self.alpha_grid[0])
        for alpha in self.alpha_grid:
            p, mse = _grid_search_var(
                endog,
                y,
                self.horizon,
                self.val_frac,
                self.p_grid,
                lambda e, p, h, a=alpha: _ridge_var_forecast(e, p, h, a),
            )
            if mse < best[0]:
                best = (mse, p, alpha)
        self._p, self._alpha = best[1], best[2]
        self._hist = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        for j, idx in enumerate(X.index):
            ex = _endog_from_X(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            pred = _ridge_var_forecast(hist, self._p, self.horizon, self._alpha)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, "corporate_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "alpha": self._alpha, "horizon": self.horizon, "model": "bvar_ridge"}


@dataclass
class StandardsSurveyForecaster:
    """NBP credit standards survey (firms) + macro lags — Ridge on forward corporate stock."""

    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StandardsSurveyForecaster:
        cols = _standards_cols(X)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
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
        return {"alpha": self._alpha, "n_features": len(self._cols), "model": "standards_survey_ridge"}


@dataclass
class FCIForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FCIForecaster:
        cols = [
            c
            for c in (
                "corporate_now",
                "corporate_l1",
                "fci",
                "fci_l1",
                "fci_l3",
                "nbp_rate_l1",
                "cpi_yoy",
                "corporate_share_l1",
            )
            if c in X.columns
        ]
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
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
        return {"alpha": self._alpha, "n_features": len(self._cols), "model": "fci_ridge"}


@dataclass
class BankFirmForecaster:
    """Bank–firm linkage: policy rate changes, standards, cross-sector credit (Ridge)."""

    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BankFirmForecaster:
        cols = _bank_firm_cols(X)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
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
        return {"alpha": self._alpha, "n_features": len(self._cols), "model": "bank_firm_ridge"}


@dataclass
class LassoForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
    _model: Lasso | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1e-2, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LassoForecaster:
        cols = _ml_feature_cols(X)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            m = Lasso(alpha=alpha, max_iter=5000)
            m.fit(Xf[cols].fillna(0.0), yf)
            pred = m.predict(Xv[cols].fillna(0.0))
            mse = _level_mse(yv, pred)
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self._alpha = best_alpha
        self._cols = cols
        self._model = Lasso(alpha=best_alpha, max_iter=5000)
        self._model.fit(X[cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X[self._cols].fillna(0.0))

    def diagnostics(self) -> dict:
        nnz = int(np.sum(np.abs(self._model.coef_) > 1e-8)) if self._model is not None else 0
        return {"alpha": self._alpha, "n_nonzero": nnz, "model": "lasso"}


@dataclass
class RandomForestForecaster:
    val_frac: float = 0.2
    n_estimators_grid: tuple[int, ...] = (100, 200)
    max_depth_grid: tuple[int | None, ...] = (4, 8, None)
    _model: RandomForestRegressor | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _params: dict = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestForecaster:
        cols = _ml_feature_cols(X)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best = (float("inf"), {})
        for n_est in self.n_estimators_grid:
            for depth in self.max_depth_grid:
                m = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=3,
                    random_state=0,
                    n_jobs=1,
                )
                m.fit(Xf[cols].fillna(0.0), yf)
                pred = m.predict(Xv[cols].fillna(0.0))
                mse = _level_mse(yv, pred)
                if mse < best[0]:
                    best = (mse, {"n_estimators": n_est, "max_depth": depth})
        self._params = best[1]
        self._cols = cols
        self._model = RandomForestRegressor(
            n_estimators=self._params["n_estimators"],
            max_depth=self._params["max_depth"],
            min_samples_leaf=3,
            random_state=0,
            n_jobs=1,
        )
        self._model.fit(X[cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X[self._cols].fillna(0.0))

    def diagnostics(self) -> dict:
        return {**self._params, "n_features": len(self._cols), "model": "random_forest"}


@dataclass
class GradientBoostingForecaster:
    val_frac: float = 0.2
    max_depth_grid: tuple[int, ...] = (3, 5)
    learning_rate_grid: tuple[float, ...] = (0.05, 0.1)
    _model: HistGradientBoostingRegressor | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _params: dict = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingForecaster:
        cols = _ml_feature_cols(X)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best = (float("inf"), {})
        for depth in self.max_depth_grid:
            for lr in self.learning_rate_grid:
                m = HistGradientBoostingRegressor(
                    max_depth=depth,
                    learning_rate=lr,
                    max_iter=200,
                    random_state=0,
                )
                m.fit(Xf[cols].fillna(0.0), yf)
                pred = m.predict(Xv[cols].fillna(0.0))
                mse = _level_mse(yv, pred)
                if mse < best[0]:
                    best = (mse, {"max_depth": depth, "learning_rate": lr})
        self._params = best[1]
        self._cols = cols
        self._model = HistGradientBoostingRegressor(
            max_depth=self._params["max_depth"],
            learning_rate=self._params["learning_rate"],
            max_iter=200,
            random_state=0,
        )
        self._model.fit(X[cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X[self._cols].fillna(0.0))

    def diagnostics(self) -> dict:
        return {**self._params, "n_features": len(self._cols), "model": "gradient_boosting"}


def model_factory(name: str, cfg: dict, *, horizon: int = 1):
    val_frac = float(cfg.get("val_frac", 0.2))
    var_cfg = cfg.get("var", {})
    p_grid = tuple(var_cfg.get("p", (1, 2, 3, 4)))
    if name == "var":
        return lambda: VARForecaster(horizon=horizon, val_frac=val_frac, p_grid=p_grid)
    if name == "bvar":
        return lambda: BVARForecaster(
            horizon=horizon,
            val_frac=val_frac,
            p_grid=tuple(var_cfg.get("p_bvar", (1, 2, 3))),
            alpha_grid=tuple(var_cfg.get("alpha", (0.1, 1.0, 10.0, 100.0))),
        )
    if name == "standards_survey":
        return lambda: StandardsSurveyForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "fci":
        return lambda: FCIForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "bank_firm":
        return lambda: BankFirmForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "lasso":
        return lambda: LassoForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("lasso", {}).get("alpha", (1e-4, 1e-3, 1e-2, 0.1, 1.0))),
        )
    if name == "random_forest":
        rf = cfg.get("random_forest", {})
        return lambda: RandomForestForecaster(
            val_frac=val_frac,
            n_estimators_grid=tuple(rf.get("n_estimators", (100, 200))),
            max_depth_grid=tuple(rf.get("max_depth", (4, 8, None))),
        )
    if name == "gradient_boosting":
        gb = cfg.get("gradient_boosting", {})
        return lambda: GradientBoostingForecaster(
            val_frac=val_frac,
            max_depth_grid=tuple(gb.get("max_depth", (3, 5))),
            learning_rate_grid=tuple(gb.get("learning_rate", (0.05, 0.1))),
        )
    raise KeyError(f"unknown tier-II corporate credit model {name!r}")
