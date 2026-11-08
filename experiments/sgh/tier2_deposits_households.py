"""Tier-II household-deposits forecasters (TRD-43): VAR/BVAR, portfolio, segmentation, PKO, ML, retention."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.api import VAR

from experiments.sgh.tier1_bonds import _FactorAR, _level_mse, _split_train_val
from experiments.sgh.tier1_deposits_households import ECMForecaster

ENDOG_COLS = ("deposits_households", "nbp_rate", "gus_cpi_yoy", "credit_consumer")
PORTFOLIO_COLS = ("deposits_households", "nbp_rate", "gus_cpi_yoy", "credit_consumer", "credit_mortgage")


def _split_train_val_xy(X: pd.DataFrame, y: pd.Series, val_frac: float) -> tuple[tuple, tuple]:
    n = len(X)
    cut = max(12, int(n * (1.0 - val_frac)))
    if cut >= n - 5:
        cut = n // 2
    return (X.iloc[:cut], y.iloc[:cut]), (X.iloc[cut:], y.iloc[cut:])


def _endog_from_X(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ENDOG_COLS if c in X.columns]
    if len(cols) < 3:
        cols = [c for c in ("deposits_households", "nbp_rate", "gus_cpi_yoy") if c in X.columns]
    missing = [c for c in ("deposits_households", "nbp_rate", "gus_cpi_yoy") if c not in cols]
    if missing:
        raise KeyError(f"tier-II deposits features missing columns: {missing}")
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
        "deposits_households",
        "credit_consumer",
        "credit_mortgage",
        "nbp_rate",
        "gus_cpi_yoy",
        "deposits_now",
    }
    return [c for c in X.columns if c not in skip and X[c].dtype != object]


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
                pred = float(X.loc[idx, "deposits_now"])
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
                pred = float(X.loc[idx, "deposits_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "alpha": self._alpha, "horizon": self.horizon, "model": "bvar_ridge"}


@dataclass
class PortfolioModelForecaster:
    """Open proxy for bank portfolio-balance models (PCA on deposit/credit/rate system)."""

    horizon: int = 1
    val_frac: float = 0.2
    n_components: int = 2
    _pca: PCA | None = field(default=None, init=False)
    _ar: _FactorAR = field(default_factory=lambda: _FactorAR(np.zeros(2), np.ones(2)), init=False)
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(5), init=False)
    _std: np.ndarray = field(default_factory=lambda: np.ones(5), init=False)
    _beta: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _intercept: float = field(default=0.0, init=False)

    def _portfolio_matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in PORTFOLIO_COLS if c in X.columns]
        return X[cols].astype(float).to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PortfolioModelForecaster:
        M = self._portfolio_matrix(X)
        n = len(M)
        tr, va = _split_train_val(n, self.val_frac)
        self._mean = np.nanmean(M[tr], axis=0)
        self._std = np.nanstd(M[tr], axis=0)
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        Z = (M - self._mean) / self._std
        self._pca = PCA(n_components=min(self.n_components, Z.shape[1]))
        scores = self._pca.fit_transform(Z)
        self._ar = _FactorAR.fit(scores[tr])
        design = np.column_stack([np.ones(len(scores)), scores])
        coef, _, _, _ = np.linalg.lstsq(design[tr], y.to_numpy()[tr], rcond=None)
        self._intercept = float(coef[0])
        self._beta = coef[1:]
        _ = _level_mse(y.iloc[va], self._predict_rows(X.iloc[va]))
        return self

    def _predict_rows(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pca is not None
        M = self._portfolio_matrix(X)
        Z = (M - self._mean) / self._std
        scores = self._pca.transform(Z)
        out = np.zeros(len(X))
        for i in range(len(X)):
            fac_h = self._ar.steps(scores[i], self.horizon)
            out[i] = self._intercept + float(np.dot(self._beta, fac_h))
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self._predict_rows(X)
        fallback = X["deposits_now"].ffill().to_numpy()
        return np.where(np.isfinite(pred), pred, fallback)

    def diagnostics(self) -> dict:
        return {"model": "portfolio_pca_ar", "n_components": self.n_components, "horizon": self.horizon}


@dataclass
class SegmentationModelForecaster:
    """Regime-specific Ridge models (high vs low NBP rate — open segmentation proxy)."""

    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    _alpha: float = field(default=1.0, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _high: Ridge | None = field(default=None, init=False)
    _low: Ridge | None = field(default=None, init=False)
    _rate_threshold: float = field(default=3.0, init=False)

    def _seg_cols(self, X: pd.DataFrame) -> list[str]:
        return [c for c in ("deposits_l1", "deposits_l3", "nbp_rate_l1", "cpi_yoy", "consumer_l1") if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SegmentationModelForecaster:
        cols = self._seg_cols(X)
        self._cols = cols
        rate = X["nbp_rate_l1"] if "nbp_rate_l1" in X.columns else X["nbp_rate"]
        self._rate_threshold = float(rate.median())
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            high_m = Ridge(alpha=alpha)
            low_m = Ridge(alpha=alpha)
            hi = rate.loc[Xf.index] >= self._rate_threshold
            lo = ~hi
            if int(hi.sum()) < 5 or int(lo.sum()) < 5:
                continue
            high_m.fit(Xf.loc[hi, cols].fillna(0.0), yf.loc[hi])
            low_m.fit(Xf.loc[lo, cols].fillna(0.0), yf.loc[lo])
            pred = np.zeros(len(Xv))
            hi_v = rate.loc[Xv.index] >= self._rate_threshold
            pred[hi_v.to_numpy()] = high_m.predict(Xv.loc[hi_v, cols].fillna(0.0))
            pred[~hi_v.to_numpy()] = low_m.predict(Xv.loc[~hi_v, cols].fillna(0.0))
            mse = _level_mse(yv, pred)
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self._alpha = best_alpha
        self._high = Ridge(alpha=best_alpha)
        self._low = Ridge(alpha=best_alpha)
        hi = rate >= self._rate_threshold
        lo = ~hi
        self._high.fit(X.loc[hi, cols].fillna(0.0), y.loc[hi])
        self._low.fit(X.loc[lo, cols].fillna(0.0), y.loc[lo])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._high is not None and self._low is not None
        rate = X["nbp_rate_l1"] if "nbp_rate_l1" in X.columns else X["nbp_rate"]
        out = np.zeros(len(X))
        hi = rate >= self._rate_threshold
        out[hi.to_numpy()] = self._high.predict(X.loc[hi, self._cols].fillna(0.0))
        out[~hi.to_numpy()] = self._low.predict(X.loc[~hi, self._cols].fillna(0.0))
        return out

    def diagnostics(self) -> dict:
        return {
            "model": "segmentation_rate_regime",
            "rate_threshold": self._rate_threshold,
            "alpha": self._alpha,
        }


@dataclass
class RetentionModelForecaster:
    """Persistence / retention proxy: heavy lag structure + liquidity index."""

    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def _retention_cols(self, X: pd.DataFrame) -> list[str]:
        return [
            c
            for c in (
                "deposits_l1",
                "deposits_l12",
                "deposits_mom_l1",
                "nbp_rate_l1",
                "liquidity_index_l1",
                "credit_deposit_ratio_l1",
            )
            if c in X.columns
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RetentionModelForecaster:
        cols = self._retention_cols(X)
        self._cols = cols
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
        self._model = Ridge(alpha=best_alpha)
        self._model.fit(X[cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X[self._cols].fillna(0.0))

    def diagnostics(self) -> dict:
        return {"model": "retention_ridge", "alpha": self._alpha, "n_features": len(self._cols)}


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


@dataclass
class PKOInternalForecaster:
    """Open proxy when proprietary PKO models are unavailable: ECM + portfolio + retention blend."""

    val_frac: float = 0.2
    horizon: int = 1
    _w_ecm: float = field(default=0.4, init=False)
    _w_port: float = field(default=0.35, init=False)
    _w_ret: float = field(default=0.25, init=False)
    _ecm: ECMForecaster = field(default_factory=ECMForecaster, init=False)
    _port: PortfolioModelForecaster = field(default_factory=PortfolioModelForecaster, init=False)
    _ret: RetentionModelForecaster = field(default_factory=RetentionModelForecaster, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PKOInternalForecaster:
        self._ecm = ECMForecaster(val_frac=self.val_frac)
        self._port = PortfolioModelForecaster(horizon=self.horizon, val_frac=self.val_frac)
        self._ret = RetentionModelForecaster(val_frac=self.val_frac)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best = (float("inf"), 0.4, 0.35, 0.25)
        for w_ecm in (0.25, 0.4, 0.55):
            for w_port in (0.2, 0.35, 0.5):
                w_ret = 1.0 - w_ecm - w_port
                if w_ret < 0.05:
                    continue
                ecm = ECMForecaster(val_frac=self.val_frac)
                port = PortfolioModelForecaster(horizon=self.horizon, val_frac=self.val_frac)
                ret = RetentionModelForecaster(val_frac=self.val_frac)
                ecm.fit(Xf, yf)
                port.fit(Xf, yf)
                ret.fit(Xf, yf)
                pred = w_ecm * ecm.predict(Xv) + w_port * port.predict(Xv) + w_ret * ret.predict(Xv)
                mse = _level_mse(yv, pred)
                if mse < best[0]:
                    best = (mse, w_ecm, w_port, w_ret)
        self._w_ecm, self._w_port, self._w_ret = best[1], best[2], best[3]
        self._ecm.fit(X, y)
        self._port.fit(X, y)
        self._ret.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            self._w_ecm * self._ecm.predict(X)
            + self._w_port * self._port.predict(X)
            + self._w_ret * self._ret.predict(X)
        )

    def diagnostics(self) -> dict:
        return {
            "model": "pko_internal_proxy",
            "w_ecm": self._w_ecm,
            "w_portfolio": self._w_port,
            "w_retention": self._w_ret,
        }


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
    if name == "portfolio_model":
        fac = cfg.get("portfolio_model", {})
        return lambda: PortfolioModelForecaster(
            horizon=horizon,
            val_frac=val_frac,
            n_components=int(fac.get("n_components", 2)),
        )
    if name == "segmentation_model":
        return lambda: SegmentationModelForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0, 100.0))),
        )
    if name == "retention_model":
        return lambda: RetentionModelForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0))),
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
    if name == "pko_internal":
        return lambda: PKOInternalForecaster(val_frac=val_frac, horizon=horizon)
    raise KeyError(f"unknown tier-II deposits model {name!r}")
