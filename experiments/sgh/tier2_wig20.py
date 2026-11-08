"""Tier-II WIG20 forecasters: HAR-RV, realized vol, externals, SV, sentiment, RF/GB (TRD-51)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from experiments.sgh.tier1_wig20 import _level_mse, _split_train_val
from experiments.sgh.wig20_panel import PRICE_COL


def _split_train_val_xy(X: pd.DataFrame, y: pd.Series, val_frac: float) -> tuple[tuple, tuple]:
    n = len(X)
    tr, va = _split_train_val(n, val_frac)
    return (X.iloc[tr], y.iloc[tr]), (X.iloc[va], y.iloc[va])


def _ml_feature_cols(X: pd.DataFrame) -> list[str]:
    skip = {PRICE_COL}
    return [c for c in X.columns if c not in skip and X[c].dtype != object]


@dataclass
class FeatureRidgeForecaster:
    feature_cols: tuple[str, ...]
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    model_name: str = "ridge"
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def _resolve_cols(self, X: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_cols if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureRidgeForecaster:
        cols = self._resolve_cols(X)
        if not cols:
            cols = [f"{PRICE_COL}_l1"] if f"{PRICE_COL}_l1" in X.columns else [PRICE_COL]
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
        fallback = X[PRICE_COL].astype(float).to_numpy()
        pred = self._model.predict(X[self._cols].fillna(0.0))
        return np.where(np.isfinite(pred), pred, fallback)

    def diagnostics(self) -> dict:
        return {"model": self.model_name, "alpha": self._alpha, "n_features": len(self._cols)}


@dataclass
class StochasticVolLevelForecaster:
    returns: pd.Series
    price_col: str
    horizon: int
    val_frac: float = 0.2
    phi_grid: tuple[float, ...] = (0.85, 0.9, 0.95)
    _phi: float = field(default=0.9, init=False)
    _drift_scale: float = field(default=1.0, init=False)

    def _log_rv(self, r: pd.Series) -> pd.Series:
        return np.log(r.astype(float) ** 2 + 1e-10)

    def _level_from_state(self, price: float, hist: pd.Series, phi: float, drift_scale: float) -> float:
        hist = hist.dropna()
        if len(hist) < 30:
            return float(price)
        log_rv = self._log_rv(hist)
        mu = float(log_rv.mean())
        last = float(log_rv.iloc[-1])
        log_rv_fc = mu + (phi ** self.horizon) * (last - mu)
        vol_fc = float(np.exp(0.5 * log_rv_fc))
        mom = float(hist.iloc[-21:].mean()) if len(hist) >= 21 else 0.0
        logret_fc = drift_scale * self.horizon * mom - 0.5 * vol_fc ** 2 * self.horizon
        return float(price * np.exp(logret_fc))

    def _predict_slice(self, X: pd.DataFrame, phi: float, drift_scale: float) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=float)
        for j, t in enumerate(X.index):
            hist = self.returns.loc[:t].dropna()
            price = float(X[self.price_col].loc[t])
            out[j] = self._level_from_state(price, hist, phi, drift_scale)
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StochasticVolLevelForecaster:
        n = len(X)
        tr, va = _split_train_val(n, self.val_frac)
        X_va, y_va = X.iloc[va], y.iloc[va]
        best = (float("inf"), self.phi_grid[0], 1.0)
        for phi in self.phi_grid:
            for drift_scale in (0.5, 1.0, 1.5):
                preds = self._predict_slice(X_va, phi, drift_scale)
                mse = _level_mse(y_va, preds)
                if mse < best[0]:
                    best = (mse, phi, drift_scale)
        self._phi, self._drift_scale = best[1], best[2]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_slice(X, self._phi, self._drift_scale)

    def diagnostics(self) -> dict:
        return {
            "model": "stochastic_vol",
            "phi": self._phi,
            "drift_scale": self._drift_scale,
            "horizon": self.horizon,
        }


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


def model_factory(
    name: str,
    cfg: dict,
    *,
    horizon: int,
    price_col: str = PRICE_COL,
    returns: pd.Series | None = None,
):
    val_frac = float(cfg.get("val_frac", 0.2))
    ridge_alpha = tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0)))
    if name == "har_rv":
        har = cfg.get("har_rv", {})
        cols = tuple(har.get("features", ("rv_d", "rv_w", "rv_m", f"{PRICE_COL}_l1", f"{PRICE_COL}_l5")))
        return lambda: FeatureRidgeForecaster(
            feature_cols=cols,
            val_frac=val_frac,
            alpha_grid=ridge_alpha,
            model_name="har_rv",
        )
    if name == "realized_vol":
        rv = cfg.get("realized_vol", {})
        cols = tuple(
            rv.get(
                "features",
                ("rv_sum_5", "rv_sum_22", "bipower_5", f"{PRICE_COL}_vol21", f"{PRICE_COL}_l1"),
            )
        )
        return lambda: FeatureRidgeForecaster(
            feature_cols=cols,
            val_frac=val_frac,
            alpha_grid=ridge_alpha,
            model_name="realized_vol",
        )
    if name == "external_factors":
        ext = cfg.get("external_factors", {})
        cols = tuple(
            ext.get(
                "features",
                (
                    f"{PRICE_COL}_l1",
                    f"{PRICE_COL}_l5",
                    "usdpln_lr_l1",
                    "eurpln_lr_l1",
                    "oil_lr_l1",
                    "gold_lr_l1",
                ),
            )
        )
        return lambda: FeatureRidgeForecaster(
            feature_cols=cols,
            val_frac=val_frac,
            alpha_grid=ridge_alpha,
            model_name="external_factors",
        )
    if name == "stochastic_vol":
        sv = cfg.get("stochastic_vol", {})
        if returns is None:
            raise ValueError("stochastic_vol requires returns series")
        return lambda: StochasticVolLevelForecaster(
            returns=returns,
            price_col=price_col,
            horizon=horizon,
            val_frac=val_frac,
            phi_grid=tuple(sv.get("phi", (0.85, 0.9, 0.95))),
        )
    if name == "sentiment":
        sent = cfg.get("sentiment", {})
        cols = tuple(
            sent.get(
                "features",
                (
                    f"{PRICE_COL}_l1",
                    "sentiment_fear",
                    "sentiment_stress",
                    "sentiment_momentum",
                    "ret_z21",
                ),
            )
        )
        return lambda: FeatureRidgeForecaster(
            feature_cols=cols,
            val_frac=val_frac,
            alpha_grid=ridge_alpha,
            model_name="sentiment",
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
    raise KeyError(f"unknown tier-II WIG20 model {name!r}")
