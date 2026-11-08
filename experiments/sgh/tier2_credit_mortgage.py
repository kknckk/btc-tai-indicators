"""Tier-II mortgage-credit forecasters: VAR, panel, contract, ML, LSTM, PKO proxy (TRD-39)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.api import VAR

from experiments.sgh.tier1_credit_mortgage import (
    ARDLForecaster,
    DynamicRegressionForecaster,
    _level_mse,
    _split_train_val,
)
from experiments.sgh.tier1_wheat import LSTMLevelForecaster

ENDOG_COLS = ("credit_mortgage", "nbp_rate", "gus_cpi_yoy", "credit_consumer")


def _split_train_val_xy(X: pd.DataFrame, y: pd.Series, val_frac: float) -> tuple[tuple, tuple]:
    n = len(X)
    cut = max(12, int(n * (1.0 - val_frac)))
    if cut >= n - 5:
        cut = n // 2
    return (X.iloc[:cut], y.iloc[:cut]), (X.iloc[cut:], y.iloc[cut:])


def _endog_from_X(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ENDOG_COLS if c in X.columns]
    if len(cols) < 3:
        cols = [c for c in ("credit_mortgage", "nbp_rate", "gus_cpi_yoy") if c in X.columns]
    missing = [c for c in ("credit_mortgage", "nbp_rate", "gus_cpi_yoy") if c not in cols]
    if missing:
        raise KeyError(f"tier-II mortgage features missing columns: {missing}")
    return X.loc[:, cols].astype(float)


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


def _grid_search_var(
    endog: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    val_frac: float,
    p_grid: tuple[int, ...],
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
            pred = _var_forecast(hist, p, horizon)
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
        "credit_mortgage",
        "credit_consumer",
        "deposits_households",
        "nbp_rate",
        "gus_cpi_yoy",
        "mortgage_now",
    }
    return [c for c in X.columns if c not in skip and X[c].dtype != object]


def _panel_cols(X: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in (
            "mortgage_now",
            "mortgage_l1",
            "consumer_l1",
            "deposits_l1",
            "mtg_share_l1",
            "consumer_share_l1",
            "consumer_mom_l1",
            "nbp_rate_l1",
            "cpi_yoy",
        )
        if c in X.columns
    ]
    return cols or [c for c in ("mortgage_now", "mortgage_l1", "nbp_rate_l1", "cpi_yoy") if c in X.columns]


def _contract_cols(X: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in (
            "mortgage_now",
            "mortgage_l1",
            "mortgage_mom_l1",
            "rate_chg_l1",
            "nbp_rate_l1",
            "nbp_rate_l3",
            "cpi_yoy",
            "cpi_l1",
            "affordability_l1",
            "consumer_l1",
        )
        if c in X.columns
    ]
    return cols or [c for c in ("mortgage_now", "mortgage_l1", "nbp_rate_l1", "cpi_yoy") if c in X.columns]


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
        self._p, _ = _grid_search_var(endog, y, self.horizon, self.val_frac, self.p_grid)
        self._hist = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        for j, idx in enumerate(X.index):
            ex = _endog_from_X(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            pred = _var_forecast(hist, self._p, self.horizon)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, "mortgage_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "horizon": self.horizon, "model": "var"}


@dataclass
class PanelForecaster:
    """Pooled macro panel proxy: mortgage stock vs consumer/deposit shares (Ridge)."""

    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PanelForecaster:
        cols = _panel_cols(X)
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
        return {"alpha": self._alpha, "n_features": len(self._cols), "model": "panel_ridge"}


@dataclass
class MortgageContractForecaster:
    """Stock–flow / affordability contract proxy (rate, CPI, collateral ratios)."""

    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MortgageContractForecaster:
        cols = _contract_cols(X)
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
        return {"alpha": self._alpha, "n_features": len(self._cols), "model": "mortgage_contract"}


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
        self._params = best[1] or {"n_estimators": 100, "max_depth": 4}
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
        self._params = best[1] or {"max_depth": 3, "learning_rate": 0.05}
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
    """Open proxy: blend ARDL, dynamic Ridge, momentum when PKO models unavailable."""

    val_frac: float = 0.2
    _w_ardl: float = field(default=0.4, init=False)
    _w_dyn: float = field(default=0.35, init=False)
    _w_mom: float = field(default=0.25, init=False)
    _ardl: ARDLForecaster = field(default_factory=ARDLForecaster, init=False)
    _dyn: DynamicRegressionForecaster = field(default_factory=DynamicRegressionForecaster, init=False)
    _mom_alpha: float = field(default=1.0, init=False)
    _mom_cols: list[str] = field(default_factory=list, init=False)
    _mom: Ridge = field(default_factory=lambda: Ridge(alpha=1.0), init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PKOInternalForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        best = (float("inf"), 0.4, 0.35, 0.25, 1.0)
        for w_ardl in (0.25, 0.4, 0.55):
            for w_dyn in (0.2, 0.35, 0.5):
                w_mom = 1.0 - w_ardl - w_dyn
                if w_mom < 0.05:
                    continue
                for alpha in (0.5, 1.0, 2.0):
                    ardl = ARDLForecaster(val_frac=self.val_frac)
                    dyn = DynamicRegressionForecaster(val_frac=self.val_frac)
                    ardl.fit(Xf, yf)
                    dyn.fit(Xf, yf)
                    mom = Ridge(alpha=alpha)
                    mom_cols = [
                        c
                        for c in ("mortgage_now", "mortgage_mom_l1", "rate_chg_l1", "affordability_l1")
                        if c in X.columns
                    ]
                    if not mom_cols:
                        mom_cols = ["mortgage_now", "mortgage_mom_l1"]
                        mom_cols = [c for c in mom_cols if c in X.columns]
                    mom.fit(Xf[mom_cols].fillna(0.0), yf)
                    pred = (
                        w_ardl * ardl.predict(Xv)
                        + w_dyn * dyn.predict(Xv)
                        + w_mom * mom.predict(Xv[mom_cols].fillna(0.0))
                    )
                    mse = _level_mse(yv, pred)
                    if mse < best[0]:
                        best = (mse, w_ardl, w_dyn, w_mom, alpha)
        self._w_ardl, self._w_dyn, self._w_mom, self._mom_alpha = best[1], best[2], best[3], best[4]
        self._ardl = ARDLForecaster(val_frac=self.val_frac)
        self._dyn = DynamicRegressionForecaster(val_frac=self.val_frac)
        self._ardl.fit(X, y)
        self._dyn.fit(X, y)
        self._mom_cols = [
            c
            for c in ("mortgage_now", "mortgage_mom_l1", "rate_chg_l1", "affordability_l1")
            if c in X.columns
        ]
        if not self._mom_cols:
            self._mom_cols = [c for c in ("mortgage_now", "mortgage_mom_l1") if c in X.columns]
        self._mom = Ridge(alpha=self._mom_alpha)
        self._mom.fit(X[self._mom_cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            self._w_ardl * self._ardl.predict(X)
            + self._w_dyn * self._dyn.predict(X)
            + self._w_mom * self._mom.predict(X[self._mom_cols].fillna(0.0))
        )

    def diagnostics(self) -> dict:
        return {
            "model": "pko_internal_proxy",
            "w_ardl": self._w_ardl,
            "w_dyn": self._w_dyn,
            "w_mom": self._w_mom,
            "mom_alpha": self._mom_alpha,
        }


def model_factory(name: str, cfg: dict, *, horizon: int = 1):
    val_frac = float(cfg.get("val_frac", 0.2))
    var_cfg = cfg.get("var", {})
    p_grid = tuple(var_cfg.get("p", (1, 2, 3, 4)))
    seed = int(cfg.get("seed", 42))
    if name == "var":
        return lambda: VARForecaster(horizon=horizon, val_frac=val_frac, p_grid=p_grid)
    if name == "panel_model":
        return lambda: PanelForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "mortgage_contract":
        return lambda: MortgageContractForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
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
    if name == "lstm":
        lc = dict(cfg.get("lstm", {}))
        lc["val_frac"] = val_frac
        return lambda: LSTMLevelForecaster(lc, seed, "mortgage_now")
    if name == "pko_internal":
        return lambda: PKOInternalForecaster(val_frac=val_frac)
    raise KeyError(f"unknown tier-II mortgage credit model {name!r}")
