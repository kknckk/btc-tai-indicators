"""Tier-II NBP reference-rate forecasters: VAR/BVAR/SVAR, expectations+comm, ML, NLP (TRD-33)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from statsmodels.tsa.api import VAR

try:
    from statsmodels.tsa.vector_ar.svar_model import SVAR
except ImportError:  # pragma: no cover
    SVAR = None  # type: ignore[misc, assignment]

ENDOG_COLS = ("nbp_rate", "gus_cpi_yoy", "gus_pmi")


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


def _endog_from_X(X: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ENDOG_COLS if c not in X.columns]
    if missing:
        raise KeyError(f"tier-II NBP features missing columns: {missing}")
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
        for j in range(k):
            yj = Y[:, j]
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Z, yj)
            coefs.append(ridge.coef_)
        state = vals[-p:].reshape(-1)
        path = vals.copy()
        for _ in range(max(1, horizon)):
            nxt = np.zeros(k)
            for j in range(k):
                nxt[j] = float(np.dot(coefs[j], state))
            path = np.vstack([path, nxt])
            state = path[-p:].reshape(-1)
        return float(path[-1, 0])
    except Exception:
        return None


def _svar_forecast(endog: pd.DataFrame, p: int, horizon: int) -> float | None:
    if SVAR is None or len(endog) < p + 12:
        return None
    try:
        A = np.eye(3)
        A[1, 0] = 1
        A[2, 0] = 1
        A[2, 1] = 1
        mod = SVAR(endog, svar_type="A", A=A)
        res = mod.fit(maxlags=p, trend="c", maxiter=500, disp=False)
        fc = res.forecast(endog.values[-p:], steps=max(1, horizon))
        row = fc[-1] if fc.ndim == 2 else fc
        return float(row[0])
    except Exception:
        return _var_forecast(endog, p, horizon)


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
                pred = float(X.loc[idx, "rate_now"])
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
                lambda e, p, h: _ridge_var_forecast(e, p, h, alpha),
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
                pred = float(X.loc[idx, "rate_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "alpha": self._alpha, "horizon": self.horizon, "model": "bvar_ridge"}


@dataclass
class SVARForecaster:
    horizon: int = 1
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3)
    _p: int = field(default=2, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    _used_svar: bool = field(default=True, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SVARForecaster:
        endog = _endog_from_X(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        self._p, _ = _grid_search_var(endog, y, self.horizon, self.val_frac, self.p_grid, _svar_forecast)
        self._hist = endog
        self._used_svar = SVAR is not None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        for j, idx in enumerate(X.index):
            ex = _endog_from_X(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            pred = _svar_forecast(hist, self._p, self.horizon)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, "rate_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "horizon": self.horizon, "model": "svar", "svar_available": self._used_svar}


@dataclass
class ExpectationsCommForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ExpectationsCommForecaster:
        cols = [
            c
            for c in (
                "rate_now",
                "rate_l1",
                "exp_rate_3m",
                "exp_rate_12m",
                "rpp_tone",
                "rpp_tone_l1",
                "tone_x_infl",
                "months_since_rate_change",
                "infl_gap",
                "pmi_gap",
            )
            if c in X.columns
        ]
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
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
        return {"alpha": self._alpha, "n_features": len(self._cols), "model": "expectations_comm"}


@dataclass
class RandomForestForecaster:
    val_frac: float = 0.2
    n_estimators_grid: tuple[int, ...] = (100, 200)
    max_depth_grid: tuple[int | None, ...] = (4, 8, None)
    _model: RandomForestRegressor | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _params: dict = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestForecaster:
        cols = [c for c in X.columns if c not in ("rpp_comm_text",) and X[c].dtype != object]
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
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
        cols = [c for c in X.columns if c not in ("rpp_comm_text",) and X[c].dtype != object]
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
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
class RppNlpForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    max_features: int = 64
    _vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    _model: Ridge | None = field(default=None, init=False)
    _alpha: float = field(default=1.0, init=False)
    _fallback: ExpectationsCommForecaster | None = field(default=None, init=False)

    def _text_series(self, X: pd.DataFrame) -> pd.Series:
        if "rpp_comm_text" in X.columns:
            base = X["rpp_comm_text"].fillna("").astype(str)
            if base.str.len().sum() >= 5:
                return base
        tone = X.get("rpp_tone", pd.Series(0.0, index=X.index)).fillna(0.0)
        infl = X.get("infl_gap", pd.Series(0.0, index=X.index)).fillna(0.0)
        return pd.Series(
            [f"rpp tone {t:.2f} infl gap {g:.2f}" for t, g in zip(tone.to_numpy(), infl.to_numpy())],
            index=X.index,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RppNlpForecaster:
        texts = self._text_series(X)
        if texts.str.len().sum() < 5:
            self._fallback = ExpectationsCommForecaster(val_frac=self.val_frac, alpha_grid=self.alpha_grid)
            self._fallback.fit(X, y)
            return self
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        tf_train = texts.loc[Xf.index]
        tf_val = texts.loc[Xv.index]
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            vec = TfidfVectorizer(max_features=self.max_features, min_df=1, ngram_range=(1, 2))
            Zf = vec.fit_transform(tf_train)
            Zv = vec.transform(tf_val)
            m = Ridge(alpha=alpha)
            m.fit(Zf, yf)
            pred = m.predict(Zv)
            mse = _level_mse(yv, pred)
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self._alpha = best_alpha
        self._vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=1, ngram_range=(1, 2))
        Z = self._vectorizer.fit_transform(texts)
        self._model = Ridge(alpha=best_alpha)
        self._model.fit(Z, y)
        self._fallback = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(X)
        assert self._vectorizer is not None and self._model is not None
        Z = self._vectorizer.transform(self._text_series(X))
        return self._model.predict(Z)

    def diagnostics(self) -> dict:
        if self._fallback is not None:
            return {"model": "rpp_nlp", "fallback": "expectations_comm", **self._fallback.diagnostics()}
        return {"model": "rpp_nlp", "alpha": self._alpha, "max_features": self.max_features}


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
    if name == "svar":
        return lambda: SVARForecaster(
            horizon=horizon,
            val_frac=val_frac,
            p_grid=tuple(var_cfg.get("p_svar", (1, 2, 3))),
        )
    if name == "expectations_comm":
        return lambda: ExpectationsCommForecaster(
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
    if name == "rpp_nlp":
        return lambda: RppNlpForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0))),
            max_features=int(cfg.get("nlp", {}).get("max_features", 64)),
        )
    raise KeyError(f"unknown tier-II NBP model {name!r}")
