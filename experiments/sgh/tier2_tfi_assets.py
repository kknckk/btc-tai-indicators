"""Tier-II TFI net-assets forecasters (TRD-45): portfolio, attractiveness, segmentation, MIDAS, ML."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from experiments.sgh.tier1_bonds import _FactorAR, _level_mse, _split_train_val
from experiments.sgh.tier1_tfi_assets import _split_train_val as _split_train_val_xy

PORTFOLIO_COLS = (
    "tfi_assets",
    "nbp_rate",
    "gus_cpi_yoy",
    "deposits_households",
    "credit_consumer",
    "bond_pl_10y",
)


def _midas_weights(k_max: int, theta1: float, theta2: float) -> np.ndarray:
    k = np.arange(1, k_max + 1, dtype=float)
    x = (k - 0.5) / k_max
    w = np.power(x, theta1 - 1.0) * np.power(1.0 - x, theta2 - 1.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(k_max) / k_max


def _midas_feature_from_mom(X: pd.DataFrame, k_max: int, theta1: float, theta2: float) -> pd.Series:
    w = _midas_weights(k_max, theta1, theta2)
    parts: list[pd.Series] = []
    for i in range(k_max):
        col = f"tfi_mom_l{i + 1}" if i > 0 else "tfi_mom_l1"
        if col not in X.columns and i == 0:
            mom = X["tfi_now"].pct_change()
            parts.append(w[i] * mom.shift(1))
        elif col in X.columns:
            parts.append(w[i] * X[col])
        else:
            lag = X["tfi_now"].pct_change().shift(i + 1)
            parts.append(w[i] * lag)
    return pd.concat(parts, axis=1).sum(axis=1, skipna=True)


def _ml_feature_cols(X: pd.DataFrame) -> list[str]:
    skip = {
        "tfi_assets",
        "tfi_now",
        "deposits_households",
        "credit_consumer",
        "bond_pl_10y",
        "nbp_rate",
        "gus_cpi_yoy",
    }
    return [c for c in X.columns if c not in skip and X[c].dtype != object]


@dataclass
class PortfolioModelForecaster:
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
        if len(cols) < 3:
            cols = [c for c in ("tfi_assets", "nbp_rate", "gus_cpi_yoy") if c in X.columns]
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
        fallback = X["tfi_now"].ffill().to_numpy()
        return np.where(np.isfinite(pred), pred, fallback)

    def diagnostics(self) -> dict:
        return {"model": "portfolio_pca_ar", "n_components": self.n_components, "horizon": self.horizon}


@dataclass
class RelativeAttractivenessForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def _attr_cols(self, X: pd.DataFrame) -> list[str]:
        return [
            c
            for c in (
                "tfi_l1",
                "tfi_l12",
                "attractiveness_l1",
                "bond_minus_dep_l1",
                "bond_minus_rate_l1",
                "perf_minus_bond_l1",
                "perf_minus_dep_l1",
                "perf_l1",
                "flow_l1",
                "nbp_rate_l1",
                "cpi_yoy",
            )
            if c in X.columns
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RelativeAttractivenessForecaster:
        cols = self._attr_cols(X)
        if "tfi_l1" not in cols:
            cols = ["tfi_l1"] + cols
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
        return {"model": "relative_attractiveness", "alpha": self._alpha, "n_features": len(self._cols)}


@dataclass
class SegmentationModelForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    _alpha: float = field(default=1.0, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _high: Ridge | None = field(default=None, init=False)
    _low: Ridge | None = field(default=None, init=False)
    _flow_threshold: float = field(default=0.0, init=False)

    def _seg_cols(self, X: pd.DataFrame) -> list[str]:
        return [
            c
            for c in (
                "tfi_l1",
                "tfi_l3",
                "flow_l1",
                "flow_l2",
                "perf_l1",
                "nbp_rate_l1",
                "attractiveness_l1",
            )
            if c in X.columns
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SegmentationModelForecaster:
        cols = self._seg_cols(X)
        self._cols = cols
        flow = X.get("flow_l1", X.get("tfi_mom_l1", pd.Series(0.0, index=X.index)))
        self._flow_threshold = float(flow.median())
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            high_m = Ridge(alpha=alpha)
            low_m = Ridge(alpha=alpha)
            hi = flow.loc[Xf.index] >= self._flow_threshold
            lo = ~hi
            if int(hi.sum()) < 5 or int(lo.sum()) < 5:
                continue
            high_m.fit(Xf.loc[hi, cols].fillna(0.0), yf.loc[hi])
            low_m.fit(Xf.loc[lo, cols].fillna(0.0), yf.loc[lo])
            pred = np.zeros(len(Xv))
            hi_v = flow.loc[Xv.index] >= self._flow_threshold
            pred[hi_v.to_numpy()] = high_m.predict(Xv.loc[hi_v, cols].fillna(0.0))
            pred[~hi_v.to_numpy()] = low_m.predict(Xv.loc[~hi_v, cols].fillna(0.0))
            mse = _level_mse(yv, pred)
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self._alpha = best_alpha
        self._high = Ridge(alpha=best_alpha)
        self._low = Ridge(alpha=best_alpha)
        hi = flow >= self._flow_threshold
        lo = ~hi
        if int(hi.sum()) < 5 or int(lo.sum()) < 5:
            mono = Ridge(alpha=best_alpha)
            mono.fit(X[cols].fillna(0.0), y)
            self._high = self._low = mono
            self._flow_threshold = float("-inf")
        else:
            self._high.fit(X.loc[hi, cols].fillna(0.0), y.loc[hi])
            self._low.fit(X.loc[lo, cols].fillna(0.0), y.loc[lo])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._high is not None and self._low is not None
        if self._high is self._low:
            return self._high.predict(X[self._cols].fillna(0.0))
        flow = X.get("flow_l1", X.get("tfi_mom_l1", pd.Series(0.0, index=X.index)))
        out = np.zeros(len(X))
        hi = flow >= self._flow_threshold
        out[hi.to_numpy()] = self._high.predict(X.loc[hi, self._cols].fillna(0.0))
        out[~hi.to_numpy()] = self._low.predict(X.loc[~hi, self._cols].fillna(0.0))
        return out

    def diagnostics(self) -> dict:
        return {
            "model": "segmentation_flow_regime",
            "flow_threshold": self._flow_threshold,
            "alpha": self._alpha,
        }


@dataclass
class MidasForecaster:
    val_frac: float = 0.2
    k_grid: tuple[int, ...] = (6, 9, 12)
    theta1_grid: tuple[float, ...] = (1.0, 2.0)
    theta2_grid: tuple[float, ...] = (3.0, 5.0)
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    _k: int = field(default=9, init=False)
    _t1: float = field(default=1.0, init=False)
    _t2: float = field(default=3.0, init=False)
    _alpha: float = field(default=1.0, init=False)
    _model: Ridge | None = field(default=None, init=False)

    def _design(self, X: pd.DataFrame, k: int, t1: float, t2: float) -> pd.DataFrame:
        parts: dict[str, pd.Series] = {"midas_mom": _midas_feature_from_mom(X, k, t1, t2)}
        if "tfi_l1" in X.columns:
            parts["tfi_l1"] = X["tfi_l1"]
        elif "tfi_now" in X.columns:
            parts["tfi_l1"] = X["tfi_now"].shift(1)
        if "flow_l1" in X.columns:
            parts["flow_l1"] = X["flow_l1"]
        return pd.DataFrame(parts, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MidasForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best = (float("inf"), self.k_grid[0], self.theta1_grid[0], self.theta2_grid[0], self.alpha_grid[0])
        for k in self.k_grid:
            for t1 in self.theta1_grid:
                for t2 in self.theta2_grid:
                    Df = self._design(Xf, k, t1, t2)
                    Dv = self._design(Xv, k, t1, t2)
                    for alpha in self.alpha_grid:
                        m = Ridge(alpha=alpha)
                        m.fit(Df.fillna(0.0), yf)
                        pred = m.predict(Dv.fillna(0.0))
                        mse = _level_mse(yv, pred)
                        if mse < best[0]:
                            best = (mse, k, t1, t2, alpha)
        self._k, self._t1, self._t2, self._alpha = best[1], best[2], best[3], best[4]
        D = self._design(X, self._k, self._t1, self._t2)
        self._model = Ridge(alpha=self._alpha)
        self._model.fit(D.fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        D = self._design(X, self._k, self._t1, self._t2)
        pred = self._model.predict(D.fillna(0.0))
        fallback = X["tfi_now"].ffill().to_numpy()
        return np.where(np.isfinite(pred), pred, fallback)

    def diagnostics(self) -> dict:
        return {"model": "midas_ridge", "k": self._k, "theta1": self._t1, "theta2": self._t2, "alpha": self._alpha}


@dataclass
class MixedFrequencyForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def _mf_cols(self, X: pd.DataFrame) -> list[str]:
        return [
            c
            for c in (
                "tfi_l1",
                "tfi_mom_l1",
                "tfi_mom_l2",
                "tfi_q_mean_l1",
                "cpi_q_mean_l1",
                "cpi_y_l1",
                "cpi_yoy",
                "cpi_l1",
                "deposits_mom_l1",
                "nbp_rate_l1",
            )
            if c in X.columns
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MixedFrequencyForecaster:
        cols = self._mf_cols(X)
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
        return {"model": "mixed_frequency", "alpha": self._alpha, "n_features": len(self._cols)}


@dataclass
class SentimentModelForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    max_features: int = 32
    _vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    _model: Ridge | None = field(default=None, init=False)
    _alpha: float = field(default=1.0, init=False)
    _fallback: Ridge | None = field(default=None, init=False)
    _fallback_cols: list[str] = field(default_factory=list, init=False)

    def _text_series(self, X: pd.DataFrame) -> pd.Series:
        flow = X.get("flow_z_l1", pd.Series(0.0, index=X.index)).fillna(0.0)
        perf = X.get("perf_z_l1", pd.Series(0.0, index=X.index)).fillna(0.0)
        sent = X.get("sentiment_index_l1", pd.Series(0.0, index=X.index)).fillna(0.0)
        return pd.Series(
            [
                f"flow {f:.2f} perf {p:.2f} sentiment {s:.2f}"
                for f, p, s in zip(flow.to_numpy(), perf.to_numpy(), sent.to_numpy())
            ],
            index=X.index,
        )

    def _ridge_cols(self, X: pd.DataFrame) -> list[str]:
        return [c for c in ("sentiment_index_l1", "flow_z_l1", "perf_z_l1", "tfi_l1") if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SentimentModelForecaster:
        texts = self._text_series(X)
        ridge_cols = self._ridge_cols(X)
        (Xf, yf), (Xv, yv) = _split_train_val_xy(X, y, self.val_frac)
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            vec = TfidfVectorizer(max_features=self.max_features, min_df=1, ngram_range=(1, 2))
            Zf = vec.fit_transform(texts.loc[Xf.index])
            Zv = vec.transform(texts.loc[Xv.index])
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
        self._fallback_cols = ridge_cols
        fb = Ridge(alpha=best_alpha)
        fb.fit(X[ridge_cols].fillna(0.0), y)
        self._fallback = fb
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._vectorizer is not None and self._model is not None
        Z = self._vectorizer.transform(self._text_series(X))
        return self._model.predict(Z)

    def diagnostics(self) -> dict:
        return {"model": "sentiment_tfidf", "alpha": self._alpha, "max_features": self.max_features}


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
    if name == "portfolio_model":
        fac = cfg.get("portfolio_model", {})
        return lambda: PortfolioModelForecaster(
            horizon=horizon,
            val_frac=val_frac,
            n_components=int(fac.get("n_components", 2)),
        )
    if name == "relative_attractiveness":
        return lambda: RelativeAttractivenessForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0, 100.0))),
        )
    if name == "segmentation_model":
        return lambda: SegmentationModelForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0, 100.0))),
        )
    if name == "midas":
        mid = cfg.get("midas", {})
        return lambda: MidasForecaster(
            val_frac=val_frac,
            k_grid=tuple(mid.get("k", (6, 9, 12))),
            theta1_grid=tuple(mid.get("theta1", (1.0, 2.0))),
            theta2_grid=tuple(mid.get("theta2", (3.0, 5.0))),
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0))),
        )
    if name == "mixed_frequency":
        return lambda: MixedFrequencyForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0, 100.0))),
        )
    if name == "sentiment_model":
        return lambda: SentimentModelForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.1, 1.0, 10.0))),
            max_features=int(cfg.get("nlp", {}).get("max_features", 32)),
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
    raise KeyError(f"unknown tier-II tfi_assets model {name!r}")
