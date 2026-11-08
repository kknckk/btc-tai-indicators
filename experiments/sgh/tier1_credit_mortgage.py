"""Tier-I mortgage-credit forecasters: ARDL, VECM, dynamic regression, logit scorecard (TRD-38)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge
from statsmodels.tsa.vector_ar.vecm import VECM


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


EXOG_BASE = ("nbp_rate_l1", "cpi_yoy", "cpi_l1", "consumer_l1", "deposits_l1", "bik_l1")


def _ardl_block(X: pd.DataFrame, p: int) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for lag in range(1, p + 1):
        parts.append(X["mortgage_now"].shift(lag).rename(f"ardl_mortgage_l{lag}"))
    for col in EXOG_BASE:
        if col not in X.columns:
            continue
        parts.append(X[col].rename(f"ardl_{col}"))
        for lag in range(2, min(p, 3) + 1):
            parts.append(X[col].shift(lag - 1).rename(f"ardl_{col}_l{lag}"))
    return pd.concat(parts, axis=1)


def _vecm_endog(X: pd.DataFrame) -> pd.DataFrame:
    rate = X["nbp_rate"] if "nbp_rate" in X.columns else X["nbp_rate_l1"]
    m = X["mortgage_now"].clip(lower=1.0)
    return pd.DataFrame(
        {"log_mortgage": np.log(m), "nbp_rate": rate, "cpi_yoy": X["cpi_yoy"]},
        index=X.index,
    )


def _vecm_level_forecast(endog: pd.DataFrame, rank: int, k_ar: int, horizon: int) -> float | None:
    if len(endog) < max(12, k_ar + rank + 4):
        return None
    try:
        mod = VECM(endog, k_ar_diff=k_ar, coint_rank=rank, deterministic="ci")
        res = mod.fit()
        fc = res.predict(steps=max(1, horizon))
        row = fc[-1] if fc.ndim == 2 else fc
        return float(np.exp(row[0]))
    except Exception:
        return None


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
        fallback = X["mortgage_now"].ffill()
        out = np.where(np.isfinite(out), out, fallback.to_numpy())
        return out

    def diagnostics(self) -> dict:
        return {"p": self._p, "n_regressors": len(self._cols)}


@dataclass
class VECMForecaster:
    horizon: int = 1
    val_frac: float = 0.2
    rank_grid: tuple[int, ...] = (0, 1, 2)
    k_ar_grid: tuple[int, ...] = (1, 2)
    _rank: int = field(default=1, init=False)
    _k_ar: int = field(default=1, init=False)
    _train_endog: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def _val_mse(self, endog: pd.DataFrame, y: pd.Series) -> tuple[float, int, int]:
        cut = max(12, int(len(endog) * (1.0 - self.val_frac)))
        if cut >= len(endog) - 5:
            cut = len(endog) // 2
        train, val_idx = endog.iloc[:cut], endog.index[cut:]
        best = (float("inf"), self.rank_grid[0], self.k_ar_grid[0])
        for rank in self.rank_grid:
            for k_ar in self.k_ar_grid:
                errs: list[float] = []
                for idx in val_idx:
                    hist = endog.loc[endog.index <= idx]
                    pred = _vecm_level_forecast(hist, rank, k_ar, self.horizon)
                    yt = y.loc[idx]
                    if pred is not None and np.isfinite(yt):
                        errs.append((pred - yt) ** 2)
                if errs:
                    mse = float(np.mean(errs))
                    if mse < best[0]:
                        best = (mse, rank, k_ar)
        return best

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VECMForecaster:
        endog = _vecm_endog(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog = endog.loc[common]
        y = y.loc[common]
        _, rank, k_ar = self._val_mse(endog, y)
        self._rank, self._k_ar = rank, k_ar
        self._train_endog = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        endog_x = _vecm_endog(X)
        out = np.full(len(X), np.nan)
        for j, idx in enumerate(X.index):
            hist = pd.concat([self._train_endog, endog_x.loc[:idx]]).drop_duplicates()
            pred = _vecm_level_forecast(hist, self._rank, self._k_ar, self.horizon)
            if pred is None or not np.isfinite(pred):
                pred = float(X.loc[idx, "mortgage_now"])
            out[j] = pred
        return out

    def diagnostics(self) -> dict:
        return {"coint_rank": self._rank, "k_ar_diff": self._k_ar, "horizon": self.horizon}


@dataclass
class DynamicRegressionForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DynamicRegressionForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        cols = [c for c in X.columns if c != "mortgage_now"]
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
class LogitScorecardForecaster:
    val_frac: float = 0.2
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    _model: LogisticRegression | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _up_delta: float = field(default=0.0, init=False)
    _down_delta: float = field(default=0.0, init=False)
    _c: float = field(default=1.0, init=False)
    _mu: pd.Series = field(default_factory=pd.Series, init=False)
    _sd: pd.Series = field(default_factory=pd.Series, init=False)
    _fallback: DynamicRegressionForecaster | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogitScorecardForecaster:
        mort = X["mortgage_now"]
        up = (y > mort).astype(int)
        if up.nunique() < 2:
            self._fallback = DynamicRegressionForecaster(val_frac=self.val_frac)
            self._fallback.fit(X, y)
            return self
        deltas = (y - mort).replace([np.inf, -np.inf], np.nan)
        self._up_delta = float(deltas[up == 1].median()) if (up == 1).any() else float(deltas.median())
        self._down_delta = float(deltas[up == 0].median()) if (up == 0).any() else 0.0
        cols = [c for c in X.columns if c != "mortgage_now"]
        mu = X[cols].mean()
        sd = X[cols].std().replace(0.0, 1.0)
        Z = ((X[cols] - mu) / sd).fillna(0.0)
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        Zf, Zv = Z.loc[Xf.index], Z.loc[Xv.index]
        uf, uv = up.loc[Xf.index], up.loc[Xv.index]
        best_mse, best_c = float("inf"), self.c_grid[0]
        for c in self.c_grid:
            clf = LogisticRegression(C=c, max_iter=500, class_weight="balanced")
            clf.fit(Zf, uf)
            p_up = clf.predict_proba(Zv)[:, 1]
            pred = mort.loc[Xv.index].to_numpy() + p_up * self._up_delta + (1.0 - p_up) * self._down_delta
            mse = _level_mse(yv, pred)
            if mse < best_mse:
                best_mse, best_c = mse, c
        self._c = best_c
        self._cols = cols
        self._model = LogisticRegression(C=best_c, max_iter=500, class_weight="balanced")
        self._model.fit(Z, up)
        self._mu, self._sd = mu, sd
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(X)
        assert self._model is not None
        cols = self._cols
        Z = ((X[cols] - self._mu) / self._sd).fillna(0.0)
        p_up = self._model.predict_proba(Z)[:, 1]
        base = X["mortgage_now"].to_numpy()
        return base + p_up * self._up_delta + (1.0 - p_up) * self._down_delta

    def diagnostics(self) -> dict:
        if self._fallback is not None:
            return {"fallback": "ridge", **self._fallback.diagnostics()}
        return {"C": self._c, "up_delta": self._up_delta, "down_delta": self._down_delta}


def model_factory(name: str, cfg: dict, *, horizon: int = 1):
    val_frac = float(cfg.get("val_frac", 0.2))
    if name == "ardl":
        return lambda: ARDLForecaster(
            val_frac=val_frac,
            p_grid=tuple(cfg.get("ardl", {}).get("p", (1, 2, 3, 4))),
        )
    if name == "vecm":
        return lambda: VECMForecaster(
            horizon=horizon,
            val_frac=val_frac,
            rank_grid=tuple(cfg.get("vecm", {}).get("coint_rank", (0, 1, 2))),
            k_ar_grid=tuple(cfg.get("vecm", {}).get("k_ar_diff", (1, 2))),
        )
    if name == "dynamic_regression":
        return lambda: DynamicRegressionForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "logit_scorecard":
        return lambda: LogitScorecardForecaster(
            val_frac=val_frac,
            c_grid=tuple(cfg.get("logit", {}).get("C", (0.01, 0.1, 1.0, 10.0))),
        )
    raise KeyError(f"unknown tier-I mortgage credit model {name!r}")
