"""Tier-II PL bond yield forecasters: PCA/VAR/macro/AFNS/LSTM (TRD-35)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from statsmodels.tsa.api import VAR

from experiments.sgh.bonds_panel import BOND_COLS, MATURITIES, MATURITY_YEARS
from experiments.sgh.tier1_bonds import (
    _FactorAR,
    _extract_factors,
    _level_mse,
    _ns_loadings,
    _split_train_val,
    _yield_matrix,
)
from experiments.sgh.tier1_wheat import LSTMLevelForecaster, _SeqLSTM, _val_mse

FACTOR_COLS = ("ns_f1", "ns_f2", "ns_f3")
EXOG_COLS = ("nbp_rate", "gus_cpi_yoy")


def _split_train_val_xy(n: int, val_frac: float) -> tuple[slice, slice]:
    return _split_train_val(n, val_frac)


def _endog_factors(X: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FACTOR_COLS if c not in X.columns]
    if missing:
        raise KeyError(f"tier-II bond features missing {missing}")
    return X.loc[:, list(FACTOR_COLS)].astype(float)


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


def _var_forecast(endog: pd.DataFrame, p: int, horizon: int) -> np.ndarray | None:
    if len(endog) < p + 8:
        return None
    try:
        mod = VAR(endog)
        res = mod.fit(maxlags=p, ic=None, trend="c")
        fc = res.forecast(endog.values[-p:], steps=max(1, horizon))
        row = fc[-1] if fc.ndim == 2 else fc
        return np.asarray(row, dtype=float)
    except Exception:
        return None


def _ridge_var_forecast(endog: pd.DataFrame, p: int, horizon: int, alpha: float) -> np.ndarray | None:
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
        return path[-1]
    except Exception:
        return None


def _varx_forecast(
    endog: pd.DataFrame,
    exog: pd.DataFrame,
    p: int,
    horizon: int,
    alpha: float,
) -> np.ndarray | None:
    if len(endog) < p + 8:
        return None
    try:
        e_vals = endog.to_numpy()
        x_vals = exog.reindex(endog.index).ffill().fillna(0.0).to_numpy()
        n, k = e_vals.shape
        q = x_vals.shape[1]
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for t in range(p, n):
            lag_e = e_vals[t - p : t].reshape(-1)
            lag_x = x_vals[t - p : t].reshape(-1)
            rows.append(np.concatenate([lag_e, lag_x]))
            targets.append(e_vals[t])
        Z = np.asarray(rows)
        Y = np.asarray(targets)
        coefs, intercepts = [], []
        for j in range(k):
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Z, Y[:, j])
            coefs.append(ridge.coef_)
            intercepts.append(float(ridge.intercept_))
        e_path = e_vals.copy()
        x_path = x_vals.copy()
        for _ in range(max(1, horizon)):
            lag_e = e_path[-p:].reshape(-1)
            lag_x = x_path[-p:].reshape(-1)
            state = np.concatenate([lag_e, lag_x])
            nxt = np.zeros(k)
            for j in range(k):
                nxt[j] = intercepts[j] + float(np.dot(coefs[j], state))
            e_path = np.vstack([e_path, nxt])
            x_path = np.vstack([x_path, x_vals[-1]])
        return e_path[-1]
    except Exception:
        return None


def _grid_search_var(
    endog: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    val_frac: float,
    p_grid: tuple[int, ...],
    forecast_fn,
    mat: float,
    lam: float,
) -> tuple[int, float]:
    n = len(endog)
    cut = max(30, int(n * (1.0 - val_frac)))
    if cut >= n - 10:
        cut = n // 2
    val_idx = endog.index[cut:]
    load_fn = lambda tau: _ns_loadings(tau, lam)
    best = (float("inf"), p_grid[0])
    for p in p_grid:
        errs: list[float] = []
        for idx in val_idx:
            hist = endog.loc[endog.index <= idx]
            fac_h = forecast_fn(hist, p, horizon)
            if fac_h is None:
                continue
            pred = float(fac_h @ load_fn(np.array([mat]))[0])
            yt = y.loc[idx]
            if np.isfinite(yt) and np.isfinite(pred):
                errs.append((pred - yt) ** 2)
        if errs:
            mse = float(np.mean(errs))
            if mse < best[0]:
                best = (mse, p)
    return best[1], best[0]


def _factor_level_pred(fac_h: np.ndarray, maturity: float, lam: float) -> float:
    row = _ns_loadings(np.array([maturity]), lam)[0]
    return float(fac_h @ row)


@dataclass
class PCACurveForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    n_components: int = 3
    _pca: PCA | None = field(default=None, init=False)
    _ar: _FactorAR = field(default_factory=lambda: _FactorAR(np.zeros(3), np.ones(3)), init=False)
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(4), init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PCACurveForecaster:
        Y = _yield_matrix(X)
        n = len(Y)
        tr, va = _split_train_val_xy(n, self.val_frac)
        self._mean = Y[tr].mean(axis=0)
        centered = Y - self._mean
        self._pca = PCA(n_components=self.n_components)
        scores = self._pca.fit_transform(centered)
        self._ar = _FactorAR.fit(scores[tr])
        mat = self._maturity()
        mat_idx = list(BOND_COLS).index(self.target_col)
        preds = []
        for i in range(va.start, va.stop):
            fac_h = self._ar.steps(scores[i], self.horizon)
            recon = fac_h @ self._pca.components_ + self._mean
            preds.append(recon[mat_idx])
        _ = _level_mse(y.iloc[va], np.asarray(preds))
        scores_all = self._pca.transform(centered)
        self._ar = _FactorAR.fit(scores_all[tr])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pca is not None
        Y = _yield_matrix(X)
        centered = Y - self._mean
        scores = self._pca.transform(centered)
        mat_idx = list(BOND_COLS).index(self.target_col)
        out = np.zeros(len(X))
        for i in range(len(X)):
            fac_h = self._ar.steps(scores[i], self.horizon)
            recon = fac_h @ self._pca.components_ + self._mean
            out[i] = recon[mat_idx]
        return out

    def diagnostics(self) -> dict:
        return {"model": "pca_curve", "target": self.target_col, "horizon": self.horizon}


@dataclass
class VARFactorsForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3, 4)
    ns_lambda: float = 0.07
    _p: int = field(default=2, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VARFactorsForecaster:
        endog = _endog_factors(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        self._p, _ = _grid_search_var(
            endog,
            y,
            self.horizon,
            self.val_frac,
            self.p_grid,
            _var_forecast,
            self._maturity(),
            self.ns_lambda,
        )
        self._hist = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        mat = self._maturity()
        for j, idx in enumerate(X.index):
            ex = _endog_factors(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            fac_h = _var_forecast(hist, self._p, self.horizon)
            if fac_h is None:
                out[j] = float(X.loc[idx, self.target_col])
            else:
                out[j] = _factor_level_pred(fac_h, mat, self.ns_lambda)
        return out

    def diagnostics(self) -> dict:
        return {"model": "var", "p": self._p, "target": self.target_col, "horizon": self.horizon}


@dataclass
class BVARFactorsForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3)
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    ns_lambda: float = 0.07
    _p: int = field(default=2, init=False)
    _alpha: float = field(default=1.0, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BVARFactorsForecaster:
        endog = _endog_factors(X).dropna()
        common = endog.index.intersection(y.dropna().index)
        endog, y = endog.loc[common], y.loc[common]
        best = (float("inf"), self.p_grid[0], self.alpha_grid[0])
        mat = self._maturity()
        for alpha in self.alpha_grid:
            for p in self.p_grid:
                fn = lambda e, pp, h, a=alpha: _ridge_var_forecast(e, pp, h, a)
                mse = 0.0
                n = len(endog)
                cut = max(30, int(n * (1.0 - self.val_frac)))
                val_idx = endog.index[cut:]
                errs: list[float] = []
                for idx in val_idx:
                    hist = endog.loc[endog.index <= idx]
                    fac_h = fn(hist, p, self.horizon)
                    if fac_h is None:
                        continue
                    pred = _factor_level_pred(fac_h, mat, self.ns_lambda)
                    yt = y.loc[idx]
                    if np.isfinite(yt):
                        errs.append((pred - yt) ** 2)
                if errs:
                    m = float(np.mean(errs))
                    if m < best[0]:
                        best = (m, p, alpha)
        self._p, self._alpha = best[1], best[2]
        self._hist = endog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        mat = self._maturity()
        for j, idx in enumerate(X.index):
            ex = _endog_factors(X.loc[[idx]])
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            fac_h = _ridge_var_forecast(hist, self._p, self.horizon, self._alpha)
            if fac_h is None:
                out[j] = float(X.loc[idx, self.target_col])
            else:
                out[j] = _factor_level_pred(fac_h, mat, self.ns_lambda)
        return out

    def diagnostics(self) -> dict:
        return {
            "model": "bvar",
            "p": self._p,
            "alpha": self._alpha,
            "target": self.target_col,
            "horizon": self.horizon,
        }


@dataclass
class VARXFactorsForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    p_grid: tuple[int, ...] = (1, 2, 3)
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    ns_lambda: float = 0.07
    _p: int = field(default=2, init=False)
    _alpha: float = field(default=1.0, init=False)
    _hist_e: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    _hist_x: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def _exog(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in EXOG_COLS if c in X.columns]
        if not cols:
            return pd.DataFrame({"macro0": 0.0}, index=X.index)
        return X.loc[:, cols].astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VARXFactorsForecaster:
        endog = _endog_factors(X).dropna()
        exog = self._exog(X).reindex(endog.index).dropna(how="all")
        common = endog.index.intersection(y.dropna().index).intersection(exog.index)
        endog, exog, y = endog.loc[common], exog.loc[common], y.loc[common]
        best = (float("inf"), self.p_grid[0], self.alpha_grid[0])
        mat = self._maturity()
        for alpha in self.alpha_grid:
            for p in self.p_grid:
                errs: list[float] = []
                n = len(endog)
                cut = max(30, int(n * (1.0 - self.val_frac)))
                for idx in endog.index[cut:]:
                    hist_e = endog.loc[endog.index <= idx]
                    hist_x = exog.loc[exog.index <= idx]
                    fac_h = _varx_forecast(hist_e, hist_x, p, self.horizon, alpha)
                    if fac_h is None:
                        continue
                    pred = _factor_level_pred(fac_h, mat, self.ns_lambda)
                    yt = y.loc[idx]
                    if np.isfinite(yt):
                        errs.append((pred - yt) ** 2)
                if errs:
                    m = float(np.mean(errs))
                    if m < best[0]:
                        best = (m, p, alpha)
        self._p, self._alpha = best[1], best[2]
        self._hist_e = endog
        self._hist_x = exog
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), np.nan)
        mat = self._maturity()
        for j, idx in enumerate(X.index):
            ex_e = _endog_factors(X.loc[[idx]])
            ex_x = self._exog(X.loc[[idx]])
            hist_e = pd.concat([self._hist_e, ex_e]).drop_duplicates()
            hist_x = pd.concat([self._hist_x, ex_x]).drop_duplicates()
            fac_h = _varx_forecast(hist_e, hist_x, self._p, self.horizon, self._alpha)
            if fac_h is None:
                out[j] = float(X.loc[idx, self.target_col])
            else:
                out[j] = _factor_level_pred(fac_h, mat, self.ns_lambda)
        return out

    def diagnostics(self) -> dict:
        return {"model": "varx", "p": self._p, "alpha": self._alpha, "target": self.target_col}


@dataclass
class MacroFinanceForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)
    _spot_col: str = field(default="", init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MacroFinanceForecaster:
        self._spot_col = self.target_col
        candidates = [
            self.target_col,
            f"{self.target_col}_l1",
            "slope",
            "slope_l1",
            "curvature",
            "ns_f1",
            "ns_f2",
            "ns_f3",
            "nbp_rate",
            "nbp_gap",
            "gus_cpi_yoy",
            "infl_gap",
        ]
        cols = [c for c in candidates if c in X.columns]
        n = len(X)
        cut = max(30, int(n * (1.0 - self.val_frac)))
        if cut >= n - 10:
            cut = n // 2
        best_mse, best_alpha = float("inf"), self.alpha_grid[0]
        for alpha in self.alpha_grid:
            m = Ridge(alpha=alpha)
            m.fit(X.iloc[:cut][cols].fillna(0.0), y.iloc[:cut])
            pred = m.predict(X.iloc[cut:][cols].fillna(0.0))
            mse = _level_mse(y.iloc[cut:], pred)
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self._alpha = best_alpha
        self._cols = cols
        self._model = Ridge(alpha=best_alpha)
        self._model.fit(X[cols].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        pred = self._model.predict(X[self._cols].fillna(0.0))
        spot = X[self._spot_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "macro_finance", "alpha": self._alpha, "n_features": len(self._cols)}


@dataclass
class AFNSForecaster:
    """Nelson–Siegel factors with VAR dynamics (joint curve, TRD-35 AFNS-lite)."""

    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    lambda_grid: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10)
    p_grid: tuple[int, ...] = (1, 2, 3)
    _lam: float = field(default=0.07, init=False)
    _p: int = field(default=2, init=False)
    _hist: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AFNSForecaster:
        Y = _yield_matrix(X)
        n = len(Y)
        tr, va = _split_train_val_xy(n, self.val_frac)
        mat = self._maturity()
        best = (float("inf"), self.lambda_grid[0], self.p_grid[0])
        for lam in self.lambda_grid:
            L = _ns_loadings(MATURITIES, lam)
            factors = _extract_factors(Y, L)
            endog = pd.DataFrame(factors, index=X.index, columns=list(FACTOR_COLS))
            for p in self.p_grid:
                preds = []
                for i in range(va.start, va.stop):
                    hist = endog.iloc[: i + 1]
                    fac_h = _var_forecast(hist, p, self.horizon)
                    if fac_h is None:
                        preds.append(np.nan)
                    else:
                        preds.append(_factor_level_pred(fac_h, mat, lam))
                mse = _level_mse(y.iloc[va], np.asarray(preds))
                if mse < best[0]:
                    best = (mse, lam, p)
        self._lam, self._p = best[1], best[2]
        L = _ns_loadings(MATURITIES, self._lam)
        factors = _extract_factors(Y, L)
        self._hist = pd.DataFrame(factors, index=X.index, columns=list(FACTOR_COLS))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Y = _yield_matrix(X)
        L = _ns_loadings(MATURITIES, self._lam)
        mat = self._maturity()
        out = np.full(len(X), np.nan)
        for j, idx in enumerate(X.index):
            row_factors = _extract_factors(Y[j : j + 1], L)[0]
            ex = pd.DataFrame([row_factors], index=[idx], columns=list(FACTOR_COLS))
            hist = pd.concat([self._hist, ex]).drop_duplicates()
            fac_h = _var_forecast(hist, self._p, self.horizon)
            if fac_h is None:
                out[j] = float(X.loc[idx, self.target_col])
            else:
                out[j] = _factor_level_pred(fac_h, mat, self._lam)
        return out

    def diagnostics(self) -> dict:
        return {"model": "afns", "lambda": self._lam, "p": self._p, "target": self.target_col}


@dataclass
class PCALSTMForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    seed: int = 42
    window_grid: tuple[int, ...] = (10, 21)
    hidden_grid: tuple[int, ...] = (8, 16)
    lr_grid: tuple[float, ...] = (1e-3,)
    max_epochs: int = 25
    patience: int = 6
    _pca: PCA | None = field(default=None, init=False)
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(4), init=False)
    _best: _SeqLSTM | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PCALSTMForecaster:
        Y = _yield_matrix(X)
        n = len(Y)
        tr, va = _split_train_val_xy(n, self.val_frac)
        self._mean = Y[tr].mean(axis=0)
        centered = Y - self._mean
        self._pca = PCA(n_components=3)
        scores = self._pca.fit_transform(centered)
        score_df = pd.DataFrame(scores, index=X.index, columns=["pc1", "pc2", "pc3"])
        m = y.notna() & score_df.notna().all(axis=1)
        Xf, yf = score_df.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val_xy(len(Xf), self.val_frac)
        X_tr = Xf.iloc[tr_sl].to_numpy(dtype=float)
        y_tr = yf.iloc[tr_sl].to_numpy(dtype=float)
        X_va = Xf.iloc[va_sl]
        y_va = yf.iloc[va_sl]
        best_score = float("inf")
        best_cfg: _SeqLSTM | None = None
        for window in self.window_grid:
            for hidden in self.hidden_grid:
                for lr in self.lr_grid:
                    try:
                        lstm = _SeqLSTM(
                            window=int(window),
                            hidden=int(hidden),
                            layers=1,
                            lr=float(lr),
                            max_epochs=self.max_epochs,
                            patience=self.patience,
                            seed=self.seed,
                        )
                        lstm.fit_arrays(X_tr, y_tr)
                        pred = lstm.predict_arrays(X_va.to_numpy(dtype=float))
                        score = _val_mse(y_va, pred)
                        if score < best_score:
                            best_score = score
                            best_cfg = lstm
                    except (ValueError, RuntimeError):
                        continue
        if best_cfg is None:
            best_cfg = _SeqLSTM(
                window=int(self.window_grid[0]),
                hidden=int(self.hidden_grid[0]),
                layers=1,
                lr=float(self.lr_grid[0]),
                max_epochs=self.max_epochs,
                patience=self.patience,
                seed=self.seed,
            )
            best_cfg.fit_arrays(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))
        else:
            best_cfg.fit_arrays(Xf.to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._best = best_cfg
        return self

    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pca is not None
        Y = _yield_matrix(X)
        return self._pca.transform(Y - self._mean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._best is not None
        scores = self._scores(X)
        pred = self._best.predict_arrays(scores)
        spot = X[self.target_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "pca_lstm", "target": self.target_col, "horizon": self.horizon}


def model_factory(name: str, cfg: dict, *, horizon: int, target_col: str):
    val_frac = float(cfg.get("val_frac", 0.2))
    seed = int(cfg.get("seed", 42))
    var_cfg = cfg.get("var", {})
    p_grid = tuple(var_cfg.get("p", (1, 2, 3, 4)))
    ns_lam = float(cfg.get("ns_lambda", 0.07))
    if name == "pca_curve":
        return lambda: PCACurveForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            n_components=int(cfg.get("pca", {}).get("n_components", 3)),
        )
    if name == "var":
        return lambda: VARFactorsForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            p_grid=p_grid,
            ns_lambda=ns_lam,
        )
    if name == "bvar":
        return lambda: BVARFactorsForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            p_grid=tuple(var_cfg.get("p_bvar", (1, 2, 3))),
            alpha_grid=tuple(var_cfg.get("alpha", (0.1, 1.0, 10.0, 100.0))),
            ns_lambda=ns_lam,
        )
    if name == "varx":
        return lambda: VARXFactorsForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            p_grid=tuple(var_cfg.get("p_varx", (1, 2, 3))),
            alpha_grid=tuple(var_cfg.get("alpha", (0.1, 1.0, 10.0, 100.0))),
            ns_lambda=ns_lam,
        )
    if name == "macro_finance":
        return lambda: MacroFinanceForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "afns":
        af = cfg.get("afns", {})
        return lambda: AFNSForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            lambda_grid=tuple(af.get("lambda", (0.04, 0.06, 0.08, 0.10))),
            p_grid=tuple(af.get("p", (1, 2, 3))),
        )
    if name == "pca_lstm":
        pl = dict(cfg.get("pca_lstm", {}))
        pl["val_frac"] = val_frac
        return lambda: PCALSTMForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            seed=seed,
            window_grid=tuple(pl.get("window_grid", (10, 21))),
            hidden_grid=tuple(pl.get("hidden_grid", (8, 16))),
            lr_grid=tuple(pl.get("lr_grid", (1e-3,))),
            max_epochs=int(pl.get("max_epochs", 25)),
            patience=int(pl.get("patience", 6)),
        )
    if name == "lstm":
        lc = dict(cfg.get("lstm", {}))
        lc["val_frac"] = val_frac
        return lambda: LSTMLevelForecaster(lc, seed, target_col)
    raise KeyError(f"unknown tier-II bond model {name!r}")
