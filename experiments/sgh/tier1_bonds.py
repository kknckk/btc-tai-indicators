"""Tier-I PL bond yield forecasters: Nelson–Siegel, Svensson, Diebold–Li (TRD-34)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from experiments.sgh.bonds_panel import BOND_COLS, MATURITIES, MATURITY_YEARS


def _split_train_val(n: int, val_frac: float) -> tuple[slice, slice]:
    cut = max(30, int(n * (1.0 - val_frac)))
    if cut >= n - 10:
        cut = n // 2
    return slice(0, cut), slice(cut, n)


def _ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    lt = lam * np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.where(lt > 1e-8, (1.0 - np.exp(-lt)) / lt, 1.0)
        f3 = f2 - np.exp(-lt)
    return np.column_stack([np.ones_like(lt), f2, f3])


def _svensson_loadings(tau: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    base = _ns_loadings(tau, lam1)
    lt2 = lam2 * np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f4 = np.where(lt2 > 1e-8, (1.0 - np.exp(-lt2)) / lt2 - np.exp(-lt2), 0.0)
    return np.column_stack([base, f4])


def _yield_matrix(X: pd.DataFrame) -> np.ndarray:
    return X[list(BOND_COLS)].to_numpy(dtype=float)


def _extract_factors(yields: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    return yields @ np.linalg.pinv(loadings).T


def _level_mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    m = y_true.notna().to_numpy() & np.isfinite(y_pred)
    if not m.any():
        return float("inf")
    return float(np.mean((y_true.to_numpy()[m] - y_pred[m]) ** 2))


@dataclass
class _FactorAR:
    intercept: np.ndarray
    phi: np.ndarray

    @classmethod
    def fit(cls, factors: np.ndarray) -> _FactorAR:
        if len(factors) < 3:
            k = factors.shape[1]
            return cls(np.zeros(k), np.ones(k))
        x, y = factors[:-1], factors[1:]
        phi = np.zeros(factors.shape[1])
        intercept = np.zeros(factors.shape[1])
        for j in range(factors.shape[1]):
            denom = float(np.dot(x[:, j], x[:, j])) + 1e-12
            phi[j] = float(np.dot(x[:, j], y[:, j]) / denom)
            intercept[j] = float(np.mean(y[:, j] - phi[j] * x[:, j]))
        return cls(intercept, phi)

    def steps(self, start: np.ndarray, h: int) -> np.ndarray:
        state = start.copy()
        for _ in range(max(1, h)):
            state = self.intercept + self.phi * state
        return state


def _predict_curve_row(
    factors: np.ndarray,
    ar: _FactorAR,
    horizon: int,
    maturity: float,
    load_fn,
) -> float:
    fac_h = ar.steps(factors, horizon)
    row = load_fn(np.array([maturity]))[0]
    return float(fac_h @ row)


@dataclass
class NelsonSiegelForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    lambda_grid: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10, 0.12)
    _lam: float = field(default=0.07, init=False)
    _ar: _FactorAR = field(default_factory=lambda: _FactorAR(np.zeros(3), np.ones(3)), init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def _load_row(self, tau: np.ndarray) -> np.ndarray:
        return _ns_loadings(tau, self._lam)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NelsonSiegelForecaster:
        Y = _yield_matrix(X)
        n = len(Y)
        tr, va = _split_train_val(n, self.val_frac)
        mat = self._maturity()
        best = (float("inf"), self.lambda_grid[0])
        for lam in self.lambda_grid:
            L = _ns_loadings(MATURITIES, lam)
            factors = _extract_factors(Y, L)
            ar = _FactorAR.fit(factors[tr])
            load_fn = lambda tau, lam=lam: _ns_loadings(tau, lam)
            preds = np.array(
                [_predict_curve_row(factors[i], ar, self.horizon, mat, load_fn) for i in range(va.start, va.stop)]
            )
            mse = _level_mse(y.iloc[va], preds)
            if mse < best[0]:
                best = (mse, lam)
        self._lam = best[1]
        factors_all = _extract_factors(Y, _ns_loadings(MATURITIES, self._lam))
        self._ar = _FactorAR.fit(factors_all[tr])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Y = _yield_matrix(X)
        factors = _extract_factors(Y, _ns_loadings(MATURITIES, self._lam))
        mat = self._maturity()
        load_fn = lambda tau: _ns_loadings(tau, self._lam)
        return np.array([_predict_curve_row(factors[i], self._ar, self.horizon, mat, load_fn) for i in range(len(Y))])

    def diagnostics(self) -> dict:
        return {"lambda": self._lam, "target": self.target_col, "horizon": self.horizon}


@dataclass
class SvenssonForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    lambda1_grid: tuple[float, ...] = (0.05, 0.07, 0.09)
    lambda2_grid: tuple[float, ...] = (0.02, 0.04, 0.06)
    _lam1: float = field(default=0.07, init=False)
    _lam2: float = field(default=0.04, init=False)
    _ar: _FactorAR = field(default_factory=lambda: _FactorAR(np.zeros(4), np.ones(4)), init=False)

    def _maturity(self) -> float:
        return float(MATURITY_YEARS[self.target_col])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SvenssonForecaster:
        Y = _yield_matrix(X)
        n = len(Y)
        tr, va = _split_train_val(n, self.val_frac)
        mat = self._maturity()
        best = (float("inf"), self.lambda1_grid[0], self.lambda2_grid[0])
        for lam1 in self.lambda1_grid:
            for lam2 in self.lambda2_grid:
                L = _svensson_loadings(MATURITIES, lam1, lam2)
                factors = _extract_factors(Y, L)
                ar = _FactorAR.fit(factors[tr])
                load_fn = lambda tau, lam1=lam1, lam2=lam2: _svensson_loadings(tau, lam1, lam2)
                preds = np.array(
                    [_predict_curve_row(factors[i], ar, self.horizon, mat, load_fn) for i in range(va.start, va.stop)]
                )
                mse = _level_mse(y.iloc[va], preds)
                if mse < best[0]:
                    best = (mse, lam1, lam2)
        self._lam1, self._lam2 = best[1], best[2]
        factors_all = _extract_factors(Y, _svensson_loadings(MATURITIES, self._lam1, self._lam2))
        self._ar = _FactorAR.fit(factors_all[tr])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Y = _yield_matrix(X)
        factors = _extract_factors(Y, _svensson_loadings(MATURITIES, self._lam1, self._lam2))
        mat = self._maturity()
        load_fn = lambda tau: _svensson_loadings(tau, self._lam1, self._lam2)
        return np.array([_predict_curve_row(factors[i], self._ar, self.horizon, mat, load_fn) for i in range(len(Y))])

    def diagnostics(self) -> dict:
        return {
            "lambda1": self._lam1,
            "lambda2": self._lam2,
            "target": self.target_col,
            "horizon": self.horizon,
        }


@dataclass
class DieboldLiForecaster:
    horizon: int
    target_col: str = "bond_pl_10y"
    val_frac: float = 0.2
    lambda_grid: tuple[float, ...] = (0.0609, 0.07, 0.08)
    _inner: NelsonSiegelForecaster = field(init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DieboldLiForecaster:
        self._inner = NelsonSiegelForecaster(
            horizon=self.horizon,
            target_col=self.target_col,
            val_frac=self.val_frac,
            lambda_grid=self.lambda_grid,
        )
        self._inner.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._inner.predict(X)

    def diagnostics(self) -> dict:
        d = self._inner.diagnostics()
        d["model"] = "diebold_li"
        return d


def model_factory(name: str, cfg: dict, *, horizon: int, target_col: str):
    val_frac = float(cfg.get("val_frac", 0.2))
    ns_cfg = cfg.get("nelson_siegel", {})
    sv_cfg = cfg.get("svensson", {})
    dl_cfg = cfg.get("diebold_li", {})
    if name == "nelson_siegel":
        return lambda: NelsonSiegelForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            lambda_grid=tuple(ns_cfg.get("lambda", (0.04, 0.06, 0.08, 0.10, 0.12))),
        )
    if name == "svensson":
        return lambda: SvenssonForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            lambda1_grid=tuple(sv_cfg.get("lambda1", (0.05, 0.07, 0.09))),
            lambda2_grid=tuple(sv_cfg.get("lambda2", (0.02, 0.04, 0.06))),
        )
    if name == "diebold_li":
        return lambda: DieboldLiForecaster(
            horizon=horizon,
            target_col=target_col,
            val_frac=val_frac,
            lambda_grid=tuple(dl_cfg.get("lambda", (0.0609, 0.07, 0.08))),
        )
    raise KeyError(f"unknown tier-I bond model {name!r}")
