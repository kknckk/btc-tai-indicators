"""Tier-I NBP reference-rate forecasters with in-fold validation (TRD-32)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from experiments.sgh.nbp_panel import PI_TARGET

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ImportError:  # pragma: no cover
    OrderedModel = None  # type: ignore[misc, assignment]


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


def _change_categories(y_level: pd.Series, rate_now: pd.Series) -> pd.Series:
    chg = y_level - rate_now
    cats = pd.Series(np.zeros(len(y_level), dtype=int), index=y_level.index)
    cats[chg <= -0.50] = 0
    cats[(chg > -0.50) & (chg < -0.01)] = 1
    cats[(chg >= -0.01) & (chg <= 0.01)] = 2
    cats[(chg > 0.01) & (chg < 0.50)] = 3
    cats[chg >= 0.50] = 4
    return cats


CHANGE_MIDPOINTS = np.array([-0.75, -0.25, 0.0, 0.25, 0.75])


@dataclass
class TaylorRuleForecaster:
    val_frac: float = 0.2
    phi_pi_grid: tuple[float, ...] = (0.5, 1.0, 1.5)
    phi_y_grid: tuple[float, ...] = (0.0, 0.25, 0.5)
    r_star_grid: tuple[float, ...] = (0.0, 0.5, 1.0)
    blend_grid: tuple[float, ...] = (0.0, 0.5, 1.0)
    _params: dict = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TaylorRuleForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        rate_v = Xv["rate_now"].fillna(Xv["rate_l1"])
        best = (float("inf"), {})
        for phi_pi in self.phi_pi_grid:
            for phi_y in self.phi_y_grid:
                for r_star in self.r_star_grid:
                    for blend in self.blend_grid:
                        i_star = (
                            r_star
                            + PI_TARGET
                            + phi_pi * Xv["infl_gap"]
                            + phi_y * (-Xv["pmi_gap"])
                        )
                        pred = blend * i_star + (1.0 - blend) * rate_v
                        mse = _level_mse(yv, pred.to_numpy())
                        if mse < best[0]:
                            best = (mse, {"phi_pi": phi_pi, "phi_y": phi_y, "r_star": r_star, "blend": blend})
        self._params = best[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self._params
        rate = X["rate_now"].fillna(X["rate_l1"]).ffill()
        i_star = (
            p["r_star"]
            + PI_TARGET
            + p["phi_pi"] * X["infl_gap"]
            + p["phi_y"] * (-X["pmi_gap"])
        )
        pred = p["blend"] * i_star + (1.0 - p["blend"]) * rate
        return pred.to_numpy()

    def diagnostics(self) -> dict:
        return {"params": dict(self._params)}


@dataclass
class ForwardLinearForecaster:
    val_frac: float = 0.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    _model: Ridge | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _alpha: float = field(default=1.0, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ForwardLinearForecaster:
        (Xf, yf), (Xv, yv) = _split_train_val(X, y, self.val_frac)
        cols = list(X.columns)
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
class OrderedForecaster:
    link: Literal["logit", "probit"]
    val_frac: float = 0.2
    _result: object | None = field(default=None, init=False)
    _cols: list[str] = field(default_factory=list, init=False)
    _fallback: ForwardLinearForecaster | None = field(default=None, init=False)
    _mids: np.ndarray = field(default_factory=lambda: CHANGE_MIDPOINTS.copy(), init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> OrderedForecaster:
        # statsmodels OrderedModel odrzuca modele z (implikowaną) stałą — np. para
        # kolumn różniących się stałą (infl_gap = cpi_yoy - 2.5). Bez tego filtra
        # każdy fold cicho spadał do fallback Ridge i „ordered logit/probit” w wynikach
        # wcale nie był estymowany (identyczne metryki jak forward_linear).
        keep: list[str] = []
        for c in X.columns:
            dup = any(
                float((X[c] - X[k]).std()) < 1e-12 for k in keep
            )
            if not dup:
                keep.append(c)
        self._cols = keep
        rate_now = X["rate_now"].reindex(y.index).fillna(X["rate_l1"])
        (Xf, yf), _ = _split_train_val(X, y, self.val_frac)
        cats_f = _change_categories(yf, Xf["rate_now"].fillna(Xf["rate_l1"]))

        if OrderedModel is None or cats_f.nunique() < 3:
            self._fallback = ForwardLinearForecaster(val_frac=self.val_frac)
            self._fallback.fit(X, y)
            self._result = None
            return self

        try:
            mod_val = OrderedModel(cats_f.astype(int), Xf[self._cols].fillna(0.0), distr=self.link)
            mod_val.fit(method="bfgs", disp=False, maxiter=200)
            cats_all = _change_categories(y, X["rate_now"].fillna(X["rate_l1"]))
            active = np.sort(pd.unique(cats_all.astype(int)))
            mod = OrderedModel(cats_all.astype(int), X[self._cols].fillna(0.0), distr=self.link)
            self._result = mod.fit(method="bfgs", disp=False, maxiter=200)
            # statsmodels numeruje kategorie 0..K-1 w kolejności obecności — midpointy
            # trzeba brać wyłącznie dla kategorii aktywnych, inaczej probs @ MIDPOINTS
            # wywala się na wymiarze, gdy w foldzie nie ma skrajnych zmian stopy.
            self._mids = CHANGE_MIDPOINTS[active]
            self._fallback = None
        except Exception:
            self._fallback = ForwardLinearForecaster(val_frac=self.val_frac)
            self._fallback.fit(X, y)
            self._result = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(X)
        assert self._result is not None
        probs = self._result.predict(exog=X[self._cols].fillna(0.0))
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = probs.reshape(-1, len(self._mids))
        exp_chg = probs @ self._mids
        rate_now = X["rate_now"].fillna(X["rate_l1"]).ffill()
        return (rate_now + exp_chg).to_numpy()

    def diagnostics(self) -> dict:
        if self._fallback is not None:
            return {"link": self.link, "fallback": "ridge", **self._fallback.diagnostics()}
        return {"link": self.link, "fallback": None}


def model_factory(name: str, cfg: dict):
    val_frac = float(cfg.get("val_frac", 0.2))
    if name == "taylor_rule":
        return lambda: TaylorRuleForecaster(
            val_frac=val_frac,
            phi_pi_grid=tuple(cfg.get("taylor", {}).get("phi_pi", (0.5, 1.0, 1.5))),
            phi_y_grid=tuple(cfg.get("taylor", {}).get("phi_y", (0.0, 0.25, 0.5))),
        )
    if name == "forward_linear":
        return lambda: ForwardLinearForecaster(
            val_frac=val_frac,
            alpha_grid=tuple(cfg.get("ridge", {}).get("alpha", (0.01, 0.1, 1.0, 10.0, 100.0))),
        )
    if name == "ordered_logit":
        return lambda: OrderedForecaster(link="logit", val_frac=val_frac)
    if name == "ordered_probit":
        return lambda: OrderedForecaster(link="probit", val_frac=val_frac)
    raise KeyError(f"unknown tier-I model {name!r}")
