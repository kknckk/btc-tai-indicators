"""Tier-I WIG20 forecasters: GARCH, EGARCH, GJR-GARCH, event study (TRD-50)."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

VolKind = Literal["garch", "gjr", "egarch"]


def _split_train_val(n: int, val_frac: float) -> tuple[slice, slice]:
    cut = max(30, int(n * (1.0 - val_frac)))
    if cut >= n - 10:
        cut = n // 2
    return slice(0, cut), slice(cut, n)


def _level_mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    m = y_true.notna().to_numpy() & np.isfinite(y_pred)
    if not m.any():
        return float("inf")
    return float(np.mean((y_true.to_numpy()[m] - y_pred[m]) ** 2))


@dataclass
class ArchLevelForecaster:
    """GARCH / GJR / EGARCH mean forecast → h-day index level (returns ×100 in ``arch``)."""

    vol_kind: VolKind
    returns: pd.Series
    price_col: str
    horizon: int
    val_frac: float = 0.2
    mean_lags_grid: tuple[int, ...] = (0, 1)
    dist_grid: tuple[str, ...] = ("normal", "t")
    state: dict = field(default_factory=dict)
    _lags: int = field(default=1, init=False)
    _dist: str = field(default="t", init=False)
    _params: pd.Series | None = field(default=None, init=False)

    def _arch(self, series: pd.Series, lags: int, dist: str):
        scaled = series.astype(float) * 100.0
        mean = "Constant" if lags == 0 else "AR"
        lags_arg = None if lags == 0 else lags
        if self.vol_kind == "egarch":
            return arch_model(
                scaled,
                mean=mean,
                lags=lags_arg,
                vol="EGARCH",
                p=1,
                q=1,
                dist=dist,
                rescale=False,
            )
        o = 1 if self.vol_kind == "gjr" else 0
        return arch_model(
            scaled,
            mean=mean,
            lags=lags_arg,
            vol="GARCH",
            p=1,
            o=o,
            q=1,
            dist=dist,
            rescale=False,
        )

    def _fit_params(self, ret: pd.Series, lags: int, dist: str) -> pd.Series:
        tr = ret.dropna()
        if len(tr) < 40:
            raise ValueError("too few returns for GARCH fit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._arch(tr, lags, dist).fit(disp="off", options={"maxiter": 500})
        if res.convergence_flag != 0 and "params" in self.state:
            raise RuntimeError(f"no convergence (flag {res.convergence_flag})")
        return res.params

    def _level_at(
        self,
        price: float,
        ret_upto: pd.Series,
        params: pd.Series,
        lags: int,
        dist: str,
    ) -> float:
        scaled = ret_upto.astype(float).dropna() * 100.0
        if len(scaled) < max(10, lags + 5):
            return float(price)
        pos = len(scaled) - 1
        res = self._arch(scaled, lags, dist).fix(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = res.forecast(horizon=self.horizon, start=pos, reindex=False, method="analytic")
        mu_h = float(f.mean.iloc[-1].sum()) / 100.0
        return float(price * np.exp(mu_h))

    def _predict_slice(
        self,
        X: pd.DataFrame,
        params: pd.Series,
        lags: int,
        dist: str,
    ) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=float)
        for j, t in enumerate(X.index):
            hist = self.returns.loc[:t].dropna()
            price = float(X[self.price_col].loc[t])
            try:
                out[j] = self._level_at(price, hist, params, lags, dist)
            except Exception:  # noqa: BLE001
                out[j] = price
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ArchLevelForecaster":
        ret = self.returns.reindex(X.index)
        n = len(X)
        tr_sl, va_sl = _split_train_val(n, self.val_frac)
        X_va, y_va = X.iloc[va_sl], y.iloc[va_sl]
        best = (float("inf"), self.mean_lags_grid[0], self.dist_grid[0])
        for lags in self.mean_lags_grid:
            for dist in self.dist_grid:
                try:
                    params = self._fit_params(ret.iloc[tr_sl], lags, dist)
                    preds = self._predict_slice(X_va, params, lags, dist)
                    mse = _level_mse(y_va, preds)
                    if mse < best[0]:
                        best = (mse, lags, dist)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("arch val grid failed lags=%s dist=%s: %s", lags, dist, exc)
        self._lags, self._dist = best[1], best[2]
        try:
            self._params = self._fit_params(ret, self._lags, self._dist)
            self.state["params"] = self._params
        except Exception as exc:  # noqa: BLE001
            self.state["failures"] = self.state.get("failures", 0) + 1
            logger.warning("%s fit failed (%s); reusing previous params", self.vol_kind, exc)
            if "params" not in self.state:
                raise
            self._params = self.state["params"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        return self._predict_slice(X, self._params, self._lags, self._dist)

    def diagnostics(self) -> dict:
        return {
            "model": self.vol_kind,
            "mean_lags": self._lags,
            "dist": self._dist,
            "fit_failures": self.state.get("failures", 0),
        }


@dataclass
class EventStudyLevelForecaster:
    """Large-move days: train average h-day forward log return; otherwise random-walk level."""

    returns: pd.Series
    price_col: str
    horizon: int
    val_frac: float = 0.2
    quantile_grid: tuple[float, ...] = (0.85, 0.90, 0.95)
    _thresh: float = field(default=0.02, init=False)
    _mu_event: float = field(default=0.0, init=False)

    def _predict_with(
        self,
        X: pd.DataFrame,
        thresh: float,
        mu_event: float,
        ret_hist: pd.Series,
    ) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=float)
        for j, t in enumerate(X.index):
            price = float(X[self.price_col].loc[t])
            r_t = float(ret_hist.loc[t]) if t in ret_hist.index else np.nan
            if np.isfinite(r_t) and abs(r_t) >= thresh:
                out[j] = price * np.exp(mu_event)
            else:
                out[j] = price
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EventStudyLevelForecaster":
        ret = self.returns.reindex(X.index)
        price = X[self.price_col].astype(float)
        fwd_log = np.log(price.shift(-self.horizon) / price)
        n = len(X)
        tr_sl, va_sl = _split_train_val(n, self.val_frac)
        ret_tr = ret.iloc[tr_sl]
        abs_tr = ret_tr.abs().dropna()
        best = (float("inf"), self.quantile_grid[0], 0.0)
        for q in self.quantile_grid:
            if len(abs_tr) < 20:
                thresh = float(abs_tr.quantile(q)) if len(abs_tr) else 0.02
            else:
                thresh = float(abs_tr.quantile(q))
            events = ret_tr.index[ret_tr.abs() >= thresh]
            ev_fwd = fwd_log.reindex(events).dropna()
            mu = float(ev_fwd.mean()) if len(ev_fwd) else 0.0
            preds = self._predict_with(X.iloc[va_sl], thresh, mu, ret)
            mse = _level_mse(y.iloc[va_sl], preds)
            if mse < best[0]:
                best = (mse, q, mu)
        q_best = best[1]
        abs_full = ret.abs().dropna()
        self._thresh = float(abs_full.quantile(q_best)) if len(abs_full) else 0.02
        events = ret.index[ret.abs() >= self._thresh]
        ev_fwd = fwd_log.reindex(events).dropna()
        self._mu_event = float(ev_fwd.mean()) if len(ev_fwd) else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ret = self.returns.reindex(X.index.union(self.returns.index)).sort_index()
        return self._predict_with(X, self._thresh, self._mu_event, ret)

    def diagnostics(self) -> dict:
        return {
            "model": "event_study",
            "threshold_abs_ret": self._thresh,
            "mu_event_logret_h": self._mu_event,
        }


def model_factory(
    name: str,
    cfg: dict,
    *,
    horizon: int,
    price_col: str,
    returns: pd.Series,
    states: dict[str, dict],
) -> Callable[[], object]:
    val_frac = float(cfg.get("val_frac", 0.2))
    if name == "garch":
        g = dict(cfg.get("garch", {}))
        return lambda: ArchLevelForecaster(
            "garch",
            returns,
            price_col,
            horizon,
            val_frac=val_frac,
            mean_lags_grid=tuple(g.get("mean_lags_grid", (0, 1))),
            dist_grid=tuple(g.get("dist_grid", ("normal", "t"))),
            state=states.setdefault("garch", {}),
        )
    if name == "egarch":
        e = dict(cfg.get("egarch", {}))
        return lambda: ArchLevelForecaster(
            "egarch",
            returns,
            price_col,
            horizon,
            val_frac=val_frac,
            mean_lags_grid=tuple(e.get("mean_lags_grid", (0, 1))),
            dist_grid=tuple(e.get("dist_grid", ("normal", "t"))),
            state=states.setdefault("egarch", {}),
        )
    if name == "gjr_garch":
        j = dict(cfg.get("gjr_garch", {}))
        return lambda: ArchLevelForecaster(
            "gjr",
            returns,
            price_col,
            horizon,
            val_frac=val_frac,
            mean_lags_grid=tuple(j.get("mean_lags_grid", (0, 1))),
            dist_grid=tuple(j.get("dist_grid", ("normal", "t"))),
            state=states.setdefault("gjr_garch", {}),
        )
    if name == "event_study":
        es = dict(cfg.get("event_study", {}))
        return lambda: EventStudyLevelForecaster(
            returns,
            price_col,
            horizon,
            val_frac=val_frac,
            quantile_grid=tuple(es.get("quantile_grid", (0.85, 0.90, 0.95))),
        )
    raise KeyError(f"unknown tier-I WIG20 model {name!r}")
