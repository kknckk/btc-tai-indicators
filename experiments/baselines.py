"""Baseline forecasters conforming to ``evaluation.walk_forward.Forecaster``.

Volatility models (naive, HAR-RV, GARCH) take the full return series by closure but only ever
read values at or before the timestamp they forecast from (HAR/naive use trailing windows,
GARCH filters recursively with parameters fitted on the train fold only).
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from arch import arch_model

logger = logging.getLogger(__name__)


class NaiveRW:
    """Random walk: zero return forecast. Variance forecast = trailing 30d mean r^2 (historical vol)."""

    def __init__(self, returns: pd.Series) -> None:
        self.hist_var = (returns ** 2).rolling(30, min_periods=30).mean()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveRW":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def predict_vol(self, X: pd.DataFrame) -> np.ndarray:
        return self.hist_var.reindex(X.index).to_numpy()


class HarRV:
    """HAR-RV (Corsi 2009) on daily realised variance proxy r^2 (no intraday data in repo).

    Regress mean r^2 over (t, t+h] on r^2_t, mean r^2 over 7d and 30d (OLS, levels).
    Return forecast = 0.
    """

    def __init__(self, returns: pd.Series, h: int) -> None:
        r2 = returns ** 2
        self.h = h
        self.r2 = r2
        self.feat = pd.DataFrame({
            "rv_d": r2, "rv_w": r2.rolling(7, min_periods=7).mean(), "rv_m": r2.rolling(30, min_periods=30).mean(),
        })
        self.res = None
        self.floor = 1e-8

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HarRV":
        h = self.h
        tgt = self.r2.rolling(h).mean().shift(-h).reindex(X.index)  # label window ends <= last train idx + h
        F = self.feat.reindex(X.index)
        m = tgt.notna() & F.notna().all(axis=1)
        self.res = sm.OLS(tgt[m], sm.add_constant(F[m])).fit()
        self.floor = float(tgt[m].quantile(0.01))
        return self

    def predict_vol(self, X: pd.DataFrame) -> np.ndarray:
        F = sm.add_constant(self.feat.reindex(X.index), has_constant="add")
        return np.maximum(self.res.predict(F).to_numpy(), self.floor)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def diagnostics(self) -> dict:
        return {"coef": {k: float(v) for k, v in self.res.params.items()}}


class GarchVol:
    """Zero-mean GARCH(1,1) (o=0) or GJR-GARCH(1,1,1) (o=1). Parameters fitted on the train fold
    (returns x100); variance forecasts for test origins are produced by filtering the fixed model
    recursively, so the forecast at origin t only uses returns <= t. Return forecast = 0.
    ``state`` is shared across folds to fall back on previous parameters if a fit fails."""

    def __init__(self, returns: pd.Series, h: int, state: dict, o: int = 1, dist: str = "t") -> None:
        self.ret = returns * 100.0
        self.h, self.o, self.dist, self.state = h, o, dist, state

    def _model(self, series: pd.Series):
        return arch_model(series, mean="Zero", vol="GARCH", p=1, o=self.o, q=1, dist=self.dist, rescale=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GarchVol":
        tr = self.ret.reindex(X.index).dropna()
        self.ok = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self._model(tr).fit(disp="off", options={"maxiter": 500})
            if res.convergence_flag != 0 and "params" in self.state:
                raise RuntimeError(f"no convergence (flag {res.convergence_flag})")
            self.state["params"] = res.params
        except Exception as exc:  # noqa: BLE001
            self.ok = False
            self.state["failures"] = self.state.get("failures", 0) + 1
            logger.warning("GARCH fit failed (%s); reusing previous params", exc)
            if "params" not in self.state:
                raise
        return self

    def predict_vol(self, X: pd.DataFrame) -> np.ndarray:
        upto = self.ret.loc[: X.index[-1]].dropna()
        pos0 = upto.index.get_indexer([X.index[0]])[0]
        res = self._model(upto).fix(self.state["params"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = res.forecast(horizon=self.h, start=pos0, reindex=False, method="analytic")
        v = f.variance.to_numpy().mean(axis=1) / 1e4  # mean daily variance over next h days
        return pd.Series(v, index=f.variance.index).reindex(X.index).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class XGBForecaster:
    """XGBoost regressor on h-day log return. Early stopping uses the LAST ``val_fraction`` of the
    train fold only (chronological), the test block is never touched."""

    def __init__(self, params: dict, seed: int = 42) -> None:
        self.p = dict(params)
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBForecaster":
        p = dict(self.p)
        vf, es = p.pop("val_fraction"), p.pop("early_stopping_rounds")
        cut = int(len(X) * (1 - vf))
        self.cols = list(X.columns)
        self.model = xgb.XGBRegressor(
            tree_method="hist", random_state=self.seed, early_stopping_rounds=es, **p
        )
        self.model.fit(X.iloc[:cut], y.iloc[:cut], eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.cols])

    def diagnostics(self) -> dict:
        gain = self.model.get_booster().get_score(importance_type="total_gain")
        tot = sum(gain.values()) or 1.0
        return {"best_iteration": int(self.model.best_iteration), "gain": {k: v / tot for k, v in gain.items()}}
