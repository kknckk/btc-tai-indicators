"""Tier-I gold futures forecasters: LSTM, GARCH, XGBoost (TRD-56)."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from experiments.sgh.tier1_wheat import LSTMLevelForecaster, XGBLevelForecaster
from experiments.sgh.tier1_wig20 import ArchLevelForecaster


def model_factory(
    name: str,
    cfg: dict,
    *,
    horizon: int,
    price_col: str,
    returns: pd.Series,
    states: dict[str, dict],
) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "lstm":
        lc = dict(cfg.get("lstm", {}))
        lc["val_frac"] = vf
        return lambda: LSTMLevelForecaster(lc, seed, price_col)
    if name == "garch":
        g = dict(cfg.get("garch", {}))
        return lambda: ArchLevelForecaster(
            "garch",
            returns,
            price_col,
            horizon,
            val_frac=vf,
            mean_lags_grid=tuple(g.get("mean_lags_grid", (0, 1))),
            dist_grid=tuple(g.get("dist_grid", ("normal", "t"))),
            state=states.setdefault("garch", {}),
        )
    if name == "xgboost":
        xc = dict(cfg.get("xgboost", {}))
        xc["val_frac"] = vf
        return lambda: XGBLevelForecaster(xc, seed, price_col)
    raise KeyError(f"unknown tier-I gold model {name!r}")
