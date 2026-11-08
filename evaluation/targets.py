"""Target definitions and benchmark forecasts for multi-series walk-forward evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from experiments.data import forward_log_return

TargetKind = Literal["log_return", "level"]
HorizonUnit = Literal["day", "month"]


@dataclass(frozen=True)
class TargetSpec:
    """What the model is asked to predict and how horizons are measured."""

    kind: TargetKind = "log_return"
    price_col: str = "log_price"
    horizon: int = 1
    unit: HorizonUnit = "day"
    naive_window: int = 30

    @classmethod
    def btc_log_return(cls, h: int) -> TargetSpec:
        return cls(kind="log_return", price_col="log_price", horizon=h, unit="day")

    @property
    def validity_col(self) -> str:
        return "log_price" if self.price_col == "log_price" else self.price_col

    def purge_rows(self, index: pd.DatetimeIndex) -> int:
        """Rows to drop at the end of train (label overlap with test block)."""
        if self.unit == "day":
            return self.horizon
        return max_forward_steps(index, self.horizon)

    def last_valid_origin(self, index: pd.DatetimeIndex) -> int:
        """Exclusive upper bound on supervised row indices (last origin + 1)."""
        if self.unit == "day":
            return len(index) - self.horizon
        off = forward_offsets(index, self.horizon)
        ok = np.where(np.isfinite(off))[0]
        return int(ok[-1] + 1) if len(ok) else 0


def max_forward_steps(index: pd.DatetimeIndex, h_months: int) -> int:
    off = forward_offsets(index, h_months)
    finite = off[np.isfinite(off)]
    return int(finite.max()) if len(finite) else h_months * 30


def forward_offsets(index: pd.DatetimeIndex, h_months: int) -> np.ndarray:
    """For each row i, number of rows until the first date >= t_i + h months (NaN if none)."""
    n = len(index)
    out = np.full(n, np.nan)
    for i in range(n):
        tgt = index[i] + pd.DateOffset(months=h_months)
        j = index.searchsorted(tgt, side="left")
        if j < n:
            out[i] = j - i
    return out


def forward_target(price: pd.Series, spec: TargetSpec) -> pd.Series:
    """Supervised label aligned on the forecast origin index."""
    if spec.kind == "log_return":
        lp = price if spec.price_col == "log_price" else np.log(price)
        if spec.unit == "day":
            return forward_log_return(lp, spec.horizon)
        off = forward_offsets(price.index, spec.horizon)
        y = pd.Series(np.nan, index=price.index, dtype=float)
        for i, steps in enumerate(off):
            if np.isfinite(steps):
                j = i + int(steps)
                y.iloc[i] = lp.iloc[j] - lp.iloc[i]
        return y.rename(f"logret_fwd_{spec.horizon}m")
    # level
    if spec.unit == "day":
        return price.shift(-spec.horizon).rename(f"level_fwd_{spec.horizon}d")
    off = forward_offsets(price.index, spec.horizon)
    y = pd.Series(np.nan, index=price.index, dtype=float)
    for i, steps in enumerate(off):
        if np.isfinite(steps):
            y.iloc[i] = price.iloc[i + int(steps)]
    return y.rename(f"level_fwd_{spec.horizon}m")


def resolve_price(panel: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    if spec.price_col == "log_price":
        if "log_price" in panel.columns:
            return panel["log_price"]
        return np.log(panel["PriceUSD"])
    return panel[spec.price_col]


def benchmark_rw_values(index: pd.DatetimeIndex, price_at_origin: pd.Series, spec: TargetSpec) -> pd.Series:
    if spec.kind == "log_return":
        return pd.Series(0.0, index=index)
    return price_at_origin.reindex(index)


def benchmark_naive(
    price: pd.Series, spec: TargetSpec, y_true: pd.Series | None = None
) -> pd.Series:
    """Trailing-mean log-return (default naive for BTC baselines) or lagged level."""
    if spec.kind == "log_return":
        r = price.diff() if spec.price_col == "log_price" else np.log(price).diff()
        mu = r.rolling(spec.naive_window).mean()
        if spec.unit == "day":
            return (mu * spec.horizon).reindex(y_true.index if y_true is not None else price.index)
        off = forward_offsets(price.index, spec.horizon)
        idx = y_true.index if y_true is not None else price.index
        out = pd.Series(np.nan, index=idx, dtype=float)
        for t in idx:
            pos = price.index.get_loc(t)
            if isinstance(pos, slice):
                pos = pos.start
            w = slice(max(0, pos - spec.naive_window + 1), pos + 1)
            out.loc[t] = float(r.iloc[w].mean() * off[pos]) if np.isfinite(off[pos]) else np.nan
        return out
    lag = spec.naive_window if spec.unit == "day" else spec.horizon
    return price.shift(lag).reindex(y_true.index if y_true is not None else price.index)


def point_scores(
    y_true: np.ndarray, y_pred: np.ndarray, spec: TargetSpec, price_origin: np.ndarray | None = None
) -> dict[str, float]:
    """Point metrics plus OOS R² vs random walk and vs naive when benchmarks are supplied."""
    from evaluation import metrics as M

    y, yhat = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if spec.kind == "log_return":
        out = M.point_metrics(y, yhat)
        rw = np.zeros_like(y)
        out["r2_oos_vs_naive"] = float("nan")
        return out
    # level: RMSE/MAE on levels; MAPE vs true level; RW = carry price at t
    out = {
        "rmse": M.rmse(y, yhat),
        "mae": M.mae(y, yhat),
        "mape_price_pct": float(100.0 * np.mean(np.abs(yhat - y) / np.maximum(np.abs(y), 1e-12))),
        "r2_oos_vs_rw": float("nan"),
        "dir_acc": float("nan"),
    }
    if price_origin is not None:
        po = np.asarray(price_origin, dtype=float)[m]
        rw_hat = po
        sse_m = np.sum((y - yhat) ** 2)
        sse_rw = np.sum((y - rw_hat) ** 2)
        out["r2_oos_vs_rw"] = float(1.0 - sse_m / sse_rw) if sse_rw > 0 else float("nan")
    return out
