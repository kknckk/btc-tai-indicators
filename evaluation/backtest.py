"""Trading backtest: positions, transaction costs, Sharpe, max drawdown, turnover."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from evaluation import metrics as M

PositionFn = Callable[[pd.Series, Optional[pd.Series]], pd.Series]


def buy_hold_position(y_pred: pd.Series, vol_pred: Optional[pd.Series] = None) -> pd.Series:
    """Long-only benchmark (economic random-walk when the return forecast is zero)."""
    return pd.Series(1.0, index=y_pred.index)


def vol_target_position(
    annual_vol: float,
    max_leverage: float,
    ann: int = M.ANN,
) -> PositionFn:
    """Long-only position scaled to target annualised volatility from predicted daily variance."""

    def fn(y_pred: pd.Series, vol_pred: Optional[pd.Series] = None) -> pd.Series:
        if vol_pred is None:
            raise ValueError("vol-targeting requires var_pred / vol_pred on the frame")
        sigma_ann = np.sqrt(vol_pred.astype(float) * ann)
        return (annual_vol / sigma_ann).clip(0, max_leverage)

    return fn


def run_backtest(
    position: pd.Series,
    next_ret: pd.Series,
    costs_bps: tuple[float, ...] = (10, 20),
    ann: int = M.ANN,
) -> dict[str, dict[str, float]]:
    """Economic performance for a position series and next-day simple returns."""
    return M.economic_metrics(position, next_ret, costs_bps, ann)


def positions_from_forecast(
    y_pred: pd.Series,
    vol_pred: Optional[pd.Series] = None,
    rule: PositionFn | None = None,
) -> pd.Series:
    """Map forecasts to positions (default: sign of predicted log-return)."""
    if rule is None:
        from evaluation.walk_forward import sign_position

        rule = sign_position
    return rule(y_pred, vol_pred)


def backtest_on_frame(
    frame: pd.DataFrame,
    position_fn: PositionFn,
    costs_bps: tuple[float, ...] = (10, 20),
) -> dict[str, dict[str, float]]:
    """Run the economic backtest on a walk-forward result frame (``y_pred``, ``next_ret``, optional ``var_pred``)."""
    vol = frame["var_pred"] if "var_pred" in frame.columns else None
    pos = position_fn(frame["y_pred"], vol)
    return run_backtest(pos, frame["next_ret"], costs_bps)


def align_forecast_frames(frames: dict[str, pd.DataFrame]) -> tuple[pd.Index, dict[str, pd.DataFrame]]:
    """Intersect indices so every model is evaluated on the same OOS days."""
    idx: pd.Index | None = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    if idx is None or len(idx) == 0:
        raise ValueError("no overlapping forecast dates across models")
    idx = idx.sort_values()
    return idx, {k: f.loc[idx] for k, f in frames.items()}


def squared_return_losses(frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Per-day squared return forecast error on the common aligned sample (for DM / MCS)."""
    _, aligned = align_forecast_frames(frames)
    return {
        m: ((f["y_true"] - f["y_pred"]) ** 2).to_numpy(dtype=float)
        for m, f in aligned.items()
    }
