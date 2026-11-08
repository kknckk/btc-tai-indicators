"""Evaluation harness: walk-forward, metrics, backtest, MCS, calibration, Diebold-Mariano."""

from evaluation.backtest import (
    backtest_on_frame,
    buy_hold_position,
    positions_from_forecast,
    run_backtest,
    vol_target_position,
)
from evaluation.calibration import (
    calibrate_frame_cqr,
    calibrate_frame_point,
    coverage_width_curve,
    split_conformal_interval,
    split_conformal_on_frame,
)
from evaluation.mcs import dm_pairwise, mcs, rank_by_mean_loss
from evaluation.targets import TargetSpec, forward_target

__all__ = [
    "TargetSpec",
    "forward_target",
    "run_backtest",
    "buy_hold_position",
    "vol_target_position",
    "backtest_on_frame",
    "mcs",
    "dm_pairwise",
    "rank_by_mean_loss",
    "split_conformal_interval",
    "split_conformal_on_frame",
    "calibrate_frame_point",
    "calibrate_frame_cqr",
    "coverage_width_curve",
]
