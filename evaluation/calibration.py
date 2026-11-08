"""Split-conformal and CQR intervals on walk-forward OOS residuals.

Calibration scores use only observations strictly before each forecast origin
(expanding OOS history), so no future residuals enter the conformal quantile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalResult:
    lower: np.ndarray
    upper: np.ndarray
    q: float
    alpha: float
    n_cal: int


@dataclass
class RollingCalibrationSummary:
    alpha: float
    nominal_coverage: float
    empirical_coverage: float
    median_width: float
    median_width_rw: float
    width_ratio_vs_rw: float
    n_eval: int


def split_conformal_scores(y_cal: np.ndarray, yhat_cal: np.ndarray) -> np.ndarray:
    """Absolute residual nonconformity scores on the calibration fold."""
    y, yh = np.asarray(y_cal, dtype=float), np.asarray(yhat_cal, dtype=float)
    m = np.isfinite(y) & np.isfinite(yh)
    return np.abs(y[m] - yh[m])


def rw_scores(y_cal: np.ndarray) -> np.ndarray:
    """Nonconformity for random-walk (zero log-return forecast): |y|."""
    y = np.asarray(y_cal, dtype=float)
    return np.abs(y[np.isfinite(y)])


def cqr_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """CQR score max(q_lo - y, y - q_hi) (Romano et al.)."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return np.maximum(lo[m] - y[m], y[m] - hi[m])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample adjusted quantile: ceil((n+1)(1-alpha))/n."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def split_conformal_interval(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float = 0.10,
) -> ConformalResult:
    """Symmetric intervals around test forecasts using calibration residuals."""
    scores = split_conformal_scores(y_cal, yhat_cal)
    q = conformal_quantile(scores, alpha)
    yh = np.asarray(yhat_test, dtype=float)
    return ConformalResult(lower=yh - q, upper=yh + q, q=q, alpha=alpha, n_cal=len(scores))


def cqr_interval(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    alpha: float = 0.10,
) -> ConformalResult:
    """Conformalized quantile regression intervals on the test rows."""
    scores = cqr_scores(y_cal, q_lo_cal, q_hi_cal)
    q = conformal_quantile(scores, alpha)
    lo = np.asarray(q_lo_test, dtype=float) - q
    hi = np.asarray(q_hi_test, dtype=float) + q
    return ConformalResult(lower=lo, upper=hi, q=q, alpha=alpha, n_cal=len(scores))


def split_conformal_on_frame(
    frame: pd.DataFrame,
    cal_frac: float = 0.2,
    alpha: float = 0.10,
    y_col: str = "y_true",
    yhat_col: str = "y_pred",
) -> tuple[ConformalResult, pd.DataFrame]:
    """Time-ordered split: first ``cal_frac`` of rows for calibration, remainder for test."""
    f = frame[[y_col, yhat_col]].dropna()
    n_cal = max(1, int(len(f) * cal_frac))
    cal, test = f.iloc[:n_cal], f.iloc[n_cal:]
    res = split_conformal_interval(
        cal[y_col].to_numpy(), cal[yhat_col].to_numpy(), test[yhat_col].to_numpy(), alpha=alpha
    )
    out = test.copy()
    out["lo"] = res.lower
    out["hi"] = res.upper
    out["covered"] = (out[y_col] >= out["lo"]) & (out[y_col] <= out["hi"])
    return res, out


def empirical_coverage(covered: np.ndarray) -> float:
    c = np.asarray(covered, dtype=bool)
    return float(c.mean()) if len(c) else float("nan")


def median_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return float("nan")
    return float(np.median(hi[m] - lo[m]))


def rolling_split_conformal(
    y: np.ndarray,
    yhat: np.ndarray,
    alpha: float = 0.20,
    min_cal: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expanding calibration on past OOS rows only (indices 0 .. i-1 for test i)."""
    y = np.asarray(y, dtype=float)
    yh = np.asarray(yhat, dtype=float)
    n = len(y)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < min_cal:
            continue
        scores = split_conformal_scores(y[:i], yh[:i])
        q = conformal_quantile(scores, alpha)
        lower[i] = yh[i] - q
        upper[i] = yh[i] + q
        covered[i] = bool(lower[i] <= y[i] <= upper[i])
    return lower, upper, covered


def rolling_rw_widths(
    y: np.ndarray,
    alpha: float = 0.20,
    min_cal: int = 30,
    yhat_rw: np.ndarray | None = None,
) -> np.ndarray:
    """Symmetric RW interval widths using only past OOS absolute errors."""
    y = np.asarray(y, dtype=float)
    yrw = np.asarray(yhat_rw, dtype=float) if yhat_rw is not None else None
    n = len(y)
    widths = np.full(n, np.nan)
    for i in range(n):
        if i < min_cal:
            continue
        if yrw is not None:
            scores = np.abs(y[:i] - yrw[:i])
            scores = scores[np.isfinite(scores)]
        else:
            scores = rw_scores(y[:i])
        q = conformal_quantile(scores, alpha)
        widths[i] = 2.0 * q
    return widths


def rolling_cqr(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float = 0.20,
    min_cal: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CQR with expanding calibration on past OOS (q_lo, q_hi) and labels."""
    y = np.asarray(y, dtype=float)
    lo_in = np.asarray(q_lo, dtype=float)
    hi_in = np.asarray(q_hi, dtype=float)
    n = len(y)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < min_cal:
            continue
        scores = cqr_scores(y[:i], lo_in[:i], hi_in[:i])
        q = conformal_quantile(scores, alpha)
        lower[i] = lo_in[i] - q
        upper[i] = hi_in[i] + q
        covered[i] = bool(lower[i] <= y[i] <= upper[i])
    return lower, upper, covered


def summarize_rolling_intervals(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    covered: np.ndarray,
    alpha: float,
    rw_widths: np.ndarray | None = None,
) -> RollingCalibrationSummary:
    m = np.isfinite(lower) & np.isfinite(upper)
    med_w = median_interval_width(lower, upper)
    med_rw = float(np.nanmedian(rw_widths[m])) if rw_widths is not None else float("nan")
    ratio = med_w / med_rw if rw_widths is not None and np.isfinite(med_rw) and med_rw > 0 else float("nan")
    return RollingCalibrationSummary(
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        empirical_coverage=empirical_coverage(covered[m]),
        median_width=med_w,
        median_width_rw=med_rw,
        width_ratio_vs_rw=ratio,
        n_eval=int(m.sum()),
    )


def calibrate_frame_point(
    frame: pd.DataFrame,
    alpha: float = 0.20,
    min_cal: int = 30,
    y_col: str = "y_true",
    yhat_col: str = "y_pred",
) -> tuple[pd.DataFrame, RollingCalibrationSummary]:
    """Split-conformal on a walk-forward OOS frame (log-return or any scalar target)."""
    y = frame[y_col].to_numpy()
    yh = frame[yhat_col].to_numpy()
    lo, hi, cov = rolling_split_conformal(y, yh, alpha=alpha, min_cal=min_cal)
    rw_w = rolling_rw_widths(y, alpha=alpha, min_cal=min_cal)
    out = frame.copy()
    out["lo"], out["hi"], out["covered"] = lo, hi, cov
    summary = summarize_rolling_intervals(y, lo, hi, cov, alpha, rw_w)
    return out, summary


def calibrate_frame_cqr(
    frame: pd.DataFrame,
    alpha: float = 0.20,
    min_cal: int = 30,
    y_col: str = "y_true",
    q_lo_col: str = "q_lo",
    q_hi_col: str = "q_hi",
) -> tuple[pd.DataFrame, RollingCalibrationSummary]:
    """CQR on a frame with nominal quantile forecasts."""
    y = frame[y_col].to_numpy()
    qlo = frame[q_lo_col].to_numpy()
    qhi = frame[q_hi_col].to_numpy()
    lo, hi, cov = rolling_cqr(y, qlo, qhi, alpha=alpha, min_cal=min_cal)
    out = frame.copy()
    out["lo"], out["hi"], out["covered"] = lo, hi, cov
    yrw = frame["price"].to_numpy() if "price" in frame.columns else None
    rw_w = rolling_rw_widths(y, alpha=alpha, min_cal=min_cal, yhat_rw=yrw)
    summary = summarize_rolling_intervals(y, lo, hi, cov, alpha, rw_w)
    return out, summary


def raw_quantile_coverage(
    frame: pd.DataFrame,
    y_col: str = "y_true",
    q_lo_col: str = "q_lo",
    q_hi_col: str = "q_hi",
) -> float:
    y = frame[y_col]
    lo, hi = frame[q_lo_col], frame[q_hi_col]
    m = y.notna() & lo.notna() & hi.notna()
    if not m.any():
        return float("nan")
    return empirical_coverage(((y[m] >= lo[m]) & (y[m] <= hi[m])).to_numpy())


def coverage_width_curve(
    y: np.ndarray,
    yhat: np.ndarray,
    alphas: np.ndarray | None = None,
    min_cal: int = 30,
) -> pd.DataFrame:
    """Empirical coverage and median width vs nominal level (split-conformal, rolling)."""
    if alphas is None:
        alphas = np.linspace(0.02, 0.50, 25)
    rows: list[dict] = []
    y = np.asarray(y, dtype=float)
    yh = np.asarray(yhat, dtype=float)
    for alpha in alphas:
        lo, hi, cov = rolling_split_conformal(y, yh, alpha=float(alpha), min_cal=min_cal)
        m = np.isfinite(lo)
        rows.append(
            {
                "alpha": float(alpha),
                "nominal_coverage": 1.0 - float(alpha),
                "empirical_coverage": empirical_coverage(cov[m]),
                "median_width": median_interval_width(lo, hi),
            }
        )
    return pd.DataFrame(rows)


def coverage_width_curve_cqr(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alphas: np.ndarray | None = None,
    min_cal: int = 30,
) -> pd.DataFrame:
    if alphas is None:
        alphas = np.linspace(0.02, 0.50, 25)
    rows: list[dict] = []
    for alpha in alphas:
        lo, hi, cov = rolling_cqr(y, q_lo, q_hi, alpha=float(alpha), min_cal=min_cal)
        m = np.isfinite(lo)
        rows.append(
            {
                "alpha": float(alpha),
                "nominal_coverage": 1.0 - float(alpha),
                "empirical_coverage": empirical_coverage(cov[m]),
                "median_width": median_interval_width(lo, hi),
            }
        )
    return pd.DataFrame(rows)
