"""Forecast and economic metrics (all arrays aligned on the forecast origin t)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 365


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mape_price(y: np.ndarray, yhat: np.ndarray) -> float:
    """MAPE (%) on the price level reconstructed from log-returns.

    P_{t+h} = P_t * exp(y), P^_{t+h} = P_t * exp(yhat)  =>  |P^ - P| / P = |exp(yhat - y) - 1|.
    """
    return float(100.0 * np.mean(np.abs(np.exp(np.asarray(yhat) - np.asarray(y)) - 1.0)))


def r2_oos_rw(y: np.ndarray, yhat: np.ndarray) -> float:
    """OOS R^2 against the random-walk (zero-return) benchmark: 1 - SSE_model / SSE_rw."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2))


def directional_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Share of correct signs among rows with a non-zero forecast (NaN if none)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    m = yhat != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.sign(y[m]) == np.sign(yhat[m])))


def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y, yhat), "mae": mae(y, yhat), "mape_price_pct": mape_price(y, yhat),
        "r2_oos_vs_rw": r2_oos_rw(y, yhat), "dir_acc": directional_accuracy(y, yhat),
    }


def qlike(rv: np.ndarray, var_hat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-observation QLIKE loss: rv/s - log(rv/s) - 1 (robust to noisy rv proxies)."""
    rv = np.maximum(np.asarray(rv, dtype=float), eps)
    s = np.maximum(np.asarray(var_hat, dtype=float), eps)
    return rv / s - np.log(rv / s) - 1.0


def vol_metrics(rv: np.ndarray, var_hat: np.ndarray) -> dict[str, float]:
    rv, var_hat = np.asarray(rv, dtype=float), np.asarray(var_hat, dtype=float)
    return {
        "qlike": float(np.mean(qlike(rv, var_hat))),
        "rmse_vol_daily": float(np.sqrt(np.mean((np.sqrt(rv) - np.sqrt(np.maximum(var_hat, 0))) ** 2))),
    }


def strategy_returns(position: pd.Series, next_ret: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """Daily simple strategy return: pos_t * (exp(r_{t+1}) - 1) - cost * |pos_t - pos_{t-1}|.

    ``position`` is decided at the close of day t and earns the NEXT day's return (no
    look-ahead). Trading cost is charged on every position change (first position from 0).
    """
    pos = position.astype(float)
    simple = np.expm1(next_ret.reindex(pos.index))
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    return pos * simple - (cost_bps / 1e4) * turnover


def sharpe(ret: pd.Series, ann: int = ANN) -> float:
    ret = ret.dropna()
    sd = ret.std(ddof=1)
    return float(np.sqrt(ann) * ret.mean() / sd) if sd > 0 else float("nan")


def max_drawdown(ret: pd.Series) -> float:
    eq = (1.0 + ret.dropna()).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def economic_metrics(
    position: pd.Series, next_ret: pd.Series, costs_bps: tuple[float, ...] = (10, 20), ann: int = ANN
) -> dict[str, dict[str, float]]:
    """Sharpe / MaxDD / turnover for gross (0 bps) and every cost level.

    turnover = mean |Δposition| per day, annualised (x365).
    """
    pos = position.dropna()
    nr = next_ret.reindex(pos.index)
    keep = nr.notna()
    pos, nr = pos[keep], nr[keep]
    out: dict[str, dict[str, float]] = {}
    turn = float(pos.diff().abs().fillna(pos.abs()).mean() * ann)
    for bps in (0, *costs_bps):
        r = strategy_returns(pos, nr, bps)
        r_fin = r.dropna()
        hit = float((r_fin > 0).mean()) if len(r_fin) else float("nan")
        out[f"{bps}bps"] = {
            "sharpe": sharpe(r, ann), "max_dd": max_drawdown(r),
            "ann_return": float(r.mean() * ann), "turnover_ann": turn,
            "hit_rate": hit,
        }
    return out
