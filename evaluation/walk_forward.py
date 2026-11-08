"""Expanding-window walk-forward harness with purge + embargo and a causality guard.

Fold geometry (rows of the supervised sample, position i = day t_i with a known price):
    train = rows [0, a - h - embargo)      # last train label ends at a-embargo-1  < a
    test  = rows [a, a + step) ∩ rows with a known h-day-ahead label
The purge of h rows removes train samples whose label window (t, t+h] overlaps the test block;
``embargo`` is an additional configurable gap. Scalers / selectors live inside the model's
``fit`` and therefore only ever see the train fold.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

import numpy as np
import pandas as pd

from evaluation import backtest as BT
from evaluation import metrics as M
from evaluation.targets import (
    TargetSpec,
    benchmark_naive,
    benchmark_rw_values,
    forward_target,
    point_scores,
    resolve_price,
)

logger = logging.getLogger(__name__)

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]
PositionFn = Callable[[pd.Series, Optional[pd.Series]], pd.Series]


class Forecaster(Protocol):
    """Model protocol. Optional: ``predict_vol(X) -> np.ndarray`` (mean daily variance over
    the next h days) and ``diagnostics() -> dict`` (stored per fold)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def sign_position(pred: pd.Series, vol_pred: Optional[pd.Series] = None) -> pd.Series:
    """Default long/short/flat rule: sign of the forecast return."""
    return np.sign(pred)


def check_causal(
    feature_fn: FeatureFn, panel: pd.DataFrame, n_probes: int = 5, seed: int = 0, tol: float = 1e-9
) -> list[str]:
    """Truncation-invariance test. Features computed on the panel truncated at day T must equal
    the features of the full panel for all rows <= T. Any future-peeking feature (shift(-k),
    centred windows, full-sample scaling) differs and is returned by name."""
    rng = np.random.default_rng(seed)
    full = feature_fn(panel)
    n = len(panel)
    # porównujemy wyłącznie kolumny numeryczne (odejmowanie); kolumny tekstowe
    # (np. surowe komunikaty RPP w tier2) nie uczestniczą w teście przyczynowości
    num_cols = [c for c in full.columns if pd.api.types.is_numeric_dtype(full[c])]
    full = full[num_cols]
    cuts = rng.integers(int(n * 0.3), n - 2, size=n_probes)
    offenders: set[str] = set()
    for cut in cuts:
        part = feature_fn(panel.iloc[: cut + 1])
        part = part[[c for c in part.columns if c in num_cols]]
        a, b = full.iloc[: cut + 1][part.columns], part
        diff = ((a - b).abs() > tol) & ~(a.isna() & b.isna())
        diff |= a.isna() ^ b.isna()
        offenders |= set(diff.columns[diff.any()])
    return sorted(offenders)


def _fold_starts(n: int, purge: int, min_train: int, embargo: int, step: int, last_origin: int | None = None) -> list[int]:
    last = last_origin if last_origin is not None else n - purge
    first = min_train + purge + embargo
    return list(range(first, last, step))


def make_folds(
    n: int,
    horizon: int,
    n_folds: int = 5,
    min_train_frac: float = 0.6,
    embargo: Optional[int] = None,
    step: Optional[int] = None,
) -> list[dict]:
    """Expanding-window folds as ``{"train": (start, end), "test": (start, end)}`` row ranges.

    Same purge/embargo rule as :func:`run_walk_forward`: the last ``horizon`` rows of a
    training fold are dropped (their labels overlap the test block) and ``embargo`` extra
    rows separate consecutive folds. Fold *k* trains on everything before its test block,
    so the window expands. Exposed separately because model runners that manage their own
    training loop (e.g. ``experiments/run_neural.py``) need the fold geometry only.
    """
    embargo = horizon if embargo is None else embargo
    min_train = int(n * min_train_frac)
    last = n - horizon
    if step is None:
        region = max(1, last - (min_train + horizon + embargo))
        step = max(horizon, region // max(1, n_folds))
    folds: list[dict] = []
    for a in _fold_starts(n, horizon, min_train, embargo, step, last_origin=n - horizon)[:n_folds]:
        tr_end = a - horizon - embargo
        te_end = min(a + step, last)
        if tr_end < 1 or te_end <= a:
            continue
        folds.append({"train": (0, tr_end), "test": (a, te_end)})
    return folds


def run_walk_forward(
    panel: pd.DataFrame,
    h: int,
    feature_fn: FeatureFn,
    model_factory: Callable[[], Forecaster],
    min_train: int = 1000,
    embargo: int = 0,
    step: int = 30,
    position_fn: PositionFn = sign_position,
    costs_bps: tuple[float, ...] = (10, 20),
    seed: int = 42,
    causal_check: bool = True,
    max_folds: Optional[int] = None,
    target: TargetSpec | None = None,
    series_id: str = "btc",
) -> dict:
    """Expanding walk-forward evaluation. Returns a dict with per-origin predictions (``frame``),
    ``folds`` (boundaries + model diagnostics) and ``metrics`` (point, vol, economic).

    Default target is BTC ``h``-day log return (``log_price``). Pass ``target`` for other series,
    level forecasts, or month horizons.
    """
    np.random.seed(seed)
    if causal_check:
        bad = check_causal(feature_fn, panel)
        if bad:
            raise ValueError(f"feature_fn is not causal (changes when future rows are removed): {bad[:10]}")

    spec = target or TargetSpec.btc_log_return(h)
    purge = spec.purge_rows(panel.index)
    idx = panel.index[panel[spec.validity_col].notna()]
    sub = panel.loc[idx]
    n = len(sub)
    X_all = feature_fn(panel).loc[idx]
    price = resolve_price(sub, spec)
    y_all = forward_target(price, spec)
    r = sub["logret_1"] if "logret_1" in sub.columns else price.diff()
    h_vol = h if spec.unit == "day" and spec.kind == "log_return" else max(h, purge)
    rv_fwd = (r ** 2).rolling(h_vol).mean().shift(-h_vol)
    next_ret = r.shift(-1)

    last_origin = spec.last_valid_origin(sub.index)
    starts = _fold_starts(n, purge, min_train, embargo, step, last_origin=last_origin)
    if max_folds:
        starts = starts[:max_folds]
    if not starts:
        raise ValueError("not enough data for a single fold")

    preds, vols, folds = [], [], []
    for k, a in enumerate(starts):
        tr_end = a - purge - embargo
        te_end = min(a + step, last_origin)
        tr, te = slice(0, tr_end), slice(a, te_end)
        X_tr, y_tr = X_all.iloc[tr], y_all.iloc[tr]
        assert y_tr.notna().all() and tr_end + purge - 1 < a, "label window of the train fold reaches the test block"
        assert X_all.index[tr_end - 1] < X_all.index[a]
        model = model_factory()
        model.fit(X_tr, y_tr)
        X_te = X_all.iloc[te]
        preds.append(pd.Series(np.asarray(model.predict(X_te), dtype=float), index=X_te.index))
        if hasattr(model, "predict_vol"):
            vols.append(pd.Series(np.asarray(model.predict_vol(X_te), dtype=float), index=X_te.index))
        info = {
            "fold": k, "train_start": str(X_tr.index[0].date()), "train_end": str(X_tr.index[-1].date()),
            "test_start": str(X_te.index[0].date()), "test_end": str(X_te.index[-1].date()),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        }
        if hasattr(model, "diagnostics"):
            info["diagnostics"] = model.diagnostics()
        folds.append(info)
        if k % 20 == 0:
            logger.info("h=%d fold %d/%d test %s..%s", h, k + 1, len(starts), info["test_start"], info["test_end"])

    pred = pd.concat(preds)
    vol_pred = pd.concat(vols) if vols else None
    px = sub["PriceUSD"] if "PriceUSD" in sub.columns else np.exp(price) if spec.price_col == "log_price" else price
    frame = pd.DataFrame({
        "y_true": y_all.loc[pred.index], "y_pred": pred, "rv_true": rv_fwd.loc[pred.index],
        "next_ret": next_ret.loc[pred.index], "price": px.loc[pred.index],
    })
    if vol_pred is not None:
        frame["var_pred"] = vol_pred
    frame = frame[frame["y_true"].notna()]

    res: dict = {
        "h": h,
        "series_id": series_id,
        "target": spec,
        "frame": frame,
        "folds": folds,
        "min_train": min_train,
        "embargo": embargo,
        "step": step,
    }
    res["metrics"] = summarize(frame, position_fn, costs_bps, target=spec, price=price)
    return res


def run_multi_series(
    panels: dict[str, pd.DataFrame],
    h: int,
    feature_fn: FeatureFn,
    model_factory: Callable[[], Forecaster],
    target: TargetSpec,
    **kwargs,
) -> dict[str, dict]:
    """Run the same walk-forward protocol on several aligned panels (e.g. multi-asset)."""
    return {
        sid: run_walk_forward(p, h, feature_fn, model_factory, target=target, series_id=sid, **kwargs)
        for sid, p in panels.items()
    }


def summarize(
    frame: pd.DataFrame,
    position_fn: PositionFn,
    costs_bps: tuple[float, ...] = (10, 20),
    since: Optional[str] = None,
    target: TargetSpec | None = None,
    price: pd.Series | None = None,
) -> dict:
    """Point, volatility and economic metrics on (optionally a sub-period of) a result frame."""
    f = frame.loc[since:] if since else frame
    out: dict = {"n_obs": int(len(f)), "start": str(f.index[0].date()), "end": str(f.index[-1].date())}
    if target is not None and target.kind == "level" and price is not None:
        po = price.reindex(f.index).to_numpy()
        out["point"] = point_scores(f["y_true"].to_numpy(), f["y_pred"].to_numpy(), target, po)
    else:
        out["point"] = M.point_metrics(f["y_true"].to_numpy(), f["y_pred"].to_numpy())
    if target is not None and price is not None:
        y = f["y_true"]
        rw = benchmark_rw_values(y.index, price, target)
        nv = benchmark_naive(price, target, y)
        m = y.notna() & f["y_pred"].notna() & rw.notna() & nv.notna()
        err_m = (y[m] - f["y_pred"][m]).to_numpy()
        err_rw = (y[m] - rw[m]).to_numpy()
        err_nv = (y[m] - nv[m]).to_numpy()
        out["vs_benchmarks"] = {
            "mse_model": float(np.mean(err_m ** 2)),
            "mse_rw": float(np.mean(err_rw ** 2)),
            "mse_naive": float(np.mean(err_nv ** 2)),
        }
    if "var_pred" in f:
        out["vol"] = M.vol_metrics(f["rv_true"].to_numpy(), f["var_pred"].to_numpy())
    pos = position_fn(f["y_pred"], f["var_pred"] if "var_pred" in f else None)
    out["economic"] = BT.run_backtest(pos, f["next_ret"], costs_bps)
    return out
