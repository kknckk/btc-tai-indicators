"""Daily WIG20 index panel for tier-I vol / event models (TRD-50) and tier-II (TRD-51)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from experiments import multiasset as ma

logger = logging.getLogger(__name__)

PRICE_COL = "wig20"
EXTERNAL_DAILY = ("usdpln", "eurpln", "oil", "gold")


def _attach_external_daily(
    panel: pd.DataFrame,
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    for sid in EXTERNAL_DAILY:
        try:
            s = ma.load_series(sid, freq="D", refresh=refresh, use_cache_only=use_cache_only)
            panel[sid] = s.reindex(panel.index).ffill()
            panel[f"{sid}_logret_1"] = np.log(panel[sid]).diff()
        except Exception as exc:  # noqa: BLE001
            logger.debug("wig20 panel: skip external %s (%s)", sid, exc)
    return panel


def build_wig20_daily_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
    with_externals: bool = True,
) -> pd.DataFrame:
    s = ma.load_series(PRICE_COL, freq="D", refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.DataFrame({PRICE_COL: s}).sort_index()
    panel["logret_1"] = np.log(panel[PRICE_COL]).diff()
    panel = panel.dropna(subset=[PRICE_COL])
    if with_externals:
        panel = _attach_external_daily(panel, use_cache_only=use_cache_only, refresh=refresh)
    return panel


def wig20_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features: index level and trailing return / vol stats only."""
    p = panel[PRICE_COL]
    r = panel["logret_1"] if "logret_1" in panel.columns else np.log(p).diff()
    out = {
        PRICE_COL: p,
        f"{PRICE_COL}_l1": p.shift(1),
        f"{PRICE_COL}_l5": p.shift(5),
        f"{PRICE_COL}_l21": p.shift(21),
        f"{PRICE_COL}_ma5": p.rolling(5, min_periods=5).mean(),
        f"{PRICE_COL}_ma21": p.rolling(21, min_periods=21).mean(),
        f"{PRICE_COL}_vol21": r.rolling(21, min_periods=21).std(),
        f"{PRICE_COL}_absret1": r.abs(),
    }
    return pd.DataFrame(out, index=panel.index)


def wig20_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal tier-II features: tier-I tabular, HAR/RV, externals, sentiment proxies."""
    base = wig20_features(panel)
    r = panel["logret_1"].reindex(base.index)
    r2 = r ** 2
    base["rv_d"] = r2
    base["rv_w"] = r2.rolling(7, min_periods=7).mean()
    base["rv_m"] = r2.rolling(30, min_periods=30).mean()
    base["rv_sum_5"] = r2.rolling(5, min_periods=5).sum()
    base["rv_sum_22"] = r2.rolling(22, min_periods=22).sum()
    abs_r = r.abs()
    base["bipower_5"] = (abs_r * abs_r.shift(1)).rolling(5, min_periods=5).mean()
    mu21 = r.rolling(21, min_periods=21).mean()
    sd21 = r.rolling(21, min_periods=21).std().replace(0, np.nan)
    base["ret_z21"] = (r - mu21) / sd21
    base["sentiment_fear"] = (-mu21 / sd21).fillna(0.0)
    vol21 = r.rolling(21, min_periods=21).std()
    base["sentiment_stress"] = (vol21 / vol21.rolling(63, min_periods=21).median()).fillna(1.0)
    base["sentiment_momentum"] = mu21.fillna(0.0)
    for sid in EXTERNAL_DAILY:
        lr = panel.get(f"{sid}_logret_1")
        if lr is None and sid in panel.columns:
            lr = np.log(panel[sid]).diff()
        if lr is not None:
            lr = lr.reindex(base.index)
            base[f"{sid}_lr_l1"] = lr.shift(1)
            base[f"{sid}_lr_l5"] = lr.rolling(5, min_periods=5).mean().shift(1)
    return base


def demo_wig20_panel(n_days: int = 900, seed: int = 50) -> pd.DataFrame:
    """Deterministic synthetic WIG20 — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    logp = np.zeros(n_days)
    logp[0] = np.log(2200.0)
    for t in range(1, n_days):
        logp[t] = logp[t - 1] + rng.normal(0, 0.008)
    price = np.exp(logp)
    panel = pd.DataFrame({PRICE_COL: price}, index=idx)
    panel["logret_1"] = pd.Series(logp, index=idx).diff()
    panel = panel.dropna(subset=[PRICE_COL])
    for j, sid in enumerate(EXTERNAL_DAILY):
        shock = rng.normal(0, 0.003 + 0.001 * j, n_days)
        logx = np.cumsum(shock) + np.log(1.0 + 0.1 * j)
        panel[sid] = np.exp(logx)
        panel[f"{sid}_logret_1"] = pd.Series(logx, index=idx).diff()
    return panel
