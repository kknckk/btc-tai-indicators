"""Daily CBOT wheat (ZW=F) panel for tier-I/II models (TRD-52 / TRD-53)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import multiasset as ma

PRICE_COL = "wheat"


def build_wheat_daily_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    s = ma.load_series(PRICE_COL, freq="D", refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.DataFrame({PRICE_COL: s}).sort_index()
    panel["logret_1"] = np.log(panel[PRICE_COL]).diff()
    return panel.dropna(subset=[PRICE_COL])


def wheat_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features: futures price and trailing lags / rolling stats only."""
    p = panel[PRICE_COL]
    r = panel["logret_1"] if "logret_1" in panel.columns else np.log(p).diff()
    out = {
        PRICE_COL: p,
        f"{PRICE_COL}_l1": p.shift(1),
        f"{PRICE_COL}_l2": p.shift(2),
        f"{PRICE_COL}_l5": p.shift(5),
        f"{PRICE_COL}_l21": p.shift(21),
        f"{PRICE_COL}_ma5": p.rolling(5, min_periods=5).mean(),
        f"{PRICE_COL}_ma21": p.rolling(21, min_periods=21).mean(),
        f"{PRICE_COL}_vol21": r.rolling(21, min_periods=21).std(),
    }
    return pd.DataFrame(out, index=panel.index)


def wheat_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I tabular + causal seasonal modes and Fourier phases."""
    base = wheat_features(panel)
    p = panel[PRICE_COL].reindex(base.index)
    r = panel["logret_1"].reindex(base.index) if "logret_1" in panel.columns else np.log(p).diff()
    mu = r.rolling(21, min_periods=21).mean()
    sd = r.rolling(21, min_periods=21).std().replace(0, np.nan)
    base["ret_z21"] = (r - mu) / sd
    base["abs_ret_z21"] = base["ret_z21"].abs()
    ma5 = p.rolling(5, min_periods=5).mean()
    ma21 = p.rolling(21, min_periods=21).mean()
    ma63 = p.rolling(63, min_periods=63).mean()
    base["emd_hf"] = p - ma5
    base["emd_mf"] = ma5 - ma21
    base["emd_lf"] = ma21 - ma63
    t = np.arange(len(base), dtype=float)
    for period in (5, 21, 63):
        phase = 2.0 * np.pi * t / period
        base[f"fourier_sin_{period}"] = np.sin(phase) * p.shift(1)
        base[f"fourier_cos_{period}"] = np.cos(phase) * p.shift(1)
    base["time_idx"] = base.index.dayofyear.astype(float)
    return base


def demo_wheat_panel(n_days: int = 900, seed: int = 52) -> pd.DataFrame:
    """Deterministic synthetic wheat futures — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    logp = np.zeros(n_days)
    logp[0] = np.log(520.0)
    for t in range(1, n_days):
        logp[t] = logp[t - 1] + rng.normal(0, 0.012)
    price = np.exp(logp)
    panel = pd.DataFrame({PRICE_COL: price}, index=idx)
    panel["logret_1"] = pd.Series(logp, index=idx).diff()
    return panel.dropna(subset=[PRICE_COL])
