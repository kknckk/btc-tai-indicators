"""Daily USD/PLN panel for tier-I FX models (TRD-48)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import multiasset as ma

PRICE_COL = "usdpln"


def build_usdpln_daily_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    s = ma.load_series(PRICE_COL, freq="D", refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.DataFrame({PRICE_COL: s}).sort_index()
    panel["logret_1"] = np.log(panel[PRICE_COL]).diff()
    return panel.dropna(subset=[PRICE_COL])


def usdpln_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features: spot FX and trailing lags / rolling stats only."""
    p = panel[PRICE_COL]
    out = {
        PRICE_COL: p,
        f"{PRICE_COL}_l1": p.shift(1),
        f"{PRICE_COL}_l2": p.shift(2),
        f"{PRICE_COL}_l5": p.shift(5),
        f"{PRICE_COL}_l21": p.shift(21),
        f"{PRICE_COL}_ma5": p.rolling(5, min_periods=5).mean(),
        f"{PRICE_COL}_ma21": p.rolling(21, min_periods=21).mean(),
        f"{PRICE_COL}_vol21": p.pct_change().rolling(21, min_periods=21).std(),
    }
    return pd.DataFrame(out, index=panel.index)


def usdpln_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I tabular + causal VMD-style modes and return z-scores."""
    base = usdpln_features(panel)
    p = panel[PRICE_COL].reindex(base.index)
    ret = p.pct_change()
    mu = ret.rolling(21, min_periods=21).mean()
    sd = ret.rolling(21, min_periods=21).std().replace(0, np.nan)
    base["ret_z21"] = (ret - mu) / sd
    base["abs_ret_z21"] = base["ret_z21"].abs()
    ma5 = p.rolling(5, min_periods=5).mean()
    ma21 = p.rolling(21, min_periods=21).mean()
    ma63 = p.rolling(63, min_periods=63).mean()
    base["vmd_hf"] = p - ma5
    base["vmd_mf"] = ma5 - ma21
    base["vmd_lf"] = ma21 - ma63
    return base


def demo_usdpln_panel(n_days: int = 900, seed: int = 48) -> pd.DataFrame:
    """Deterministic synthetic USD/PLN — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    logp = np.zeros(n_days)
    logp[0] = np.log(3.95)
    for t in range(1, n_days):
        logp[t] = logp[t - 1] + rng.normal(0, 0.002)
    price = np.exp(logp)
    panel = pd.DataFrame({PRICE_COL: price}, index=idx)
    panel["logret_1"] = pd.Series(logp, index=idx).diff()
    return panel.dropna(subset=[PRICE_COL])
