"""Daily WTI crude (CL=F) panel for tier-I/II models (TRD-54 / TRD-55)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import multiasset as ma

PRICE_COL = "oil"


def build_oil_daily_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    s = ma.load_series(PRICE_COL, freq="D", refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.DataFrame({PRICE_COL: s}).sort_index()
    panel["logret_1"] = np.log(panel[PRICE_COL]).diff()
    return panel.dropna(subset=[PRICE_COL])


def oil_features(panel: pd.DataFrame) -> pd.DataFrame:
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


def oil_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I tabular + causal MA-EMD modes and return z-scores."""
    base = oil_features(panel)
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
    for imf in ("emd_hf", "emd_mf", "emd_lf"):
        s = base[imf]
        base[f"{imf}_l1"] = s.shift(1)
        base[f"{imf}_l2"] = s.shift(2)
    base["emd_trend_l1"] = ma63.shift(1)
    base["emd_trend_l2"] = ma63.shift(2)
    return base


def demo_oil_panel(n_days: int = 900, seed: int = 54) -> pd.DataFrame:
    """Deterministic synthetic WTI futures — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    logp = np.zeros(n_days)
    logp[0] = np.log(72.0)
    for t in range(1, n_days):
        logp[t] = logp[t - 1] + rng.normal(0, 0.018)
    price = np.exp(logp)
    panel = pd.DataFrame({PRICE_COL: price}, index=idx)
    panel["logret_1"] = pd.Series(logp, index=idx).diff()
    return panel.dropna(subset=[PRICE_COL])
