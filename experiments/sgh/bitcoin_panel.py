"""Daily BTC price + on-chain panel for tier-I models (TRD-58)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.data import build_features, load_panel, signed_log1p

PRICE_COL = "PriceUSD"


def build_bitcoin_daily_panel(
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Merged on-chain + ``PriceUSD`` panel (``experiments.data.load_panel``)."""
    return load_panel(start=start, end=end)


def bitcoin_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal on-chain feature set plus spot level for level-forecast fallbacks."""
    X = build_features(panel)
    X[PRICE_COL] = panel[PRICE_COL]
    return X


def demo_bitcoin_panel(n_days: int = 900, seed: int = 58) -> pd.DataFrame:
    """Deterministic synthetic BTC panel — smoke tests only (not production metrics)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_days, freq="D")
    logp = np.zeros(n_days)
    logp[0] = np.log(8000.0)
    for t in range(1, n_days):
        logp[t] = logp[t - 1] + rng.normal(0, 0.025)
    price = np.exp(logp)
    panel = pd.DataFrame({PRICE_COL: price}, index=idx)
    panel["log_price"] = pd.Series(logp, index=idx)
    panel["logret_1"] = panel["log_price"].diff()
    r = panel["logret_1"]
    panel["rv_7"] = r.rolling(7, min_periods=7).std() * np.sqrt(365)
    panel["rv_30"] = r.rolling(30, min_periods=30).std() * np.sqrt(365)
    cap = price * rng.uniform(0.9, 1.1, n_days) * 1e6
    panel["CapMrktCurUSD"] = cap
    panel["slog1p_CapMrktCurUSD"] = signed_log1p(cap)
    panel["MVRV"] = 1.2 + rng.normal(0, 0.08, n_days).cumsum() * 0.01
    panel["NUPL"] = rng.normal(0.0, 0.15, n_days)
    return panel
