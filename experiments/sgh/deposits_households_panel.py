"""Monthly panel for household-deposits tier-I models (TRD-42)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import macro_pl as mp

DEPOSITS_COL = "deposits_households"


def build_deposits_households_monthly_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    deposits = mp.load_series(DEPOSITS_COL, freq="M", refresh=refresh, use_cache_only=use_cache_only)
    rate = mp.load_series("nbp_rate", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    cpi = mp.load_series("gus_cpi_yoy", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    consumer = mp.load_series("credit_consumer", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    mortgage = mp.load_series("credit_mortgage", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.concat([deposits, rate, cpi, consumer, mortgage], axis=1, join="inner").sort_index()
    panel["logret_1"] = panel[DEPOSITS_COL].pct_change()
    need = [DEPOSITS_COL, "nbp_rate", "gus_cpi_yoy"]
    return panel.dropna(subset=need)


def deposits_households_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features at month-end (lags only)."""
    d = panel[DEPOSITS_COL]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    out: dict[str, pd.Series] = {
        "deposits_now": d,
        "deposits_l1": d.shift(1),
        "deposits_l2": d.shift(2),
        "deposits_l3": d.shift(3),
        "deposits_l12": d.shift(12),
        "deposits_mom_l1": d.pct_change().shift(1),
        "nbp_rate": rate,
        "nbp_rate_l1": rate.shift(1),
        "nbp_rate_l3": rate.shift(3),
        "cpi_yoy": cpi,
        "cpi_l1": cpi.shift(1),
    }
    if "credit_consumer" in panel.columns:
        c = panel["credit_consumer"]
        out["consumer_l1"] = c.shift(1)
        out["consumer_mom_l1"] = c.pct_change().shift(1)
    if "credit_mortgage" in panel.columns:
        m = panel["credit_mortgage"]
        out["mortgage_l1"] = m.shift(1)
        out["mortgage_mom_l1"] = m.pct_change().shift(1)
    return pd.DataFrame(out, index=panel.index)


def _expanding_z_lagged(s: pd.Series, min_periods: int = 12) -> pd.Series:
    """Causal z-score: at *t* uses history strictly before *t*."""
    lagged = s.shift(1)
    mu = lagged.expanding(min_periods=min_periods).mean()
    sd = lagged.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return (lagged - mu) / sd


def deposits_households_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I lags + macro endog for VAR/BVAR, portfolio/ML."""
    out = deposits_households_features(panel)
    out[DEPOSITS_COL] = panel[DEPOSITS_COL]
    out["nbp_rate"] = panel["nbp_rate"]
    out["gus_cpi_yoy"] = panel["gus_cpi_yoy"]
    if "credit_consumer" in panel.columns:
        out["credit_consumer"] = panel["credit_consumer"]
    if "credit_mortgage" in panel.columns:
        out["credit_mortgage"] = panel["credit_mortgage"]
    dep = panel[DEPOSITS_COL]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    parts = [
        _expanding_z_lagged(rate).rename("liquidity_z_rate"),
        _expanding_z_lagged(cpi).rename("liquidity_z_cpi"),
        _expanding_z_lagged(dep.pct_change()).rename("liquidity_z_deposits_mom"),
    ]
    if "credit_consumer" in panel.columns:
        parts.append(_expanding_z_lagged(panel["credit_consumer"].pct_change()).rename("liquidity_z_consumer_mom"))
    liq = pd.concat(parts, axis=1)
    out["liquidity_index"] = liq.mean(axis=1, skipna=True)
    out["liquidity_index_l1"] = out["liquidity_index"].shift(1)
    if "credit_consumer" in panel.columns:
        ratio = (panel["credit_consumer"] / dep.replace(0, np.nan)).shift(1)
        out["credit_deposit_ratio_l1"] = ratio
    rate_med = rate.expanding(min_periods=24).median().shift(1)
    out["high_rate_regime"] = (rate.shift(1) >= rate_med).astype(float)
    return out


def demo_deposits_households_panel(n_months: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic monthly panel — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rate = np.clip(3.0 + rng.normal(0, 0.15, n_months).cumsum() * 0.08, 0.5, 8.0)
    cpi = np.clip(2.5 + rng.normal(0, 0.2, n_months).cumsum() * 0.05, 0.5, 10.0)
    consumer = np.zeros(n_months)
    mortgage = np.zeros(n_months)
    deposits = np.zeros(n_months)
    consumer[0] = 250_000.0
    mortgage[0] = 400_000.0
    deposits[0] = 800_000.0
    for t in range(1, n_months):
        g = 0.003 + 0.0012 * (rate[t - 1] - 3.0) + 0.0005 * (cpi[t] - 2.5)
        g += rng.normal(0, 0.0025)
        deposits[t] = deposits[t - 1] * (1.0 + g)
        consumer[t] = consumer[t - 1] * (1.0 + 0.004 + rng.normal(0, 0.003))
        mortgage[t] = mortgage[t - 1] * (1.0 + 0.003 + rng.normal(0, 0.002))
    panel = pd.DataFrame(
        {
            DEPOSITS_COL: deposits,
            "nbp_rate": rate,
            "gus_cpi_yoy": cpi,
            "credit_consumer": consumer,
            "credit_mortgage": mortgage,
        },
        index=idx,
    )
    panel["logret_1"] = panel[DEPOSITS_COL].pct_change()
    return panel
