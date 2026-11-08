"""Monthly panel for mortgage-credit tier-I models (TRD-38)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import macro_pl as mp
from experiments.sgh.credit_consumer_panel import _optional_bik_monthly

MORTGAGE_COL = "credit_mortgage"


def build_credit_mortgage_monthly_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    mortgage = mp.load_series(MORTGAGE_COL, freq="M", refresh=refresh, use_cache_only=use_cache_only)
    rate = mp.load_series("nbp_rate", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    cpi = mp.load_series("gus_cpi_yoy", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    consumer = mp.load_series("credit_consumer", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    deposits = mp.load_series("deposits_households", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    parts = [mortgage, rate, cpi, consumer, deposits]
    bik = _optional_bik_monthly(refresh=refresh, use_cache_only=use_cache_only)
    if bik is not None:
        parts.append(bik)
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    panel["logret_1"] = panel[MORTGAGE_COL].pct_change()
    need = [MORTGAGE_COL, "nbp_rate", "gus_cpi_yoy"]
    return panel.dropna(subset=need)


def credit_mortgage_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features at month-end (lags only)."""
    m = panel[MORTGAGE_COL]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    out: dict[str, pd.Series] = {
        "mortgage_now": m,
        "mortgage_l1": m.shift(1),
        "mortgage_l2": m.shift(2),
        "mortgage_l3": m.shift(3),
        "mortgage_l12": m.shift(12),
        "mortgage_mom_l1": m.pct_change().shift(1),
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
    if "deposits_households" in panel.columns:
        d = panel["deposits_households"]
        out["deposits_l1"] = d.shift(1)
    if "bik_index" in panel.columns:
        b = panel["bik_index"]
        out["bik_l1"] = b.shift(1)
        out["bik_mom_l1"] = b.pct_change().shift(1)
    return pd.DataFrame(out, index=panel.index)


def credit_mortgage_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I lags + macro endog for VAR, panel ratios, contract stress."""
    out = credit_mortgage_features(panel)
    out["credit_mortgage"] = panel[MORTGAGE_COL]
    out["nbp_rate"] = panel["nbp_rate"]
    out["gus_cpi_yoy"] = panel["gus_cpi_yoy"]
    if "credit_consumer" in panel.columns:
        out["credit_consumer"] = panel["credit_consumer"]
    if "deposits_households" in panel.columns:
        out["deposits_households"] = panel["deposits_households"]
    m = panel[MORTGAGE_COL]
    rate = panel["nbp_rate"]
    out["rate_chg_l1"] = rate.diff().shift(1)
    if "credit_consumer" in panel.columns and "deposits_households" in panel.columns:
        c = panel["credit_consumer"]
        d = panel["deposits_households"]
        total = m + c + d
        out["mtg_share_l1"] = (m / total.replace(0, np.nan)).shift(1)
        out["consumer_share_l1"] = (c / total.replace(0, np.nan)).shift(1)
        out["affordability_l1"] = (d / m.replace(0, np.nan)).shift(1)
    return out


def demo_credit_mortgage_panel(n_months: int = 200, seed: int = 43) -> pd.DataFrame:
    """Deterministic synthetic monthly panel — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rate = np.clip(3.0 + rng.normal(0, 0.15, n_months).cumsum() * 0.08, 0.5, 8.0)
    cpi = np.clip(2.5 + rng.normal(0, 0.2, n_months).cumsum() * 0.05, 0.5, 10.0)
    consumer = np.zeros(n_months)
    deposits = np.zeros(n_months)
    mortgage = np.zeros(n_months)
    mortgage[0] = 400_000.0
    consumer[0] = 250_000.0
    deposits[0] = 800_000.0
    for t in range(1, n_months):
        g = 0.003 + 0.0006 * (cpi[t] - 2.5) - 0.0015 * (rate[t] - rate[t - 1])
        g += rng.normal(0, 0.0025)
        mortgage[t] = mortgage[t - 1] * (1.0 + g)
        consumer[t] = consumer[t - 1] * (1.0 + 0.004 + rng.normal(0, 0.003))
        deposits[t] = deposits[t - 1] * (1.0 + 0.002 + rng.normal(0, 0.0015))
    panel = pd.DataFrame(
        {
            MORTGAGE_COL: mortgage,
            "nbp_rate": rate,
            "gus_cpi_yoy": cpi,
            "credit_consumer": consumer,
            "deposits_households": deposits,
        },
        index=idx,
    )
    panel["logret_1"] = panel[MORTGAGE_COL].pct_change()
    return panel
