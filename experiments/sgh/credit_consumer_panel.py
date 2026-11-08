"""Monthly panel for consumer-credit tier-I models (TRD-36)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import macro_pl as mp

CREDIT_COL = "credit_consumer"


def _optional_bik_monthly(refresh: bool = False, use_cache_only: bool = False) -> pd.Series | None:
    """Optional BIK indicator from ``data/macro_pl/raw/bik_macro.csv`` (time,bik_index)."""
    path = mp.RAW_DIR / "bik_macro.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "time" not in df.columns:
        return None
    val_col = next((c for c in df.columns if c != "time"), None)
    if val_col is None:
        return None
    idx = pd.to_datetime(df["time"], errors="coerce")
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").to_numpy(), index=idx, name="bik_index")
    s.index = s.index.to_period("M").to_timestamp("M")
    return s.sort_index().dropna()


def build_credit_consumer_monthly_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    credit = mp.load_series(CREDIT_COL, freq="M", refresh=refresh, use_cache_only=use_cache_only)
    rate = mp.load_series("nbp_rate", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    cpi = mp.load_series("gus_cpi_yoy", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    mortgage = mp.load_series("credit_mortgage", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    deposits = mp.load_series("deposits_households", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    parts = [credit, rate, cpi, mortgage, deposits]
    bik = _optional_bik_monthly(refresh=refresh, use_cache_only=use_cache_only)
    if bik is not None:
        parts.append(bik)
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    panel["logret_1"] = panel[CREDIT_COL].pct_change()
    need = [CREDIT_COL, "nbp_rate", "gus_cpi_yoy"]
    return panel.dropna(subset=need)


def credit_consumer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features at month-end (lags only)."""
    c = panel[CREDIT_COL]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    out: dict[str, pd.Series] = {
        "credit_now": c,
        "credit_l1": c.shift(1),
        "credit_l2": c.shift(2),
        "credit_l3": c.shift(3),
        "credit_l12": c.shift(12),
        "credit_mom_l1": c.pct_change().shift(1),
        "nbp_rate": rate,
        "nbp_rate_l1": rate.shift(1),
        "nbp_rate_l3": rate.shift(3),
        "cpi_yoy": cpi,
        "cpi_l1": cpi.shift(1),
    }
    if "credit_mortgage" in panel.columns:
        m = panel["credit_mortgage"]
        out["mortgage_l1"] = m.shift(1)
        out["mortgage_mom_l1"] = m.pct_change().shift(1)
    if "deposits_households" in panel.columns:
        d = panel["deposits_households"]
        out["deposits_l1"] = d.shift(1)
    if "bik_index" in panel.columns:
        b = panel["bik_index"]
        out["bik_l1"] = b.shift(1)
        out["bik_mom_l1"] = b.pct_change().shift(1)
    return pd.DataFrame(out, index=panel.index)


def _expanding_z_lagged(s: pd.Series, min_periods: int = 12) -> pd.Series:
    """Causal z-score: at *t* uses history strictly before *t*."""
    lagged = s.shift(1)
    mu = lagged.expanding(min_periods=min_periods).mean()
    sd = lagged.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return (lagged - mu) / sd


def credit_consumer_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I lags + macro endog for VAR/BVAR, FCI, ML."""
    out = credit_consumer_features(panel)
    out["credit_consumer"] = panel[CREDIT_COL]
    out["nbp_rate"] = panel["nbp_rate"]
    out["gus_cpi_yoy"] = panel["gus_cpi_yoy"]
    if "credit_mortgage" in panel.columns:
        out["credit_mortgage"] = panel["credit_mortgage"]
    if "deposits_households" in panel.columns:
        out["deposits_households"] = panel["deposits_households"]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    credit = panel[CREDIT_COL]
    parts = [
        _expanding_z_lagged(rate).rename("fci_z_rate"),
        _expanding_z_lagged(cpi).rename("fci_z_cpi"),
    ]
    if "credit_mortgage" in panel.columns:
        mm = panel["credit_mortgage"].pct_change()
        parts.append(_expanding_z_lagged(mm).rename("fci_z_mortgage_mom"))
    cm = credit.pct_change()
    parts.append((-_expanding_z_lagged(cm)).rename("fci_z_credit_mom"))
    if "deposits_households" in panel.columns:
        dm = panel["deposits_households"].pct_change()
        parts.append((-_expanding_z_lagged(dm)).rename("fci_z_deposits_mom"))
    fci_frame = pd.concat(parts, axis=1)
    out["fci"] = fci_frame.mean(axis=1, skipna=True)
    out["fci_l1"] = out["fci"].shift(1)
    out["fci_l3"] = out["fci"].shift(3)
    if "deposits_households" in panel.columns:
        dep = panel["deposits_households"]
        ratio = (credit / dep.replace(0, np.nan)).shift(1)
        out["credit_deposit_ratio_l1"] = ratio
    return out


def demo_credit_consumer_panel(n_months: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic monthly panel — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rate = np.clip(3.0 + rng.normal(0, 0.15, n_months).cumsum() * 0.08, 0.5, 8.0)
    cpi = np.clip(2.5 + rng.normal(0, 0.2, n_months).cumsum() * 0.05, 0.5, 10.0)
    mortgage = np.zeros(n_months)
    deposits = np.zeros(n_months)
    credit = np.zeros(n_months)
    credit[0] = 250_000.0
    mortgage[0] = 400_000.0
    deposits[0] = 800_000.0
    for t in range(1, n_months):
        g = 0.004 + 0.0008 * (cpi[t] - 2.5) - 0.002 * (rate[t] - rate[t - 1])
        g += rng.normal(0, 0.003)
        credit[t] = credit[t - 1] * (1.0 + g)
        mortgage[t] = mortgage[t - 1] * (1.0 + 0.003 + rng.normal(0, 0.002))
        deposits[t] = deposits[t - 1] * (1.0 + 0.002 + rng.normal(0, 0.0015))
    panel = pd.DataFrame(
        {
            CREDIT_COL: credit,
            "nbp_rate": rate,
            "gus_cpi_yoy": cpi,
            "credit_mortgage": mortgage,
            "deposits_households": deposits,
        },
        index=idx,
    )
    panel["logret_1"] = panel[CREDIT_COL].pct_change()
    return panel
