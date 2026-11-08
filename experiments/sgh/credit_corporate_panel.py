"""Monthly panel for corporate-credit tier-I models (TRD-40)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import macro_pl as mp

CORPORATE_COL = "credit_corporate"


def _optional_credit_standards_firms(
    refresh: bool = False, use_cache_only: bool = False
) -> pd.Series | None:
    """NBP credit standards survey — firms (``raw/nbp_credit_standards_firms.csv``, time + value). **do weryfikacji**."""
    del refresh, use_cache_only
    path = mp.RAW_DIR / "nbp_credit_standards_firms.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "time" not in df.columns:
        return None
    val_col = next((c for c in df.columns if c != "time"), None)
    if val_col is None:
        return None
    idx = pd.to_datetime(df["time"], errors="coerce")
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=idx,
        name="credit_standards_firms",
    )
    s.index = s.index.to_period("M").to_timestamp("M")
    return s.sort_index().dropna()


def build_credit_corporate_monthly_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    corporate = mp.load_series(CORPORATE_COL, freq="M", refresh=refresh, use_cache_only=use_cache_only)
    rate = mp.load_series("nbp_rate", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    cpi = mp.load_series("gus_cpi_yoy", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    consumer = mp.load_series("credit_consumer", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    mortgage = mp.load_series("credit_mortgage", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    parts = [corporate, rate, cpi, consumer, mortgage]
    standards = _optional_credit_standards_firms(refresh=refresh, use_cache_only=use_cache_only)
    if standards is not None:
        parts.append(standards)
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    panel["logret_1"] = panel[CORPORATE_COL].pct_change()
    need = [CORPORATE_COL, "nbp_rate", "gus_cpi_yoy"]
    return panel.dropna(subset=need)


def credit_corporate_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features at month-end (lags only)."""
    c = panel[CORPORATE_COL]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    dlog = np.log(c.clip(lower=1.0)).diff()
    drate = rate.diff()
    out: dict[str, pd.Series] = {
        "corporate_now": c,
        "corporate_l1": c.shift(1),
        "corporate_l2": c.shift(2),
        "corporate_l3": c.shift(3),
        "corporate_l12": c.shift(12),
        "corporate_mom_l1": c.pct_change().shift(1),
        "corporate_dlog_l1": dlog.shift(1),
        "corporate_dlog_l2": dlog.shift(2),
        "nbp_rate": rate,
        "nbp_rate_l1": rate.shift(1),
        "nbp_rate_l3": rate.shift(3),
        "nbp_drate": drate,
        "nbp_drate_l1": drate.shift(1),
        "nbp_drate_l2": drate.shift(2),
        "nbp_drate_l3": drate.shift(3),
        "cpi_yoy": cpi,
        "cpi_l1": cpi.shift(1),
    }
    if "credit_consumer" in panel.columns:
        hc = panel["credit_consumer"]
        out["consumer_l1"] = hc.shift(1)
        out["consumer_mom_l1"] = hc.pct_change().shift(1)
    if "credit_mortgage" in panel.columns:
        m = panel["credit_mortgage"]
        out["mortgage_l1"] = m.shift(1)
    if "credit_standards_firms" in panel.columns:
        st = panel["credit_standards_firms"]
        out["standards_l1"] = st.shift(1)
        out["standards_l2"] = st.shift(2)
    return pd.DataFrame(out, index=panel.index)


def demo_credit_corporate_panel(n_months: int = 200, seed: int = 44) -> pd.DataFrame:
    """Deterministic synthetic monthly panel — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rate = np.clip(3.0 + rng.normal(0, 0.15, n_months).cumsum() * 0.08, 0.5, 8.0)
    cpi = np.clip(2.5 + rng.normal(0, 0.2, n_months).cumsum() * 0.05, 0.5, 10.0)
    consumer = np.zeros(n_months)
    mortgage = np.zeros(n_months)
    corporate = np.zeros(n_months)
    corporate[0] = 520_000.0
    consumer[0] = 250_000.0
    mortgage[0] = 400_000.0
    for t in range(1, n_months):
        g = 0.0035 + 0.0005 * (cpi[t] - 2.5) - 0.0018 * (rate[t] - rate[t - 1])
        g += rng.normal(0, 0.0028)
        corporate[t] = corporate[t - 1] * (1.0 + g)
        consumer[t] = consumer[t - 1] * (1.0 + 0.004 + rng.normal(0, 0.003))
        mortgage[t] = mortgage[t - 1] * (1.0 + 0.003 + rng.normal(0, 0.002))
    panel = pd.DataFrame(
        {
            CORPORATE_COL: corporate,
            "nbp_rate": rate,
            "gus_cpi_yoy": cpi,
            "credit_consumer": consumer,
            "credit_mortgage": mortgage,
        },
        index=idx,
    )
    rng_std = np.random.default_rng(seed + 1)
    standards = 0.1 * rng_std.normal(0, 1, n_months).cumsum() * 0.05
    panel["credit_standards_firms"] = standards
    panel["logret_1"] = panel[CORPORATE_COL].pct_change()
    return panel


def _expanding_z_lagged(s: pd.Series, min_periods: int = 12) -> pd.Series:
    """Causal z-score: at *t* uses history strictly before *t*."""
    lagged = s.shift(1)
    mu = lagged.expanding(min_periods=min_periods).mean()
    sd = lagged.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return (lagged - mu) / sd


def credit_corporate_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I lags + macro endog for VAR/BVAR, FCI, ML (TRD-41)."""
    out = credit_corporate_features(panel)
    out["credit_corporate"] = panel[CORPORATE_COL]
    out["nbp_rate"] = panel["nbp_rate"]
    out["gus_cpi_yoy"] = panel["gus_cpi_yoy"]
    if "credit_consumer" in panel.columns:
        out["credit_consumer"] = panel["credit_consumer"]
    if "credit_mortgage" in panel.columns:
        out["credit_mortgage"] = panel["credit_mortgage"]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    corp = panel[CORPORATE_COL]
    parts = [
        _expanding_z_lagged(rate).rename("fci_z_rate"),
        _expanding_z_lagged(cpi).rename("fci_z_cpi"),
    ]
    if "credit_consumer" in panel.columns:
        cm = panel["credit_consumer"].pct_change()
        parts.append(_expanding_z_lagged(cm).rename("fci_z_consumer_mom"))
    if "credit_mortgage" in panel.columns:
        mm = panel["credit_mortgage"].pct_change()
        parts.append(_expanding_z_lagged(mm).rename("fci_z_mortgage_mom"))
    corp_mom = corp.pct_change()
    parts.append((-_expanding_z_lagged(corp_mom)).rename("fci_z_corporate_mom"))
    fci_frame = pd.concat(parts, axis=1)
    out["fci"] = fci_frame.mean(axis=1, skipna=True)
    out["fci_l1"] = out["fci"].shift(1)
    out["fci_l3"] = out["fci"].shift(3)
    if "credit_consumer" in panel.columns and "credit_mortgage" in panel.columns:
        total = panel["credit_consumer"] + panel["credit_mortgage"] + corp
        out["corporate_share_l1"] = (corp / total.replace(0, np.nan)).shift(1)
    return out
