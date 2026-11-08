"""Monthly panel for TFI net-assets tier-I models (TRD-44)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import macro_pl as mp

TFI_COL = "tfi_assets"


def _optional_series(name: str, *, use_cache_only: bool, refresh: bool) -> pd.Series | None:
    try:
        return mp.load_series(name, freq="M", refresh=refresh, use_cache_only=use_cache_only)
    except FileNotFoundError:
        return None


def build_tfi_assets_monthly_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    tfi = mp.load_series(TFI_COL, freq="M", refresh=refresh, use_cache_only=use_cache_only)
    rate = mp.load_series("nbp_rate", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    cpi = mp.load_series("gus_cpi_yoy", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    parts: list[pd.Series] = [tfi, rate, cpi]
    for name in ("deposits_households", "credit_consumer", "bond_pl_10y"):
        s = _optional_series(name, use_cache_only=use_cache_only, refresh=refresh)
        if s is not None:
            parts.append(s)
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    panel["logret_1"] = panel[TFI_COL].pct_change()
    need = [TFI_COL, "nbp_rate", "gus_cpi_yoy"]
    return panel.dropna(subset=need)


def _performance_proxy(panel: pd.DataFrame) -> pd.Series:
    """Causal monthly performance proxy for fund portfolios (lag applied in features)."""
    if "bond_pl_10y" in panel.columns:
        y = panel["bond_pl_10y"]
        return (-y.diff() / 100.0).rename("perf_raw")
    dep = panel.get("deposits_households")
    if dep is not None:
        return dep.pct_change().rename("perf_raw")
    return panel[TFI_COL].pct_change().rename("perf_raw")


def tfi_assets_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features at month-end (lags only)."""
    a = panel[TFI_COL]
    rate = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    mom = a.pct_change()
    perf = _performance_proxy(panel)
    flow = (mom - perf).rename("flow_raw")
    out: dict[str, pd.Series] = {
        "tfi_now": a,
        "tfi_l1": a.shift(1),
        "tfi_l2": a.shift(2),
        "tfi_l3": a.shift(3),
        "tfi_l12": a.shift(12),
        "tfi_mom_l1": mom.shift(1),
        "tfi_mom_l2": mom.shift(2),
        "perf_l1": perf.shift(1),
        "perf_l3": perf.shift(3),
        "flow_l1": flow.shift(1),
        "flow_l2": flow.shift(2),
        "nbp_rate": rate,
        "nbp_rate_l1": rate.shift(1),
        "nbp_rate_l3": rate.shift(3),
        "cpi_yoy": cpi,
        "cpi_l1": cpi.shift(1),
    }
    if "deposits_households" in panel.columns:
        d = panel["deposits_households"]
        out["deposits_l1"] = d.shift(1)
        out["deposits_mom_l1"] = d.pct_change().shift(1)
    if "credit_consumer" in panel.columns:
        c = panel["credit_consumer"]
        out["consumer_l1"] = c.shift(1)
        out["consumer_mom_l1"] = c.pct_change().shift(1)
    if "bond_pl_10y" in panel.columns:
        b = panel["bond_pl_10y"]
        out["bond_10y_l1"] = b.shift(1)
        out["bond_10y_chg_l1"] = b.diff().shift(1)
    return pd.DataFrame(out, index=panel.index)


def _expanding_z_lagged(s: pd.Series, min_periods: int = 24) -> pd.Series:
    lagged = s.shift(1)
    mu = lagged.expanding(min_periods=min_periods).mean()
    sd = lagged.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return (lagged - mu) / sd


def tfi_assets_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I lags + portfolio endog, attractiveness, sentiment proxies."""
    out = tfi_assets_features(panel)
    out[TFI_COL] = panel[TFI_COL]
    out["nbp_rate"] = panel["nbp_rate"]
    out["gus_cpi_yoy"] = panel["gus_cpi_yoy"]
    for name in ("deposits_households", "credit_consumer", "bond_pl_10y"):
        if name in panel.columns:
            out[name] = panel[name]
    a = panel[TFI_COL]
    rate = panel["nbp_rate"]
    mom = a.pct_change()
    perf = out.get("perf_l1")
    if perf is None:
        perf = _performance_proxy(panel).shift(1)
    flow = out.get("flow_l1")
    if flow is None:
        flow = (mom - _performance_proxy(panel)).shift(1)
    out["flow_z_l1"] = _expanding_z_lagged(flow.rename("flow")).shift(0)
    out["perf_z_l1"] = _expanding_z_lagged(perf.rename("perf")).shift(0)
    out["tfi_mom_z_l1"] = _expanding_z_lagged(mom.rename("mom")).shift(0)
    out["sentiment_index"] = pd.concat(
        [out["flow_z_l1"], out["perf_z_l1"], out["tfi_mom_z_l1"]],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["sentiment_index_l1"] = out["sentiment_index"].shift(1)
    dep_yield = None
    if "deposits_households" in panel.columns:
        dep = panel["deposits_households"]
        dep_yield = (rate / 100.0 + dep.pct_change()).rename("dep_yield_proxy")
        out["dep_yield_l1"] = dep_yield.shift(1)
    bond_yield = panel.get("bond_pl_10y")
    if bond_yield is not None:
        out["bond_minus_rate_l1"] = (bond_yield - rate).shift(1)
        if dep_yield is not None:
            out["bond_minus_dep_l1"] = (bond_yield / 100.0 - dep_yield).shift(1)
        out["perf_minus_bond_l1"] = (perf - bond_yield.diff() / 100.0).shift(0)
    elif dep_yield is not None:
        out["perf_minus_dep_l1"] = (perf - dep_yield).shift(0)
    spread = out.get("bond_minus_dep_l1")
    if spread is None:
        spread = out.get("bond_minus_rate_l1")
    perf_spread = out.get("perf_minus_bond_l1")
    if perf_spread is None:
        perf_spread = out.get("perf_minus_dep_l1")
    if spread is not None and perf_spread is not None:
        out["attractiveness_index"] = pd.concat([spread, perf_spread], axis=1).mean(axis=1, skipna=True)
    elif spread is not None:
        out["attractiveness_index"] = spread
    elif perf_spread is not None:
        out["attractiveness_index"] = perf_spread
    else:
        out["attractiveness_index"] = pd.Series(0.0, index=panel.index)
    out["attractiveness_l1"] = out["attractiveness_index"].shift(1)
    cpi = panel["gus_cpi_yoy"]
    out["cpi_q_mean_l1"] = cpi.rolling(3, min_periods=2).mean().shift(1)
    out["cpi_y_l1"] = cpi.shift(12)
    out["tfi_q_mean_l1"] = a.rolling(3, min_periods=2).mean().shift(1)
    rate_med = rate.expanding(min_periods=24).median().shift(1)
    out["high_rate_regime"] = (rate.shift(1) >= rate_med).astype(float)
    flow_med = flow.expanding(min_periods=24).median().shift(1)
    out["high_flow_regime"] = (flow.shift(1) >= flow_med).astype(float)
    return out


def demo_tfi_assets_panel(n_months: int = 200, seed: int = 45) -> pd.DataFrame:
    """Deterministic synthetic monthly panel — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rate = np.clip(3.0 + rng.normal(0, 0.12, n_months).cumsum() * 0.07, 0.5, 8.0)
    cpi = np.clip(2.5 + rng.normal(0, 0.18, n_months).cumsum() * 0.04, 0.5, 10.0)
    bond = 4.5 + rng.normal(0, 0.08, n_months).cumsum() * 0.05
    deposits = np.zeros(n_months)
    consumer = np.zeros(n_months)
    assets = np.zeros(n_months)
    deposits[0] = 700_000.0
    consumer[0] = 280_000.0
    assets[0] = 120_000.0
    for t in range(1, n_months):
        perf = 0.004 + 0.0008 * rng.normal()
        flow = 0.002 + 0.15 * perf + rng.normal(0, 0.003)
        assets[t] = assets[t - 1] * (1.0 + perf + flow)
        deposits[t] = deposits[t - 1] * (1.0 + 0.003 + rng.normal(0, 0.002))
        consumer[t] = consumer[t - 1] * (1.0 + 0.004 + rng.normal(0, 0.0025))
    panel = pd.DataFrame(
        {
            TFI_COL: assets,
            "nbp_rate": rate,
            "gus_cpi_yoy": cpi,
            "deposits_households": deposits,
            "credit_consumer": consumer,
            "bond_pl_10y": bond,
        },
        index=idx,
    )
    panel["logret_1"] = panel[TFI_COL].pct_change()
    return panel
