"""Daily PL bond-yield panel (2/5/7/10Y) for tier-I/II curve models (TRD-34, TRD-35)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import macro_pl as mp

BOND_COLS = ("bond_pl_2y", "bond_pl_5y", "bond_pl_7y", "bond_pl_10y")
MATURITY_YEARS = {c: y for c, y in zip(BOND_COLS, (2, 5, 7, 10))}
MATURITIES = np.array([2.0, 5.0, 7.0, 10.0], dtype=float)


def build_bonds_daily_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    parts = [
        mp.load_series(name, freq="D", refresh=refresh, use_cache_only=use_cache_only)
        for name in BOND_COLS
    ]
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    ref = panel[BOND_COLS[-1]]
    panel["logret_1"] = ref.diff() / 100.0
    return panel.dropna(subset=list(BOND_COLS))


def bonds_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features: spot yields and short lags only."""
    out: dict[str, pd.Series] = {}
    for c in BOND_COLS:
        out[c] = panel[c]
        out[f"{c}_l1"] = panel[c].shift(1)
        out[f"{c}_l5"] = panel[c].shift(5)
    return pd.DataFrame(out, index=panel.index)


def demo_bonds_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic yield curve — smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    lam = 0.07
    lt = lam * MATURITIES
    f2 = (1.0 - np.exp(-lt)) / lt
    f3 = f2 - np.exp(-lt)
    load = np.column_stack([np.ones(4), f2, f3])
    betas = np.zeros((n_days, 3))
    betas[0] = [4.5, -1.2, 0.8]
    for t in range(1, n_days):
        betas[t] = betas[t - 1] + rng.normal(0, [0.02, 0.03, 0.02])
    noise = rng.normal(0, 0.03, (n_days, 4))
    yields = betas @ load.T + noise
    panel = pd.DataFrame(
        {c: yields[:, i] for i, c in enumerate(BOND_COLS)},
        index=idx,
    )
    panel["logret_1"] = panel["bond_pl_10y"].diff() / 100.0
    nbp = 4.0 + 0.3 * np.sin(np.arange(n_days) / 80.0)
    cpi = 3.0 + 0.15 * np.cos(np.arange(n_days) / 120.0)
    panel["nbp_rate"] = nbp
    panel["gus_cpi_yoy"] = cpi
    return panel.dropna(subset=list(BOND_COLS))


def build_bonds_tier2_daily_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Bond yields plus macro series (ffill to business days) for tier-II VARX / macro-finance."""
    panel = build_bonds_daily_panel(use_cache_only=use_cache_only, refresh=refresh)
    for name in ("nbp_rate", "gus_cpi_yoy"):
        try:
            s = mp.load_series(name, freq="D", refresh=refresh, use_cache_only=use_cache_only)
            panel[name] = s.reindex(panel.index).ffill()
        except FileNotFoundError:
            panel[name] = np.nan
    return panel


def bonds_tier2_features(panel: pd.DataFrame, *, ns_lambda: float = 0.07) -> pd.DataFrame:
    """Causal tier-II features: yields, lags, NS factors (contemporaneous curve), slope, macro lags."""
    from experiments.sgh.tier1_bonds import _extract_factors, _ns_loadings

    base = bonds_features(panel)
    Y = panel[list(BOND_COLS)].astype(float)
    L = _ns_loadings(MATURITIES, ns_lambda)
    fac = _extract_factors(Y.to_numpy(), L)
    base["ns_f1"] = fac[:, 0]
    base["ns_f2"] = fac[:, 1]
    base["ns_f3"] = fac[:, 2]
    slope = panel["bond_pl_10y"] - panel["bond_pl_2y"]
    base["slope"] = slope
    base["slope_l1"] = slope.shift(1)
    base["curvature"] = 2.0 * panel["bond_pl_5y"] - panel["bond_pl_2y"] - panel["bond_pl_10y"]
    if "nbp_rate" in panel.columns:
        nbp = panel["nbp_rate"].astype(float)
        base["nbp_rate"] = nbp
        base["nbp_l1"] = nbp.shift(1)
        base["nbp_gap"] = nbp - panel["bond_pl_2y"]
    if "gus_cpi_yoy" in panel.columns:
        cpi = panel["gus_cpi_yoy"].astype(float)
        base["gus_cpi_yoy"] = cpi
        base["cpi_l1"] = cpi.shift(1)
        base["infl_gap"] = cpi - 2.5
    return base
