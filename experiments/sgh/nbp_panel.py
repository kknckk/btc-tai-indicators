"""Monthly panel for NBP reference-rate tier-I/II models (TRD-32, TRD-33)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments import macro_pl as mp
from experiments.config import ROOT

PI_TARGET = 2.5
RPP_COMM_CSV = ROOT / "data/macro_pl/raw/rpp_communications.csv"


def build_nbp_monthly_panel(
    *,
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    rate = mp.load_series("nbp_rate", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    cpi = mp.load_series("gus_cpi_yoy", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    pmi = mp.load_series("gus_pmi", freq="M", refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.concat([rate, cpi, pmi], axis=1, join="inner").sort_index()
    panel["logret_1"] = panel["nbp_rate"].diff() / 100.0
    panel = panel.dropna(subset=["nbp_rate", "gus_cpi_yoy", "gus_pmi"])
    return attach_rpp_communications(panel)


def nbp_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal features at month-end (lags only, no centered windows)."""
    r = panel["nbp_rate"]
    cpi = panel["gus_cpi_yoy"]
    pmi = panel["gus_pmi"]
    return pd.DataFrame(
        {
            "rate_now": r,
            "rate_l1": r.shift(1),
            "rate_l3": r.shift(3),
            "cpi_yoy": cpi,
            "cpi_l1": cpi.shift(1),
            "pmi": pmi,
            "pmi_l1": pmi.shift(1),
            "infl_gap": cpi - PI_TARGET,
            "pmi_gap": pmi - 50.0,
        },
        index=panel.index,
    )


def _months_since_rate_change(rate: pd.Series) -> pd.Series:
    out = np.zeros(len(rate), dtype=float)
    prev = float(rate.iloc[0])
    since = 0.0
    for i in range(len(rate)):
        if i > 0 and abs(float(rate.iloc[i]) - prev) > 0.01:
            since = 0.0
            prev = float(rate.iloc[i])
        elif i > 0:
            since += 1.0
        out[i] = since
    return pd.Series(out, index=rate.index)


def attach_rpp_communications(panel: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Optional ``raw/rpp_communications.csv`` (date, tone, text) — do weryfikacji."""
    path = path or RPP_COMM_CSV
    out = panel.copy()
    if not path.is_file():
        out["rpp_tone"] = 0.0
        out["rpp_comm_text"] = ""
        return out
    raw = pd.read_csv(path)
    if "date" not in raw.columns:
        out["rpp_tone"] = 0.0
        out["rpp_comm_text"] = ""
        return out
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    tone = raw["tone"] if "tone" in raw.columns else pd.Series(0.0, index=raw.index)
    text = raw["text"] if "text" in raw.columns else pd.Series("", index=raw.index)
    tone_m = tone.resample("ME").last().reindex(out.index, method="ffill").fillna(0.0)
    text_m = text.resample("ME").last().reindex(out.index, method="ffill").fillna("")
    out["rpp_tone"] = tone_m.astype(float)
    out["rpp_comm_text"] = text_m.astype(str)
    return out


def nbp_tier2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tier-II features: tier-I lags + macro endog, expectations proxy, RPP tone/NLP text."""
    base = nbp_features(panel)
    r = panel["nbp_rate"]
    tone = panel.get("rpp_tone", pd.Series(0.0, index=panel.index)).reindex(panel.index).fillna(0.0)
    text = panel.get("rpp_comm_text", pd.Series("", index=panel.index)).reindex(panel.index).fillna("")
    out = base.copy()
    out["nbp_rate"] = r
    out["gus_cpi_yoy"] = panel["gus_cpi_yoy"]
    out["gus_pmi"] = panel["gus_pmi"]
    out["rpp_tone"] = tone
    out["rpp_tone_l1"] = tone.shift(1)
    out["exp_rate_3m"] = r + 0.25 * (panel["gus_cpi_yoy"] - PI_TARGET)
    out["exp_rate_12m"] = r + (panel["gus_cpi_yoy"] - PI_TARGET)
    out["tone_x_infl"] = tone * out["infl_gap"]
    out["months_since_rate_change"] = _months_since_rate_change(r)
    out["rpp_comm_text"] = text
    return out


def demo_nbp_panel(n_months: int = 180, seed: int = 42) -> pd.DataFrame:
    """Deterministic smoke panel — not for production reporting."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n_months, freq="ME")
    cpi = 2.0 + rng.normal(0, 0.8, n_months).cumsum() * 0.05
    cpi = np.clip(cpi, 0.5, 12.0)
    pmi = 50 + rng.normal(0, 2.0, n_months).cumsum() * 0.2
    pmi = np.clip(pmi, 42, 58)
    rate = np.zeros(n_months)
    rate[0] = 4.0
    for t in range(1, n_months):
        target = 1.5 + 0.5 * cpi[t] + 0.15 * (50 - pmi[t])
        rate[t] = rate[t - 1] + np.clip(0.25 * (target - rate[t - 1]), -0.75, 0.75)
    panel = pd.DataFrame(
        {"nbp_rate": rate, "gus_cpi_yoy": cpi, "gus_pmi": pmi},
        index=idx,
    )
    panel["logret_1"] = panel["nbp_rate"].diff() / 100.0
    tone = np.clip(0.12 * (cpi - 3.0) - 0.04 * (pmi - 50.0), -1.0, 1.0)
    panel["rpp_tone"] = tone
    panel["rpp_comm_text"] = [
        f"RPP komunikat inflacja {cpi[i]:.1f} pmi {pmi[i]:.1f} ton {tone[i]:.2f}" for i in range(n_months)
    ]
    return panel
