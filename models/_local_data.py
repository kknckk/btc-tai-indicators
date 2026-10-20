"""Fallback data loader, used only when `experiments.data.load_panel` is not
importable (i.e. the parallel branch adding it hasn't been merged yet).

Mirrors the outer-join-on-`time` merge logic of
`analytical_pipeline/utils.py::load_merged_data`, but also joins in BTC
price from `api/csv/PriceUSD.csv` so callers can derive log-returns.
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ONCHAIN_DIR = os.path.join(_REPO_ROOT, "data_ingestion", "csv")
_PRICE_CSV = os.path.join(_REPO_ROOT, "api", "csv", "PriceUSD.csv")

DEFAULT_FEATURES = [
    "AdrActCnt",
    "TxCntSec",
    "FeeTotUSD",
    "FeeMeanUSD",
    "HashRate",
    "NUPL",
    "MVRV_Z",
    "RevUSD",
]


def load_panel(
    feature_columns: Optional[List[str]] = None, min_coverage: float = 0.5
) -> pd.DataFrame:
    """Outer-join BTC price with selected on-chain CSVs on `time`.

    Returns a DataFrame sorted by `time` with columns `time`, `PriceUSD`,
    and any requested feature whose non-null coverage is >= `min_coverage`.
    No fill/shift is performed here; NaN handling and feature engineering
    (returns, targets, windows) are the caller's responsibility.
    """
    feature_columns = feature_columns or DEFAULT_FEATURES
    price = pd.read_csv(_PRICE_CSV)
    price["time"] = pd.to_datetime(price["time"])
    merged = price[["time", "PriceUSD"]].copy()

    for name in feature_columns:
        path = os.path.join(_ONCHAIN_DIR, f"{name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        if name not in df.columns and "value" in df.columns:
            df = df.rename(columns={"value": name})
        merged = pd.merge(merged, df[["time", name]], on="time", how="outer")

    merged = merged.sort_values("time").reset_index(drop=True)
    coverage = merged.notnull().mean()
    keep = [c for c in merged.columns if c in ("time", "PriceUSD") or coverage.get(c, 0) >= min_coverage]
    return merged[keep]
