"""Data layer: merged daily panel of on-chain metrics + BTC price.

CLI: ``python -m experiments.data --check``
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from experiments.config import ROOT, load_config

logger = logging.getLogger(__name__)

ONCHAIN_DIR = ROOT / "data_ingestion" / "csv"
PRICE_CSV = ROOT / "api" / "csv" / "PriceUSD.csv"
PANEL_PARQUET = ROOT / "data" / "panel.parquet"
ANN = 365
# Several series (NVT*, TxTfrVal*, UTXOCnt, ShareExUSD, BlkSizeMeanByte) are sampled every
# ~4 days; a causal forward-fill of at most this many days makes them usable as daily features.
SPARSE_FFILL = 7

# Columns shipped in data_ingestion/csv that are derived from price. They are kept in the
# panel (nothing is dropped silently) but never used as features.
#   dir_t1 == 1[logret_{t+1} > 0]  -> FORWARD-LOOKING LABEL (verified: 100% sign agreement
#   with next-day return). r_t / rv_* duplicate our own causal logret_1 / rv_*.
PRICE_DERIVED_COLUMNS = frozenset({"dir_t1", "r_t", "rv_t", "rv_7d", "rv_30d"})
FORWARD_LOOKING_COLUMNS = frozenset({"dir_t1"})
# Raw price-level columns: non-stationary, excluded from features (log_price kept in panel).
PRICE_LEVEL_COLUMNS = frozenset({"PriceUSD", "PriceBTC", "log_price"})

# Level-type metrics (positive, growing, heavy tailed) -> signed log1p transform.
LEVEL_COLUMNS = (
    "AdrActCnt AdrBalCnt BlkCnt BlkSizeMeanByte CDD CapFutExp10yrUSD CapMrktCurUSD CapRealUSD "
    "DiffMean FeeMeanNtv FeeMeanUSD FeeMedNtv FeeMedUSD FeeTotNtv FeeTotUSD FlowInExNtv "
    "FlowInExUSD FlowOutExNtv FlowOutExUSD HashRate IssTotNtv IssTotUSD NetFlowNtv NetFlowUSD "
    "RealizedPrice RevAllTimeUSD RevUSD SplyCur SplyExpFut10yr TxCnt TxCntSec TxMeanByte TxTfrCnt "
    "TxTfrValMeanNtv TxTfrValMeanUSD TxTfrValNtv TxTfrValUSD UTXOCnt CapAct1yrUSD SplyAct180d "
    "SplyAct1yr SplyAct30d"
).split()


def signed_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Monotone transform sign(x)*log1p(|x|); equals log1p(x) for x >= 0."""
    return np.sign(x) * np.log1p(np.abs(x))


def _read_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    value_col = [c for c in df.columns if c != "time"]
    s = pd.Series(
        df[value_col[0]].to_numpy(dtype=float) if value_col else np.array([], dtype=float),
        index=pd.to_datetime(df["time"], errors="coerce"),
        name=path.stem,
    )
    s = s[s.index.notna()]
    return s[~s.index.duplicated(keep="last")]


def load_panel(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Outer-join all on-chain CSVs and PriceUSD on a daily DatetimeIndex.

    Derived columns: ``logret_1``, ``rv_7``, ``rv_30`` (annualised rolling std of
    ``logret_1``, trailing windows only), ``log_price`` and ``slog1p_<metric>`` for every
    level-type metric in ``LEVEL_COLUMNS`` that is present. Every column is causal: value
    at row t uses data up to and including t.
    """
    series: dict[str, pd.Series] = {}
    for path in sorted(ONCHAIN_DIR.glob("*.csv")):
        series[path.stem] = _read_series(path)
    price = _read_series(PRICE_CSV)
    price.name = "PriceUSD"
    series["PriceUSD"] = price  # api/csv copy is identical to data_ingestion/csv/PriceUSD.csv

    panel = pd.concat(series, axis=1, sort=True)
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="D"))
    panel.index.name = "time"

    p = panel["PriceUSD"].where(panel["PriceUSD"] > 0)
    panel["log_price"] = np.log(p)
    panel["logret_1"] = panel["log_price"].diff()
    r = panel["logret_1"]
    panel["rv_7"] = r.rolling(7, min_periods=7).std() * np.sqrt(ANN)
    panel["rv_30"] = r.rolling(30, min_periods=30).std() * np.sqrt(ANN)
    lv = {f"slog1p_{c}": signed_log1p(panel[c]) for c in LEVEL_COLUMNS if c in panel.columns}
    panel = pd.concat([panel, pd.DataFrame(lv, index=panel.index)], axis=1)

    if start is not None:
        panel = panel.loc[pd.Timestamp(start):]
    if end is not None:
        panel = panel.loc[:pd.Timestamp(end)]
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    return panel


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-column first/last valid date, n obs and % missing (over panel's full index)."""
    rows = []
    for c in panel.columns:
        s = panel[c]
        rows.append({
            "column": c,
            "first": s.first_valid_index(),
            "last": s.last_valid_index(),
            "n_obs": int(s.notna().sum()),
            "pct_missing": 100.0 * float(s.isna().mean()),
        })
    return pd.DataFrame(rows).set_index("column")


def forward_log_return(log_price: pd.Series, h: int) -> pd.Series:
    """y_t = log P_{t+h} - log P_t  (uses prices strictly AFTER t; label only, never a feature)."""
    return (log_price.shift(-h) - log_price).rename(f"logret_fwd_{h}")


def detect_forward_looking_columns(
    panel: pd.DataFrame, max_lead: int = 7, agree_thr: float = 0.95, rho_thr: float = 0.5
) -> dict[str, str]:
    """Flag columns that (almost) encode future returns.

    A column is flagged if, for some lead k in 1..max_lead, its sign agrees with the sign of
    logret_{t+k} on >= ``agree_thr`` of rows (binary label leaks) or its Spearman correlation
    with logret_{t+k} exceeds ``rho_thr``. Contemporaneous/lagged price-linked series stay
    below these thresholds.
    """
    r = panel["logret_1"]
    flagged: dict[str, str] = {}
    for c in panel.columns:
        if c in ("logret_1", "log_price", "PriceUSD", "PriceBTC"):
            continue
        s = panel[c]
        for k in range(1, max_lead + 1):
            fut = r.shift(-k)
            m = s.notna() & fut.notna() & (fut != 0)
            if m.sum() < 200:
                continue
            agree = float(((s[m] > 0) == (fut[m] > 0)).mean())
            rho = float(s[m].corr(fut[m], method="spearman")) if s[m].nunique() > 2 else 0.0
            if max(agree, 1 - agree) >= agree_thr and s[m].nunique() <= 2:
                flagged[c] = f"sign agreement {agree:.3f} with logret_(t+{k})"
                break
            if abs(rho) >= rho_thr:
                flagged[c] = f"spearman {rho:.2f} with logret_(t+{k})"
                break
    return flagged


def build_features(panel: pd.DataFrame, min_obs: int | None = None) -> pd.DataFrame:
    """Default causal feature set (all rows t use info <= t).

    * lagged/cumulative returns (1,2,3,5,7,14,30d), rv_7, rv_30
    * level-type on-chain metrics: 7d and 30d change of signed-log1p level
    * ratio-type metrics (MVRV, NUPL, SOPR, ...): raw value and 7d change
    Sparse (~4-day) series are causally forward-filled (<= SPARSE_FFILL days). Columns with fewer than ``min_obs`` non-null rows (e.g. empty / single-row CSVs) are skipped
    and logged; NaNs are left for the model to handle (XGBoost handles them natively).
    """
    min_obs = load_config()["min_feature_obs"] if min_obs is None else min_obs
    r = panel["logret_1"]
    feats: dict[str, pd.Series] = {}
    for k in (1, 2, 3, 5, 7, 14, 30):
        feats[f"ret_cum_{k}"] = r.rolling(k, min_periods=k).sum()
    feats["rv_7"] = panel["rv_7"]
    feats["rv_30"] = panel["rv_30"]
    skip = set(PRICE_DERIVED_COLUMNS) | set(PRICE_LEVEL_COLUMNS) | {"logret_1", "rv_7", "rv_30"}
    for c in panel.columns:
        if c in skip or c in LEVEL_COLUMNS:
            continue
        if c.startswith("slog1p_"):
            base = panel[c].ffill(limit=SPARSE_FFILL)
            if base.notna().sum() < min_obs:
                logger.warning("skipping %s (%d obs)", c, base.notna().sum())
                continue
            feats[f"{c}_d7"] = base.diff(7)
            feats[f"{c}_d30"] = base.diff(30)
        else:
            s = panel[c].ffill(limit=SPARSE_FFILL)
            if s.notna().sum() < min_obs:
                logger.warning("skipping %s (%d obs)", c, s.notna().sum())
                continue
            feats[c] = s
            feats[f"{c}_d7"] = s.diff(7)
    return pd.DataFrame(feats, index=panel.index)


FeatureSpec = Sequence[str] | Callable[[pd.DataFrame], pd.DataFrame] | None


def make_supervised(
    panel: pd.DataFrame,
    target: str = "logret_fwd_h",
    h: int = 1,
    features: FeatureSpec = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for horizon ``h``.

    Target construction: ``y_t = log(P_{t+h}) - log(P_t)`` = sum of the next h daily log
    returns, i.e. it is computed with ``log_price.shift(-h)`` and therefore uses ONLY prices
    strictly after day t. X is built exclusively from causal columns at row t (information
    available at the close of day t); the target column and any forward-looking column
    (see ``FORWARD_LOOKING_COLUMNS``) are never part of X (asserted). Rows where y is NaN
    (last h days, missing price) are dropped; X may still contain NaNs.

    ``features``: None -> ``build_features``; list of column names; or callable(panel)->DataFrame.
    ``target`` must be ``'logret_fwd_h'`` (resolved to ``logret_fwd_{h}``) or ``logret_fwd_{h}``.
    """
    if target not in ("logret_fwd_h", f"logret_fwd_{h}"):
        raise ValueError(f"unsupported target {target!r}")
    y = forward_log_return(panel["log_price"], h)
    if features is None:
        X = build_features(panel)
    elif callable(features):
        X = features(panel)
    else:
        X = panel[list(features)].copy()
    bad = [c for c in X.columns if c.startswith("logret_fwd") or c in FORWARD_LOOKING_COLUMNS]
    if bad:
        raise ValueError(f"forward-looking columns in features: {bad}")
    assert X.index.equals(panel.index) and X.index.is_monotonic_increasing
    m = y.notna()
    return X.loc[m], y.loc[m]


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print coverage and write data/panel.parquet")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel = load_panel()
    if args.check:
        cov = coverage_report(panel)
        print(f"date range : {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)} days)")
        print(f"n columns  : {panel.shape[1]}")
        n_files = len(list(ONCHAIN_DIR.glob('*.csv')))
        print(f"source csv : {n_files} on-chain files + api/csv/PriceUSD.csv")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(cov.assign(first=cov["first"].dt.date, last=cov["last"].dt.date).round(1).to_string())
        short = cov[(cov["n_obs"] < 500)]
        print("\nSHORT/EMPTY series (<500 obs, kept in panel, excluded from features):",
              {c: int(n) for c, n in short["n_obs"].items()})
        fl = detect_forward_looking_columns(panel)
        print("FORWARD-LOOKING columns detected:", fl or "none")
        PANEL_PARQUET.parent.mkdir(exist_ok=True)
        panel.to_parquet(PANEL_PARQUET)
        print(f"wrote {PANEL_PARQUET} (not committed; ignored via data/.gitignore)")


if __name__ == "__main__":
    main()
