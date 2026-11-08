"""Multi-asset daily/monthly price series (7 verified sources).

CLI: ``python -m experiments.multiasset --check``

Sources (verified 2026-10-29): NBP table A for EUR/USD vs PLN; Yahoo chart API for
WIG20, wheat/oil/gold futures; local ``api/csv/PriceUSD.csv`` for bitcoin (Coin Metrics
pipeline — not synthetic). Stooq.pl is intentionally not used (PoW block).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from experiments.config import ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = ROOT / "data" / "multiasset"
BITCOIN_CSV = ROOT / "api" / "csv" / "PriceUSD.csv"
NBP_BASE = "https://api.nbp.pl/api/exchangerates/rates/a"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart"
USER_AGENT = "btc-tai-indicators/multiasset"

SERIES_NAMES = (
    "eurpln",
    "usdpln",
    "wig20",
    "wheat",
    "oil",
    "gold",
    "bitcoin",
)

Freq = Literal["D", "M"]
Align = Literal["outer", "intersection"]


class TradingCalendar(str, Enum):
    """Which session calendar governs expected observation days."""

    NBP = "nbp"  # Polish FX fixing, business days
    GPW = "gpw"  # Warsaw Stock Exchange
    CME = "cme"  # US futures (approximated as US weekdays)
    CRYPTO = "crypto"  # 365d panel; gaps only on missing prints


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    calendar: TradingCalendar
    source: Literal["nbp", "yahoo", "local"]
    nbp_code: str | None = None
    yahoo_symbol: str | None = None
    local_path: Path | None = None


SPECS: dict[str, SeriesSpec] = {
    "eurpln": SeriesSpec("eurpln", TradingCalendar.NBP, "nbp", nbp_code="eur"),
    "usdpln": SeriesSpec("usdpln", TradingCalendar.NBP, "nbp", nbp_code="usd"),
    "wig20": SeriesSpec("wig20", TradingCalendar.GPW, "yahoo", yahoo_symbol="WIG20.WA"),
    "wheat": SeriesSpec("wheat", TradingCalendar.CME, "yahoo", yahoo_symbol="ZW=F"),
    "oil": SeriesSpec("oil", TradingCalendar.CME, "yahoo", yahoo_symbol="CL=F"),
    "gold": SeriesSpec("gold", TradingCalendar.CME, "yahoo", yahoo_symbol="GC=F"),
    "bitcoin": SeriesSpec("bitcoin", TradingCalendar.CRYPTO, "local", local_path=BITCOIN_CSV),
}


def _http_get_json(url: str, timeout: float = 60.0) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cache_paths(name: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{name}.csv", CACHE_DIR / f"{name}.parquet"


def _read_cached(name: str) -> pd.Series | None:
    _, pq = _cache_paths(name)
    csv_path, _ = _cache_paths(name)
    path = pq if pq.exists() else csv_path if csv_path.exists() else None
    if path is None:
        return None
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "time" not in df.columns or name not in df.columns:
        raise ValueError(f"cache {path} must have columns time, {name}")
    s = pd.Series(
        df[name].to_numpy(dtype=float),
        index=pd.to_datetime(df["time"]),
        name=name,
    )
    return _clean_series(s)


def _write_cache(name: str, s: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"time": s.index, name: s.values})
    csv_path, pq_path = _cache_paths(name)
    out.to_csv(csv_path, index=False)
    out.to_parquet(pq_path, index=False)


def _clean_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[s.index.notna()]
    dup = int(s.index.duplicated(keep="last").sum())
    if dup:
        logger.warning("%s: dropped %d duplicate dates (kept last)", s.name, dup)
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s


def validate_series(s: pd.Series, calendar: TradingCalendar | None = None) -> dict:
    """Return quality metrics: duplicates (post-clean), internal gaps on expected business days."""
    s = _clean_series(s)
    cal = calendar or SPECS[s.name].calendar if s.name in SPECS else TradingCalendar.CRYPTO
    if len(s) < 2:
        return {
            "name": s.name,
            "n_obs": len(s),
            "first": s.index.min() if len(s) else pd.NaT,
            "last": s.index.max() if len(s) else pd.NaT,
            "n_internal_gaps": 0,
            "calendar": cal.value,
        }
    expected = _expected_index(s.index.min(), s.index.max(), cal)
    observed = pd.DatetimeIndex(s.dropna().index.unique()).sort_values()
    missing = expected.difference(observed)
    # Weekends/holidays already excluded from expected; remaining are true gaps.
    return {
        "name": s.name,
        "n_obs": int(s.notna().sum()),
        "first": s.first_valid_index(),
        "last": s.last_valid_index(),
        "n_internal_gaps": int(len(missing)),
        "calendar": cal.value,
        "gap_examples": [d.date().isoformat() for d in missing[:5]],
    }


def _expected_index(start: pd.Timestamp, end: pd.Timestamp, cal: TradingCalendar) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if cal == TradingCalendar.CRYPTO:
        return pd.date_range(start, end, freq="D")
    days = pd.date_range(start, end, freq="B")  # Mon–Fri; NBP/GPW/CME approx
    if cal in (TradingCalendar.NBP, TradingCalendar.GPW):
        return days  # Polish holidays omitted — conservative gap detection
    return days


def _fetch_nbp(code: str, start: date | None = None, end: date | None = None) -> pd.Series:
    if start is None:
        url = f"{NBP_BASE}/{code}/?format=json"
    else:
        end_s = (end or date.today()).isoformat()
        url = f"{NBP_BASE}/{code}/{start.isoformat()}/{end_s}/?format=json"
    try:
        payload = _http_get_json(url)
    except HTTPError as e:
        if e.code == 404 and start is not None:
            payload = _http_get_json(f"{NBP_BASE}/{code}/{start.isoformat()}/?format=json")
        else:
            raise
    rates = payload.get("rates") or []
    if not rates:
        raise ValueError(f"NBP returned no rates for {code}")
    idx = pd.to_datetime([r["effectiveDate"] for r in rates])
    vals = [float(r["mid"]) for r in rates]
    return _clean_series(pd.Series(vals, index=idx, name=code))


def _yahoo_period1(ts: pd.Timestamp) -> int:
    return int(ts.tz_localize(timezone.utc).timestamp()) if ts.tzinfo is None else int(ts.timestamp())


def _fetch_yahoo(symbol: str, start: date | None = None, end: date | None = None) -> pd.Series:
    end_ts = pd.Timestamp(end or date.today()).normalize() + pd.Timedelta(days=1)
    if start is None:
        q = urlencode({"interval": "1d", "range": "max"})
    else:
        start_ts = pd.Timestamp(start).normalize()
        q = urlencode(
            {
                "interval": "1d",
                "period1": _yahoo_period1(start_ts),
                "period2": _yahoo_period1(end_ts),
            }
        )
    url = f"{YAHOO_CHART}/{symbol}?{q}"
    payload = _http_get_json(url)
    chart = (payload.get("chart") or {}).get("result")
    if not chart:
        err = (payload.get("chart") or {}).get("error") or payload
        raise ValueError(f"Yahoo chart empty for {symbol}: {err}")
    block = chart[0]
    ts = block.get("timestamp") or []
    quotes = (block.get("indicators") or {}).get("quote") or [{}]
    closes = quotes[0].get("close") or []
    rows: list[tuple[pd.Timestamp, float]] = []
    for t, c in zip(ts, closes):
        if t is None or c is None:
            continue
        rows.append((pd.to_datetime(t, unit="s").normalize(), float(c)))
    if not rows:
        raise ValueError(f"Yahoo chart has no closes for {symbol}")
    idx, vals = zip(*rows)
    name = symbol.replace("=", "_").replace(".", "_").lower()
    return _clean_series(pd.Series(vals, index=pd.DatetimeIndex(idx), name=name))


def _fetch_local(path: Path, name: str) -> pd.Series:
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != "time"][0]
    s = pd.Series(
        df[col].to_numpy(dtype=float),
        index=pd.to_datetime(df["time"], errors="coerce"),
        name=name,
    )
    return _clean_series(s)


def _incremental_start(cached: pd.Series | None, overlap_days: int = 7) -> date | None:
    if cached is None or cached.empty:
        return None
    last = cached.last_valid_index()
    if last is pd.NaT:
        return None
    return (last - pd.Timedelta(days=overlap_days)).date()


def fetch_series(name: str, refresh: bool = False) -> pd.Series:
    """Download (or incrementally update) one series and refresh cache."""
    if name not in SPECS:
        raise KeyError(f"unknown series {name!r}; expected one of {SERIES_NAMES}")
    spec = SPECS[name]
    cached = None if refresh else _read_cached(name)
    start = _incremental_start(cached)

    if spec.source == "nbp":
        assert spec.nbp_code
        try:
            chunk = _fetch_nbp(spec.nbp_code, start=start)
        except (HTTPError, URLError, TimeoutError) as e:
            if cached is not None:
                logger.warning("NBP fetch failed for %s (%s); using cache", name, e)
                return cached
            raise
        chunk.name = name
        s = _clean_series(pd.concat([cached, chunk]).sort_index()) if cached is not None else chunk
    elif spec.source == "yahoo":
        assert spec.yahoo_symbol
        try:
            chunk = _fetch_yahoo(spec.yahoo_symbol, start=start)
        except (HTTPError, URLError, TimeoutError, ValueError) as e:
            if cached is not None:
                logger.warning("Yahoo fetch failed for %s (%s); using cache", name, e)
                return cached
            raise
        chunk.name = name
        s = _clean_series(pd.concat([cached, chunk]).sort_index()) if cached is not None else chunk
    else:
        assert spec.local_path
        s = _fetch_local(spec.local_path, name)
        if cached is not None and len(s) <= len(cached):
            s = cached
    s.name = name
    _write_cache(name, s)
    return s


def _to_freq(s: pd.Series, freq: Freq) -> pd.Series:
    if freq == "D":
        return s
    out = s.resample("ME").last()
    out.name = s.name
    return out


def load_series(
    name: str,
    freq: Freq = "D",
    refresh: bool = False,
    use_cache_only: bool = False,
) -> pd.Series:
    """Load one series at daily (``D``) or month-end (``M``) frequency."""
    if name not in SPECS:
        raise KeyError(name)
    if use_cache_only:
        cached = _read_cached(name)
        if cached is None:
            raise FileNotFoundError(
                f"no cache for {name}; run fetch or python -m experiments.multiasset --check"
            )
        s = cached
    else:
        s = fetch_series(name, refresh=refresh)
    return _to_freq(s, freq)


def load_multiasset(
    names: Iterable[str] | None = None,
    freq: Freq = "D",
    align: Align = "outer",
    refresh: bool = False,
    use_cache_only: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load multiple series on a common time index (outer or intersection join)."""
    keys = tuple(names) if names is not None else SERIES_NAMES
    series = {
        n: load_series(n, freq=freq, refresh=refresh, use_cache_only=use_cache_only) for n in keys
    }
    panel = pd.concat(series, axis=1, sort=True)
    if align == "intersection":
        panel = panel.dropna(how="any")
    if start is not None:
        panel = panel.loc[pd.Timestamp(start):]
    if end is not None:
        panel = panel.loc[:pd.Timestamp(end)]
    panel.index.name = "time"
    return panel


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-column first/last valid date, n obs and % missing over the panel index."""
    rows = []
    for c in panel.columns:
        s = panel[c]
        rows.append(
            {
                "column": c,
                "calendar": SPECS[c].calendar.value if c in SPECS else "",
                "first": s.first_valid_index(),
                "last": s.last_valid_index(),
                "n_obs": int(s.notna().sum()),
                "pct_missing": 100.0 * float(s.isna().mean()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def calendar_alignment_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarise calendar mix and pairwise overlap (business-day alignment context)."""
    rows = []
    for c in panel.columns:
        s = panel[c].dropna()
        q = validate_series(s.rename(c), SPECS[c].calendar if c in SPECS else None)
        rows.append(
            {
                "column": c,
                "calendar": q["calendar"],
                "first": q["first"],
                "last": q["last"],
                "n_obs": q["n_obs"],
                "n_internal_gaps": q["n_internal_gaps"],
            }
        )
    rep = pd.DataFrame(rows).set_index("column")
    if panel.shape[1] >= 2:
        overlap = int(panel.dropna(how="any").shape[0])
        rep.attrs["intersection_rows"] = overlap
    return rep


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Multi-asset loader (7 verified series)")
    ap.add_argument("--check", action="store_true", help="fetch/cache all series and print reports")
    ap.add_argument("--refresh", action="store_true", help="full re-download (not incremental)")
    ap.add_argument("--cache-only", action="store_true", help="read data/multiasset only")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.check:
        ap.print_help()
        return

    panel = load_multiasset(refresh=args.refresh, use_cache_only=args.cache_only)
    cov = coverage_report(panel)
    cal = calendar_alignment_report(panel)
    print(f"series     : {', '.join(panel.columns)}")
    print(f"index      : {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)} rows, align=outer)")
    print(f"intersection rows (all non-null): {cal.attrs.get('intersection_rows', 'n/a')}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\nCOVERAGE:")
        print(
            cov.assign(first=cov["first"].dt.date, last=cov["last"].dt.date)
            .round(1)
            .to_string()
        )
        print("\nCALENDAR / GAPS:")
        print(
            cal.assign(first=cal["first"].dt.date, last=cal["last"].dt.date).to_string()
        )
    for name in panel.columns:
        q = validate_series(panel[name].dropna().rename(name))
        if q["n_internal_gaps"]:
            logger.info(
                "%s: %d gap(s) on %s calendar (examples %s)",
                name,
                q["n_internal_gaps"],
                q["calendar"],
                q.get("gap_examples"),
            )


if __name__ == "__main__":
    main()
