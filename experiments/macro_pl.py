"""Polish macro / banking series loader (SGH TRD-29).

CLI: ``python -m experiments.macro_pl --check``

Cache: ``data/macro_pl/{name}.csv`` (+ parquet). Raw downloads: ``data/macro_pl/raw/``.
See ``data/macro_pl/SPEC.yaml`` and ``data/macro_pl/SOURCES.md`` for licenses and update cadence.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from experiments.config import ROOT
from experiments.multiasset import TradingCalendar, _clean_series, validate_series

logger = logging.getLogger(__name__)

CACHE_DIR = ROOT / "data" / "macro_pl"
RAW_DIR = CACHE_DIR / "raw"
META_PATH = CACHE_DIR / "meta.json"
USER_AGENT = "btc-tai-indicators/macro_pl"

NBP_INTEREST = "https://api.nbp.pl/api/interestrates"
MF_YIELD_CURVE_PAGE = "https://www.gov.pl/web/finanse/krajowa-krzywa-dochodowosci"
NBP_CREDITS_XLS_DEFAULT = (
    "https://www.nbp.pl/home.aspx?f=/statystyka/sektor_pieniezny/kredyty/kredyty.xlsx"
)
IZFIA_STATS_PAGE = "https://www.izfia.pl/statystyki"

Freq = Literal["D", "M"]
Align = Literal["outer", "intersection"]

SERIES_NAMES = (
    "nbp_rate",
    "gus_cpi_yoy",
    "gus_pmi",
    "bond_pl_2y",
    "bond_pl_5y",
    "bond_pl_7y",
    "bond_pl_10y",
    "credit_consumer",
    "credit_mortgage",
    "credit_corporate",
    "deposits_households",
    "tfi_assets",
)

BOND_SERIES = ("bond_pl_2y", "bond_pl_5y", "bond_pl_7y", "bond_pl_10y")
GUS_SERIES = ("gus_cpi_yoy", "gus_pmi")
CREDIT_SERIES = (
    "credit_consumer",
    "credit_mortgage",
    "credit_corporate",
    "deposits_households",
)


class SourceStatus(str, Enum):
    VERIFIED = "verified"
    NEEDS_VERIFICATION = "do_weryfikacji"
    MANUAL_ONLY = "manual_only"
    PAID_OR_CONTRACT = "paid_or_contract"


@dataclass(frozen=True)
class MacroSeriesSpec:
    name: str
    native_freq: Freq
    calendar: TradingCalendar
    status: SourceStatus
    license_note: str
    update_cadence: str


SPECS: dict[str, MacroSeriesSpec] = {
    "nbp_rate": MacroSeriesSpec(
        "nbp_rate",
        "D",
        TradingCalendar.NBP,
        SourceStatus.NEEDS_VERIFICATION,
        "NBP API — public information",
        "RPP decisions (irregular)",
    ),
    "gus_cpi_yoy": MacroSeriesSpec(
        "gus_cpi_yoy",
        "M",
        TradingCalendar.NBP,
        SourceStatus.NEEDS_VERIFICATION,
        "GUS BDL / manual CSV — CPI inflation y/y %",
        "monthly",
    ),
    "gus_pmi": MacroSeriesSpec(
        "gus_pmi",
        "M",
        TradingCalendar.NBP,
        SourceStatus.NEEDS_VERIFICATION,
        "GUS BDL / manual CSV — PMI manufacturing",
        "monthly",
    ),
    **{
        k: MacroSeriesSpec(
            k,
            "D",
            TradingCalendar.NBP,
            SourceStatus.NEEDS_VERIFICATION,
            "MF gov.pl public data; GPW Benchmark SPW is commercial",
            "daily business",
        )
        for k in BOND_SERIES
    },
    **{
        k: MacroSeriesSpec(
            k,
            "M",
            TradingCalendar.NBP,
            SourceStatus.NEEDS_VERIFICATION,
            "NBP monetary statistics (nbportal)",
            "monthly",
        )
        for k in (*CREDIT_SERIES, "tfi_assets")
    },
}

TENOR_TO_SERIES = {2: "bond_pl_2y", 5: "bond_pl_5y", 7: "bond_pl_7y", 10: "bond_pl_10y"}


def _http_get_bytes(url: str, timeout: float = 90.0) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, timeout: float = 60.0) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cache_paths(name: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{name}.csv", CACHE_DIR / f"{name}.parquet"


def _read_cached(name: str) -> pd.Series | None:
    csv_path, pq_path = _cache_paths(name)
    path = pq_path if pq_path.exists() else csv_path if csv_path.exists() else None
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


def _write_raw(filename: str, data: bytes) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    path.write_bytes(data)
    return path


def validate_macro_series(s: pd.Series, spec: MacroSeriesSpec | None = None) -> dict:
    s = _clean_series(s)
    spec = spec or SPECS.get(s.name)
    if spec is None or spec.native_freq == "D":
        cal = spec.calendar if spec else TradingCalendar.NBP
        return validate_series(s, cal)
    if len(s) < 2:
        return {
            "name": s.name,
            "n_obs": len(s),
            "first": s.index.min() if len(s) else pd.NaT,
            "last": s.index.max() if len(s) else pd.NaT,
            "n_internal_gaps": 0,
            "calendar": "monthly",
        }
    start, end = s.index.min(), s.index.max()
    expected = pd.date_range(start, end, freq="ME")
    observed = pd.DatetimeIndex(s.dropna().index.unique()).sort_values()
    missing = expected.difference(observed)
    return {
        "name": s.name,
        "n_obs": int(s.notna().sum()),
        "first": s.first_valid_index(),
        "last": s.last_valid_index(),
        "n_internal_gaps": int(len(missing)),
        "calendar": "monthly",
        "gap_examples": [d.date().isoformat() for d in missing[:5]],
    }


def _expand_step_to_business_days(events: pd.Series) -> pd.Series:
    events = events.sort_index()
    start, end = events.index.min(), events.index.max()
    idx = pd.date_range(start, end, freq="B")
    daily = events.reindex(idx, method="ffill")
    daily.name = events.name
    return _clean_series(daily)


def fetch_nbp_reference_rate(refresh: bool = False) -> pd.Series:
    cached = None if refresh else _read_cached("nbp_rate")
    url = f"{NBP_INTEREST}/?format=json"
    try:
        payload = _http_get_json(url)
    except (HTTPError, URLError, TimeoutError) as e:
        if cached is not None:
            logger.warning("NBP interest API failed (%s); using cache", e)
            return cached
        raise
    rates = payload.get("rates") or []
    if not rates:
        raise ValueError("NBP interestrates returned empty rates")
    _write_raw("nbp_interestrates.json", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    idx = pd.to_datetime([r["effectiveDate"] for r in rates])
    vals = []
    for r in rates:
        v = r.get("referenceRate")
        if v is None:
            v = r.get("refRate") or r.get("stopaReferencyjna")
        if v is None:
            raise KeyError(f"NBP rate row missing referenceRate keys: {list(r.keys())}")
        vals.append(float(v))
    events = pd.Series(vals, index=idx, name="nbp_rate")
    s = _expand_step_to_business_days(events)
    _write_cache("nbp_rate", s)
    return s


GUS_MACRO_CSV = RAW_DIR / "gus_macro.csv"
GUS_BDL_BASE = "https://bdl.stat.gov.pl/api/v1/data/by-variable"
GUS_BDL_CPI_VAR = 64428
GUS_BDL_PMI_VAR = 339055


def _parse_gus_macro_csv(path: Path) -> dict[str, pd.Series]:
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in ("time", "data", "date", "okres")), df.columns[0])
    cpi_col = next(
        (c for c in df.columns if c in ("gus_cpi_yoy",) or "cpi" in c.lower()),
        None,
    )
    pmi_col = next(
        (c for c in df.columns if c in ("gus_pmi",) or "pmi" in c.lower()),
        None,
    )
    if cpi_col is None or pmi_col is None:
        raise ValueError(f"gus_macro.csv needs CPI and PMI columns; got {list(df.columns)}")
    idx = pd.to_datetime(df[time_col], errors="coerce")
    out = {
        "gus_cpi_yoy": pd.Series(pd.to_numeric(df[cpi_col], errors="coerce").values, index=idx, name="gus_cpi_yoy"),
        "gus_pmi": pd.Series(pd.to_numeric(df[pmi_col], errors="coerce").values, index=idx, name="gus_pmi"),
    }
    for k in out:
        out[k] = _clean_series(out[k])
    return out


def _fetch_gus_bdl_variable(var_id: int) -> pd.Series:
    """Monthly observations from GUS BDL (year-month granularity)."""
    url = f"{GUS_BDL_BASE}/{var_id}?format=json&page-size=500"
    payload = _http_get_json(url)
    rows = payload.get("results") or []
    if not rows:
        raise ValueError(f"BDL variable {var_id} returned no rows")
    idx, vals = [], []
    for row in rows:
        y, m = int(row["year"]), int(row["period"])
        idx.append(pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0))
        vals.append(float(row["value"]))
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=str(var_id))
    return _clean_series(s)


def fetch_gus_macro(refresh: bool = False) -> dict[str, pd.Series]:
    cached_cpi = None if refresh else _read_cached("gus_cpi_yoy")
    cached_pmi = None if refresh else _read_cached("gus_pmi")
    if cached_cpi is not None and cached_pmi is not None:
        return {"gus_cpi_yoy": cached_cpi, "gus_pmi": cached_pmi}

    if GUS_MACRO_CSV.exists():
        parsed = _parse_gus_macro_csv(GUS_MACRO_CSV)
        _write_cache("gus_cpi_yoy", parsed["gus_cpi_yoy"])
        _write_cache("gus_pmi", parsed["gus_pmi"])
        _update_meta("gus_macro", {"path": str(GUS_MACRO_CSV), "status": SourceStatus.NEEDS_VERIFICATION.value})
        return parsed

    try:
        cpi = _fetch_gus_bdl_variable(GUS_BDL_CPI_VAR)
        pmi = _fetch_gus_bdl_variable(GUS_BDL_PMI_VAR)
        cpi.name, pmi.name = "gus_cpi_yoy", "gus_pmi"
        _write_cache("gus_cpi_yoy", cpi)
        _write_cache("gus_pmi", pmi)
        _update_meta("gus_macro", {"source": "bdl", "status": SourceStatus.NEEDS_VERIFICATION.value})
        return {"gus_cpi_yoy": cpi, "gus_pmi": pmi}
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as e:
        if cached_cpi is not None and cached_pmi is not None:
            logger.warning("GUS fetch failed (%s); using cache", e)
            return {"gus_cpi_yoy": cached_cpi, "gus_pmi": cached_pmi}
        raise FileNotFoundError(
            f"GUS CPI/PMI unavailable ({e}). Place monthly file at {GUS_MACRO_CSV} "
            "(columns: time, gus_cpi_yoy, gus_pmi) or refresh when BDL is reachable."
        ) from e


def _discover_attachment_url(page_url: str, pattern: re.Pattern[str]) -> str | None:
    html = _http_get_bytes(page_url).decode("utf-8", errors="replace")
    m = pattern.search(html)
    return m.group(1) if m else None


def _parse_mf_yield_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in ("data", "date", "dzien", "day")), df.columns[0])

    def _find_tenor(years: int) -> str | None:
        for key, orig in cols.items():
            if re.search(rf"\b{years}\b", key) or f"{years}y" in key or f"{years}l" in key:
                return orig
        return None

    mapping = {years: _find_tenor(years) for years in (2, 5, 7, 10)}
    if not all(mapping.values()):
        numeric = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) >= 4:
            ordered = numeric[:4]
            mapping = {2: ordered[0], 5: ordered[1], 7: ordered[2], 10: ordered[3]}
        else:
            raise ValueError(f"cannot map tenors in MF CSV columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df[date_col], errors="coerce")
    for years, col in mapping.items():
        out[f"y{years}"] = pd.to_numeric(df[col], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")
    return out


def fetch_bonds_pl_yields(refresh: bool = False) -> dict[str, pd.Series]:
    manual = RAW_DIR / "mf_yield_curve.csv"
    if manual.exists():
        text = manual.read_text(encoding="utf-8", errors="replace")
        panel = _parse_mf_yield_csv(text)
        source_note = "raw/mf_yield_curve.csv"
    else:
        csv_url = _discover_attachment_url(
            MF_YIELD_CURVE_PAGE,
            re.compile(r'href="(https://[^"]+\.csv[^"]*)"', re.I),
        )
        if not csv_url:
            cached = {n: _read_cached(n) for n in BOND_SERIES}
            if not refresh and all(v is not None for v in cached.values()):
                return {k: v for k, v in cached.items() if v is not None}
            raise FileNotFoundError(
                "MF yield CSV link not found on gov.pl page; place file at "
                f"{manual}. GPW Benchmark SPW requires a license."
            )
        raw = _http_get_bytes(csv_url)
        _write_raw("mf_yield_curve.csv", raw)
        panel = _parse_mf_yield_csv(raw.decode("utf-8", errors="replace"))
        source_note = csv_url

    result: dict[str, pd.Series] = {}
    for years, name in TENOR_TO_SERIES.items():
        col = f"y{years}"
        s = pd.Series(panel[col].to_numpy(dtype=float), index=panel["time"], name=name)
        s = _clean_series(s)
        _write_cache(name, s)
        result[name] = s
    _update_meta("bonds_mf", {"source": source_note, "status": SourceStatus.NEEDS_VERIFICATION.value})
    return result


def _read_excel_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except ImportError as e:
        raise ImportError("pip install openpyxl to parse NBP/IZFiA XLSX") from e


def _parse_nbp_credits_xlsx(path: Path) -> dict[str, pd.Series]:
    df = _read_excel_table(path)
    df = df.dropna(how="all")
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.notna().sum() < 3:
        for c in df.columns:
            trial = pd.to_datetime(df[c], errors="coerce")
            if trial.notna().sum() >= 3:
                date_col = c
                dates = trial
                break
    df = df.assign(_time=dates).dropna(subset=["_time"]).sort_values("_time")

    def _col(pattern: str) -> str | None:
        for c in df.columns:
            if c == date_col or c == "_time":
                continue
            label = str(c).lower()
            if re.search(pattern, label):
                return c
        return None

    mapping = {
        "credit_consumer": _col(r"konsum|consumer|gospod.*kons"),
        "credit_mortgage": _col(r"mieszk|hipot|housing|mortgage"),
        "credit_corporate": _col(r"przedsi|firm|corporate|non.?fin"),
        "deposits_households": _col(r"depoz.*gosp|household.*dep|dep.*dom"),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(
            f"NBP XLS columns not matched for {missing}; headers={list(df.columns)[:12]}"
        )
    out: dict[str, pd.Series] = {}
    for name, col in mapping.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        idx = pd.DatetimeIndex(df["_time"]).to_period("M").to_timestamp("M")
        s = pd.Series(vals.to_numpy(dtype=float), index=idx, name=name)
        out[name] = _clean_series(s)
    return out


def fetch_nbp_credits_deposits(refresh: bool = False) -> dict[str, pd.Series]:
    path = RAW_DIR / "nbp_credits_deposits.xlsx"
    if not path.exists():
        try:
            raw = _http_get_bytes(NBP_CREDITS_XLS_DEFAULT)
            _write_raw("nbp_credits_deposits.xlsx", raw)
        except (HTTPError, URLError, TimeoutError) as e:
            cached = {n: _read_cached(n) for n in CREDIT_SERIES}
            if not refresh and all(v is not None for v in cached.values()):
                logger.warning("NBP credits XLS download failed (%s); using cache", e)
                return {k: v for k, v in cached.items() if v is not None}
            raise FileNotFoundError(
                f"Download failed ({e}); place bulletin at {path} (see SOURCES.md)"
            ) from e
    series_map = _parse_nbp_credits_xlsx(path)
    for name, s in series_map.items():
        _write_cache(name, s)
    _update_meta("nbp_monetary", {"path": str(path), "status": SourceStatus.NEEDS_VERIFICATION.value})
    return series_map


def _parse_izfia_tfi_xlsx(path: Path) -> pd.Series:
    df = _read_excel_table(path)
    df = df.dropna(how="all")
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    val_col = None
    for c in df.columns[1:]:
        label = str(c).lower()
        if "aktywa" in label or "net" in label or "tfi" in label:
            val_col = c
            break
    if val_col is None:
        val_col = df.columns[1]
    frame = df.assign(_time=dates).dropna(subset=["_time"])
    idx = pd.DatetimeIndex(frame["_time"]).to_period("M").to_timestamp("M")
    s = pd.Series(pd.to_numeric(frame[val_col], errors="coerce"), index=idx, name="tfi_assets")
    return _clean_series(s)


def fetch_tfi_assets(refresh: bool = False) -> pd.Series:
    path = RAW_DIR / "izfia_tfi.xlsx"
    if not path.exists():
        url = _discover_attachment_url(
            IZFIA_STATS_PAGE,
            re.compile(r'href="([^"]+\.xlsx[^"]*)"', re.I),
        )
        if url and not url.startswith("http"):
            url = "https://www.izfia.pl/" + url.lstrip("/")
        if not url:
            cached = _read_cached("tfi_assets")
            if cached is not None:
                return cached
            raise FileNotFoundError(
                f"No IZFiA XLS link; place statistics file at {path}. "
                "Analizy.pl is commercial — manual export only."
            )
        raw = _http_get_bytes(url)
        _write_raw("izfia_tfi.xlsx", raw)
    s = _parse_izfia_tfi_xlsx(RAW_DIR / "izfia_tfi.xlsx")
    _write_cache("tfi_assets", s)
    _update_meta("tfi_assets", {"status": SourceStatus.NEEDS_VERIFICATION.value})
    return s


def _update_meta(key: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    meta[key] = {**payload, "updated": datetime.now(timezone.utc).isoformat()}
    meta["verified_at"] = None
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_series(name: str, refresh: bool = False) -> pd.Series:
    if name not in SPECS:
        raise KeyError(f"unknown series {name!r}; expected one of {SERIES_NAMES}")
    if name == "nbp_rate":
        return fetch_nbp_reference_rate(refresh=refresh)
    if name in GUS_SERIES:
        return fetch_gus_macro(refresh=refresh)[name]
    if name in BOND_SERIES:
        return fetch_bonds_pl_yields(refresh=refresh)[name]
    if name in CREDIT_SERIES:
        return fetch_nbp_credits_deposits(refresh=refresh)[name]
    if name == "tfi_assets":
        return fetch_tfi_assets(refresh=refresh)
    raise KeyError(name)


def _to_freq(s: pd.Series, freq: Freq) -> pd.Series:
    spec = SPECS[s.name]
    if freq == spec.native_freq:
        return s
    if freq == "M" and spec.native_freq == "D":
        out = s.resample("ME").last()
        out.name = s.name
        return out
    if freq == "D" and spec.native_freq == "M":
        raise ValueError(f"{s.name} is monthly-native; cannot upsample to daily")
    return s


def load_series(
    name: str,
    freq: Freq | None = None,
    refresh: bool = False,
    use_cache_only: bool = False,
) -> pd.Series:
    spec = SPECS[name]
    target_freq = freq or spec.native_freq
    if use_cache_only:
        cached = _read_cached(name)
        if cached is None:
            raise FileNotFoundError(
                f"no cache for {name}; run python -m experiments.macro_pl --check"
            )
        return _to_freq(cached, target_freq)
    s = fetch_series(name, refresh=refresh)
    return _to_freq(s, target_freq)


def load_macro_panel(
    names: Iterable[str] | None = None,
    freq: Freq | None = None,
    align: Align = "outer",
    refresh: bool = False,
    use_cache_only: bool = False,
) -> pd.DataFrame:
    keys = tuple(names) if names is not None else SERIES_NAMES
    series = {}
    for n in keys:
        f = freq or SPECS[n].native_freq
        series[n] = load_series(n, freq=f, refresh=refresh, use_cache_only=use_cache_only)
    panel = pd.concat(series, axis=1, sort=True)
    if align == "intersection":
        panel = panel.dropna(how="any")
    panel.index.name = "time"
    return panel


def source_report() -> pd.DataFrame:
    rows = []
    for name, spec in SPECS.items():
        cached = _read_cached(name)
        rows.append(
            {
                "series": name,
                "status": spec.status.value,
                "native_freq": spec.native_freq,
                "update": spec.update_cadence,
                "license": spec.license_note,
                "cached": cached is not None,
                "n_obs": len(cached) if cached is not None else 0,
            }
        )
    return pd.DataFrame(rows).set_index("series")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Polish macro/banking loader (TRD-29)")
    ap.add_argument("--check", action="store_true", help="fetch/cache all series and print report")
    ap.add_argument("--refresh", action="store_true", help="force re-download")
    ap.add_argument("--cache-only", action="store_true", help="read data/macro_pl only")
    ap.add_argument("--series", action="append", help="limit to one series (repeatable)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.check:
        ap.print_help()
        return

    names = tuple(args.series) if args.series else SERIES_NAMES
    errors: list[str] = []
    for name in names:
        try:
            load_series(name, refresh=args.refresh, use_cache_only=args.cache_only)
            logger.info("OK %s", name)
        except Exception as e:
            errors.append(f"{name}: {e}")
            logger.error("FAIL %s: %s", name, e)

    rep = source_report()
    print(rep.to_string())
    if errors:
        print("\nERRORS:")
        for line in errors:
            print(" -", line)
        print("\nGPW Benchmark SPW and Analizy.pl require paid/contract access — see SOURCES.md.")
    else:
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            meta["verified_at"] = datetime.now(timezone.utc).isoformat()
            META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
