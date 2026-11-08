"""Unit tests for experiments.macro_pl (TRD-29) with fixtures — no synthetic production data."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments import macro_pl as mp


NBP_INTEREST_FIXTURE = {
    "table": "103/A/NBP",
    "rates": [
        {"no": "1/A/NBP/2024", "effectiveDate": "2024-01-01", "referenceRate": 5.75},
        {"no": "2/A/NBP/2024", "effectiveDate": "2024-01-10", "referenceRate": 5.50},
    ],
}

NBP_RATES_XML_FIXTURE = """<?xml version="1.0" encoding="utf-8"?>
<stopy_procentowe_archiwum data_publikacji="2024-01-15">
    <pozycje obowiazuje_od="2024-01-01">
        <pozycja id="ref" oprocentowanie="5,75" />
        <pozycja id="lom" oprocentowanie="6,25" />
    </pozycje>
    <pozycje obowiazuje_od="2024-01-10">
        <pozycja id="ref" oprocentowanie="5,50" />
        <pozycja id="lom" oprocentowanie="6,00" />
    </pozycje>
</stopy_procentowe_archiwum>
"""

MF_CSV_FIXTURE = """data,2Y,5Y,7Y,10Y
2024-01-02,5.1,5.2,5.25,5.3
2024-01-03,5.11,5.21,5.26,5.31
"""


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(mp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mp, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(mp, "META_PATH", tmp_path / "meta.json")
    return tmp_path


def test_expand_nbp_reference_rate(cache_dir, monkeypatch):
    monkeypatch.setattr(mp, "_http_get_bytes", lambda url, **kw: NBP_RATES_XML_FIXTURE.encode("utf-8"))
    monkeypatch.setattr(mp, "_http_get_json", lambda url: NBP_INTEREST_FIXTURE)
    s = mp.fetch_nbp_reference_rate(refresh=True)
    assert s.name == "nbp_rate"
    assert s.loc[pd.Timestamp("2024-01-02")] == 5.75
    assert s.loc[pd.Timestamp("2024-01-09")] == 5.75
    assert s.loc[pd.Timestamp("2024-01-10")] == 5.50
    csv_p, pq_p = mp._cache_paths("nbp_rate")
    assert csv_p.exists() and pq_p.exists()


def test_expand_nbp_reference_rate_legacy_json_fallback(cache_dir, monkeypatch):
    """Gdy snapshot XML nie odpowiada (sieć), fetcher spada na stare JSON API."""
    from urllib.error import URLError

    monkeypatch.setattr(mp, "_http_get_bytes", lambda url, **kw: (_ for _ in ()).throw(URLError("down")))
    monkeypatch.setattr(mp, "_http_get_json", lambda url: NBP_INTEREST_FIXTURE)
    s = mp.fetch_nbp_reference_rate(refresh=True)
    assert s.loc[pd.Timestamp("2024-01-10")] == 5.50


def test_parse_mf_yield_csv_to_four_series(cache_dir):
    mp.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (mp.RAW_DIR / "mf_yield_curve.csv").write_text(MF_CSV_FIXTURE, encoding="utf-8")
    out = mp.fetch_bonds_pl_yields(refresh=True)
    assert set(out) == set(mp.BOND_SERIES)
    assert len(out["bond_pl_10y"]) == 2


def test_parse_nbp_credits_xlsx(cache_dir):
    pytest.importorskip("openpyxl")
    mp.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = mp.RAW_DIR / "nbp_credits_deposits.xlsx"
    df = pd.DataFrame(
        {
            "Okres": ["2024-01-31", "2024-02-29"],
            "Kredyty konsumpcyjne": [100.0, 101.0],
            "Kredyty mieszkaniowe": [200.0, 201.0],
            "Kredyty przedsiębiorstw": [300.0, 302.0],
            "Depozyty gospodarstw domowych": [400.0, 405.0],
        }
    )
    df.to_excel(path, index=False, engine="openpyxl")
    out = mp.fetch_nbp_credits_deposits(refresh=True)
    assert out["credit_consumer"].iloc[-1] == 101.0
    assert out["deposits_households"].iloc[0] == 400.0


def test_validate_macro_monthly_gaps():
    idx = pd.to_datetime(["2024-01-31", "2024-03-31"])
    s = pd.Series([1.0, 2.0], index=idx, name="credit_consumer")
    q = mp.validate_macro_series(s, mp.SPECS["credit_consumer"])
    assert q["n_internal_gaps"] == 1


def test_load_panel_cache_only(cache_dir, monkeypatch):
    monkeypatch.setattr(mp, "_http_get_bytes", lambda url, **kw: NBP_RATES_XML_FIXTURE.encode("utf-8"))
    monkeypatch.setattr(mp, "_http_get_json", lambda url: NBP_INTEREST_FIXTURE)
    mp.fetch_nbp_reference_rate(refresh=True)
    s = mp.load_series("nbp_rate", use_cache_only=True)
    assert len(s) > 0


def test_eurostat_series_parses_jsonstat(cache_dir, monkeypatch):
    """Fixturka JSON-STAT Eurostatu: dwie daty, jedna wartość pusta."""
    payload = {
        "dimension": {"time": {"category": {"index": {"2024-01": 0, "2024-02": 1, "2024-03": 2}}}},
        "value": {"0": 2.8, "2": 3.1},
    }
    monkeypatch.setattr(mp, "_http_get_json", lambda url, **kw: payload)
    s = mp._fetch_eurostat_series("prc_hicp_manr", {"geo": "PL"}, "gus_cpi_yoy")
    assert s.name == "gus_cpi_yoy"
    assert list(s.index.strftime("%Y-%m")) == ["2024-01", "2024-03"]
    assert s.iloc[-1] == 3.1
