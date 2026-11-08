"""Unit tests for experiments.multiasset (3 series: NBP, GPW Yahoo, local bitcoin)."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments import multiasset as ma


NBP_FIXTURE = {
    "table": "A",
    "currency": "euro",
    "code": "eur",
    "rates": [
        {"no": "001/A/NBP/2024", "effectiveDate": "2024-01-02", "mid": 4.35},
        {"no": "002/A/NBP/2024", "effectiveDate": "2024-01-03", "mid": 4.36},
        {"no": "003/A/NBP/2024", "effectiveDate": "2024-01-04", "mid": 4.34},
    ],
}

YAHOO_FIXTURE = {
    "chart": {
        "result": [
            {
                "timestamp": [1704067200, 1704153600, 1704240000],
                "indicators": {"quote": [{"close": [2200.0, 2210.5, 2195.0]}]},
            }
        ]
    }
}


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ma, "CACHE_DIR", tmp_path)
    return tmp_path


def test_validate_deduplicates_and_detects_gap():
    idx = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-04"])
    s = pd.Series([4.0, 4.1, 4.2], index=idx, name="eurpln")
    clean = ma._clean_series(s)
    assert clean.index.is_unique
    assert len(clean) == 2
    q = ma.validate_series(clean, ma.TradingCalendar.NBP)
    assert q["n_internal_gaps"] == 1  # missing 2024-01-03 (business day)


def test_load_three_series_from_cache_and_mocks(cache_dir, monkeypatch):
    def fake_http(url: str):
        if "api.nbp.pl" in url:
            return NBP_FIXTURE
        if "WIG20.WA" in url:
            return YAHOO_FIXTURE
        raise AssertionError(url)

    monkeypatch.setattr(ma, "_http_get_json", fake_http)

    # bitcoin from repo CSV (real file, not synthetic)
    btc = ma.fetch_series("bitcoin")
    assert btc.name == "bitcoin"
    assert btc.notna().sum() > 100

    eur = ma.fetch_series("eurpln")
    assert list(eur.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    wig = ma.fetch_series("wig20")
    assert len(wig) == 3

    panel = ma.load_multiasset(
        names=("eurpln", "wig20", "bitcoin"),
        use_cache_only=True,
    )
    assert panel.shape[1] == 3
    cov = ma.coverage_report(panel)
    assert set(cov.index) == {"eurpln", "wig20", "bitcoin"}
    assert (cov["pct_missing"] >= 0).all()


def test_monthly_resample(cache_dir, monkeypatch):
    monkeypatch.setattr(ma, "_http_get_json", lambda url: NBP_FIXTURE)
    s = ma.load_series("eurpln", freq="M", refresh=True)
    assert len(s) == 1
    assert s.index[0].day >= 28  # month-end


def test_cache_writes_csv_and_parquet(cache_dir, monkeypatch):
    monkeypatch.setattr(ma, "_http_get_json", lambda url: NBP_FIXTURE)
    ma.fetch_series("usdpln")
    csv_p, pq_p = ma._cache_paths("usdpln")
    assert csv_p.exists() and pq_p.exists()
    reread = pd.read_parquet(pq_p)
    assert "usdpln" in reread.columns
