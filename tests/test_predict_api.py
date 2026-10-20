"""Testy endpointów predykcji (``api/routers/predict.py``) przez ``TestClient``.

Uruchomienie:
    /opt/data/venvs/btcai/bin/python -m pytest tests/test_predict_api.py -q

Testy nie wymagają GCP ani sieci: ``USE_LOCAL_CSV=true`` sprawia, że ``api.main``
nie inicjalizuje klienta BigQuery, a prognoza czyta ``api/csv/PriceUSD.csv``.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
for _path in (str(ROOT), str(API_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Musi być ustawione PRZED importem api.main (moduł czyta env przy imporcie).
import os

os.environ["USE_LOCAL_CSV"] = "true"
os.environ["PREDICT_ARTIFACT_DIR"] = str(ROOT / "models" / "artifacts")
os.environ.pop("PREDICT_ARTIFACT_GCS", None)

from fastapi.testclient import TestClient  # noqa: E402

# Importujemy dokładnie ten moduł, którego używa produkcyjny entrypoint
# (``uvicorn main:app`` w katalogu api/), a nie jego kopię ``api.main``.
from main import app  # noqa: E402
from routers import predict as predict_router  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module")
def price_history():
    history, source = predict_router.load_price_history()
    assert isinstance(source, str)
    return history


def test_models_endpoint_lists_all_models() -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    payload = response.json()

    assert payload["default_model"] == "auto"
    names = {item["model"] for item in payload["models"]}
    assert {"stub", "chronos", "chronos-bolt", "timesfm", "moirai"} <= names

    stub = next(item for item in payload["models"] if item["model"] == "stub")
    assert stub["stub"] is True
    assert stub["serving_mode"] == "stub"


def test_predict_default_returns_stub_quantiles(price_history) -> None:
    response = client.post("/api/v1/predict", json={"horizon": 7})
    assert response.status_code == 200
    payload = response.json()

    assert payload["mode"] == "stub"
    assert payload["stub"] is True
    assert payload["horizon"] == 7
    assert len(payload["point_forecast"]) == 7
    assert set(payload["quantiles"]) >= {"p10", "p50", "p90"}

    p10 = payload["quantiles"]["p10"]
    p50 = payload["quantiles"]["p50"]
    p90 = payload["quantiles"]["p90"]
    assert len(p10) == len(p50) == len(p90) == 7
    assert all(low <= mid <= high for low, mid, high in zip(p10, p50, p90))

    # Ostatnia cena i data muszą odpowiadać realnemu plikowi CSV.
    last_day, last_price = price_history[-1]
    assert payload["as_of"] == last_day.isoformat()
    assert payload["last_price"] == pytest.approx(last_price, rel=1e-6)
    assert payload["n_history"] == len(price_history)

    dates = [point["date"] for point in payload["point_forecast"]]
    assert dates == [(last_day + timedelta(days=i)).isoformat() for i in range(1, 8)]


def test_predict_respects_as_of_and_horizon(price_history) -> None:
    cutoff = date(2024, 1, 1)
    response = client.post(
        "/api/v1/predict", json={"horizon": 30, "model": "stub", "as_of": cutoff.isoformat()}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["as_of"] == cutoff.isoformat()
    assert len(payload["point_forecast"]) == 30
    assert payload["n_history"] == sum(1 for day, _ in price_history if day <= cutoff)
    expected_day, expected_last = max(
        ((day, price) for day, price in price_history if day <= cutoff), key=lambda item: item[0]
    )
    assert payload["as_of"] == expected_day.isoformat()
    assert payload["last_price"] == pytest.approx(expected_last, rel=1e-6)
    # horyzont nie może wyciekać przyszłych cen – prognoza startuje po as_of
    assert payload["point_forecast"][0]["date"] == (cutoff + timedelta(days=1)).isoformat()


def test_predict_is_deterministic() -> None:
    body = {"horizon": 10, "model": "stub", "as_of": "2025-01-01"}
    first = client.post("/api/v1/predict", json=body)
    second = client.post("/api/v1/predict", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["point_forecast"] == second.json()["point_forecast"]
    assert first.json()["quantiles"] == second.json()["quantiles"]


def test_predict_tsfm_without_artifact_falls_back_to_stub() -> None:
    response = client.post(
        "/api/v1/predict", json={"horizon": 3, "model": "chronos-bolt"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_model"] == "chronos-bolt"
    assert payload["mode"] == "stub"
    assert payload["fallback_reason"]
    assert "artefakt" in payload["fallback_reason"]


def test_predict_validation_errors() -> None:
    assert client.post("/api/v1/predict", json={"horizon": 0}).status_code == 422
    assert client.post("/api/v1/predict", json={"horizon": 1000}).status_code == 422
    assert client.post("/api/v1/predict", json={"horizon": 5, "model": "nope"}).status_code == 422
    assert (
        client.post("/api/v1/predict", json={"horizon": 5, "as_of": "01-02-2024"}).status_code
        == 400
    )
    # data przed początkiem historii
    assert (
        client.post("/api/v1/predict", json={"horizon": 5, "as_of": "2009-01-01"}).status_code
        == 400
    )


def test_stub_artifact_params_are_applied(tmp_path, monkeypatch) -> None:
    """Artefakt z ``params.sigma_daily`` musi zawęzić pasmo kwantyli."""
    artifact = tmp_path / "stub.json"
    artifact.write_text(
        '{"model": "stub", "context_len": 64, "quantiles": [0.1, 0.5, 0.9],'
        ' "params": {"sigma_daily": 0.001, "drift_daily": 0.0}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(predict_router, "ARTIFACT_DIR", tmp_path)
    predict_router._ARTIFACT_CACHE.clear()

    response = client.post("/api/v1/predict", json={"horizon": 5, "model": "stub"})
    assert response.status_code == 200
    payload = response.json()
    p10, p50, p90 = (
        payload["quantiles"]["p10"],
        payload["quantiles"]["p50"],
        payload["quantiles"]["p90"],
    )
    # sigma 0.1%/dzień => pasmo p10-p90 wąskie (< 1% ceny na 5 dni)
    width = (p90[-1] - p10[-1]) / p50[-1]
    assert width < 0.01, f"pasmo {width:.4%} nie odpowiada sigma=0.001"
    assert payload["mode"] == "stub"
