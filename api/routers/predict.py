"""Endpointy predykcji BTC: ``POST /api/v1/predict`` oraz ``GET /api/v1/models``.

Zasada działania
----------------
* Router **nie trenuje** modeli. Serwuje prognozę z artefaktu modelu
  (``models/artifacts/<model>.json`` albo GCS przez ``PREDICT_ARTIFACT_GCS``).
* **STUB mode**: gdy artefaktu nie ma (albo model wymaga biblioteki, której nie
  ma w obrazie API), zwracamy jawny baseline random-walk + empiryczne kwantyle
  z oznaczeniem ``mode="stub"`` / ``stub=true`` i powodem w ``fallback_reason``.
  Nigdy nie udajemy, że to prognoza wytrenowanego modelu.
* Cena BTC czytana jest z CSV (``data_ingestion/csv/PriceUSD.csv`` lub
  ``api/csv/PriceUSD.csv``) – API nie potrzebuje do tego BigQuery.

Format artefaktu (JSON)
-----------------------
.. code-block:: json

    {
      "model": "stub",
      "artifact_version": 1,
      "created_at": "2026-05-24T02:00:00Z",
      "context_len": 256,
      "quantiles": [0.1, 0.5, 0.9],
      "params": {"sigma_daily": 0.031, "drift_daily": 0.0002},
      "training": {"n_obs": 5789, "last_price": 76619.87}
    }

Dla modeli TSFM artefakt może dodatkowo nieść ``model_id`` (np.
``amazon/chronos-bolt-small``) i ``context_len``; wtedy API ładuje bibliotekę
leniwie, a przy ``DependencyMissing`` degraduje do stub z czytelnym powodem.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve()
# W repo: <repo>/api/routers/predict.py -> parents[2] = repo root.
# W obrazie: /app/routers/predict.py -> parents[2] = "/", więc cofamy się do /app,
# gdzie Dockerfile kopiuje models/.
REPO_ROOT = _HERE.parents[2] if (_HERE.parents[2] / "models").is_dir() else _HERE.parents[1]
if str(REPO_ROOT) not in sys.path:  # obraz API kopiuje models/ do /app/models
    sys.path.insert(0, str(REPO_ROOT))

from models.tsfm_wrapper import (  # noqa: E402  (po ustawieniu sys.path)
    DEFAULT_MODEL_IDS,
    DependencyMissing,
    TSFMConfig,
    available_models,
    create_forecaster,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predict"])

ARTIFACT_DIR = Path(
    os.environ.get("PREDICT_ARTIFACT_DIR", str(REPO_ROOT / "models" / "artifacts"))
)
ARTIFACT_GCS_PREFIX = os.environ.get("PREDICT_ARTIFACT_GCS", "").rstrip("/")
PRICE_CSV_ENV = os.environ.get("PREDICT_PRICE_CSV", "")
MAX_HORIZON = 90
DEFAULT_CONTEXT_LEN = 256

PRICE_CANDIDATES: Tuple[Path, ...] = (
    Path(__file__).resolve().parents[1] / "csv" / "PriceUSD.csv",  # obraz API (/app/csv)
    REPO_ROOT / "data_ingestion" / "csv" / "PriceUSD.csv",
    REPO_ROOT / "api" / "csv" / "PriceUSD.csv",
)

_SERIES_CACHE: Dict[str, Tuple[float, List[Tuple[date, float]]]] = {}
_ARTIFACT_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

ModelName = Literal["auto", "stub", "chronos", "chronos-bolt", "timesfm", "moirai"]


# ---------------------------------------------------------------------------
# Modele odpowiedzi
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    horizon: int = Field(7, ge=1, le=MAX_HORIZON, description="Horyzont w dniach (1-90)")
    model: ModelName = Field("auto", description="auto | stub | chronos | chronos-bolt | timesfm | moirai")
    as_of: Optional[str] = Field(
        None,
        description="Data odcięcia historii (YYYY-MM-DD); domyślnie ostatnia dostępna obserwacja",
        examples=["2026-05-23"],
    )


class ForecastPoint(BaseModel):
    date: str
    value: float


class PredictResponse(BaseModel):
    model: str
    requested_model: str
    mode: Literal["artifact", "stub"]
    stub: bool
    as_of: str
    horizon: int
    context_len: int
    n_history: int
    last_price: float
    point_forecast: List[ForecastPoint]
    quantiles: Dict[str, List[float]]
    generated_at: str
    fallback_reason: Optional[str] = None
    disclaimer: str


class ModelInfo(BaseModel):
    model: str
    available: bool
    missing_packages: List[str] = Field(default_factory=list)
    install_hint: Optional[str] = None
    stub: bool = False
    artifact_present: bool = False
    serving_mode: Literal["artifact", "stub"]


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    artifact_dir: str
    artifact_gcs_prefix: Optional[str] = None


# ---------------------------------------------------------------------------
# Dane cenowe (CSV, bez BigQuery)
# ---------------------------------------------------------------------------
def _price_candidates() -> List[Path]:
    candidates: List[Path] = []
    if PRICE_CSV_ENV:
        candidates.append(Path(PRICE_CSV_ENV))
    candidates.extend(PRICE_CANDIDATES)
    return candidates


def _read_price_csv(path: Path) -> List[Tuple[date, float]]:
    """Czyta ``time,PriceUSD`` do posortowanej listy ``(date, price)``."""
    rows: List[Tuple[date, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: pusty plik CSV")
        value_field = "PriceUSD" if "PriceUSD" in reader.fieldnames else reader.fieldnames[-1]
        for raw in reader:
            try:
                day = datetime.strptime(str(raw["time"]).strip()[:10], "%Y-%m-%d").date()
                value = float(raw[value_field])
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0:
                rows.append((day, value))
    if not rows:
        raise ValueError(f"{path}: brak poprawnych obserwacji ceny")
    rows.sort(key=lambda item: item[0])
    deduped: Dict[date, float] = {day: value for day, value in rows}
    return sorted(deduped.items())


def load_price_history() -> Tuple[List[Tuple[date, float]], str]:
    """Zwraca ``(historia, ścieżka)``; cache unieważniany zmianą mtime."""
    for path in _price_candidates():
        if not path.exists():
            continue
        key = str(path.resolve())
        mtime = path.stat().st_mtime
        cached = _SERIES_CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1], str(path)
        series = _read_price_csv(path)
        _SERIES_CACHE[key] = (mtime, series)
        return series, str(path)
    raise FileNotFoundError(
        "Brak pliku z ceną BTC. Sprawdzone: " + ", ".join(str(p) for p in _price_candidates())
    )


# ---------------------------------------------------------------------------
# Artefakty modeli (lokalnie lub GCS)
# ---------------------------------------------------------------------------
def _download_gcs(uri: str) -> Optional[Dict[str, Any]]:
    """Pobiera JSON z ``gs://bucket/blob``; brak biblioteki => None + log."""
    if not uri.startswith("gs://"):
        return None
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        LOGGER.warning("PREDICT_ARTIFACT_GCS ustawione, ale google-cloud-storage nie jest zainstalowane")
        return None
    try:
        bucket_name, blob_name = uri[5:].split("/", 1)
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        return json.loads(blob.download_as_text())
    except Exception as exc:  # pragma: no cover - zależy od GCP
        LOGGER.warning("Nie udało się pobrać artefaktu %s: %s", uri, exc)
        return None


def _read_artifact_file(path: Path) -> Optional[Dict[str, Any]]:
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    cached = _ARTIFACT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Niepoprawny artefakt %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Artefakt %s nie jest obiektem JSON", path)
        return None
    _ARTIFACT_CACHE[key] = (mtime, payload)
    return payload


def load_artifact(model: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Zwraca ``(artefakt, powód_braku)`` dla danego modelu."""
    local_path = ARTIFACT_DIR / f"{model}.json"
    if local_path.exists():
        return _read_artifact_file(local_path), None
    if ARTIFACT_GCS_PREFIX:
        remote = f"{ARTIFACT_GCS_PREFIX}/{model}.json"
        payload = _download_gcs(remote)
        if payload is not None:
            return payload, None
        return None, f"brak artefaktu {remote}"
    return None, f"brak pliku artefaktu {local_path}"


def _pick_artifact_model() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Dla ``model="auto"``: najnowszy artefakt w katalogu, o ile istnieje."""
    if not ARTIFACT_DIR.exists():
        return None, None
    candidates = sorted(ARTIFACT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = _read_artifact_file(path)
        if payload is not None:
            return path.stem, payload
    return None, None


# ---------------------------------------------------------------------------
# Prognoza
# ---------------------------------------------------------------------------
def _build_forecaster(
    model: str,
    artifact: Optional[Dict[str, Any]],
    horizon: int,
) -> Tuple[Any, str, Optional[str]]:
    """Zwraca ``(forecaster, tryb, powód_fallbacku)``.

    ``model="stub"`` zawsze daje tryb stub. Model TSFM bez artefaktu również
    kończy się stubem (zgodnie z kontraktem API), a brak biblioteki jest
    raportowany w ``fallback_reason``.
    """
    params = dict((artifact or {}).get("params") or {})
    context_len = int((artifact or {}).get("context_len") or DEFAULT_CONTEXT_LEN)
    quantiles = tuple((artifact or {}).get("quantiles") or (0.1, 0.5, 0.9))
    model_id = (artifact or {}).get("model_id") or DEFAULT_MODEL_IDS.get(model)

    config = TSFMConfig(
        model=model,
        context_len=context_len,
        horizon=horizon,
        quantiles=quantiles,
        model_id=model_id,
        extra=params,
    )
    if model == "stub":
        return create_forecaster(config), "stub", None

    if artifact is None:
        reason = f"brak artefaktu modelu '{model}' – serwowany jest stub (random walk)"
        stub_config = TSFMConfig(model="stub", context_len=context_len, horizon=horizon, quantiles=quantiles)
        return create_forecaster(stub_config), "stub", reason

    try:
        forecaster = create_forecaster(config)
        # Wymuszamy leniwy import już teraz, żeby brak biblioteki zamienić na
        # kontrolowany fallback do stub, a nie na błąd 500 w środku prognozy.
        if hasattr(forecaster, "_load"):
            forecaster._load()  # type: ignore[attr-defined]
        return forecaster, "artifact", None
    except DependencyMissing as exc:
        reason = f"{exc} – serwowany jest stub"
        LOGGER.warning("Fallback do stub: %s", exc)
        stub_config = TSFMConfig(model="stub", context_len=context_len, horizon=horizon, quantiles=quantiles)
        return create_forecaster(stub_config), "stub", reason
    except Exception as exc:  # pragma: no cover - np. uszkodzony artefakt
        reason = f"model '{model}' nie uruchomił się ({type(exc).__name__}: {exc}) – stub"
        LOGGER.exception("Fallback do stub po błędzie modelu")
        stub_config = TSFMConfig(model="stub", context_len=context_len, horizon=horizon, quantiles=quantiles)
        return create_forecaster(stub_config), "stub", reason


def _parse_as_of(as_of: Optional[str]) -> Optional[date]:
    if not as_of:
        return None
    try:
        return datetime.strptime(as_of.strip()[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Parametr 'as_of' musi mieć format YYYY-MM-DD") from exc


def _artifact_has(model: str) -> bool:
    return (ARTIFACT_DIR / f"{model}.json").exists()


@router.get("/models", response_model=ModelsResponse, summary="Lista modeli i tryb serwowania")
def list_models() -> ModelsResponse:
    """Status modeli TSFM + informacja, czy istnieje artefakt (artifact vs stub)."""
    models: List[ModelInfo] = []
    for item in available_models():
        # Tylko lokalny plik traktujemy jako "artefakt obecny"; prefiks GCS jest
        # raportowany osobno, bez zgadywania, czy blob istnieje (oszczędza requesty).
        artifact_present = _artifact_has(item["model"])
        serving_mode: Literal["artifact", "stub"] = (
            "artifact" if artifact_present and item["model"] != "stub" else "stub"
        )
        models.append(
            ModelInfo(
                model=item["model"],
                available=bool(item["available"]),
                missing_packages=list(item["missing_packages"]),
                install_hint=item["install_hint"],
                stub=bool(item["stub"]),
                artifact_present=artifact_present,
                serving_mode=serving_mode,
            )
        )
    return ModelsResponse(
        models=models,
        default_model="auto",
        artifact_dir=str(ARTIFACT_DIR),
        artifact_gcs_prefix=ARTIFACT_GCS_PREFIX or None,
    )


@router.post("/predict", response_model=PredictResponse, summary="Prognoza punktowa + kwantyle")
def predict(request: PredictRequest) -> PredictResponse:
    """Prognoza ceny BTC na ``horizon`` dni (p10/p50/p90) na podstawie historii CSV."""
    try:
        history, source = load_price_history()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Niepoprawne dane cenowe: {exc}") from exc

    cutoff = _parse_as_of(request.as_of)
    if cutoff is not None:
        history = [(day, price) for day, price in history if day <= cutoff]
        if not history:
            raise HTTPException(
                status_code=400,
                detail=f"Brak obserwacji do daty {cutoff.isoformat()}",
            )

    last_day, last_price = history[-1]
    if len(history) < 8:
        raise HTTPException(
            status_code=400,
            detail=f"Za mało historii: {len(history)} obserwacji (minimum 8)",
        )

    requested = request.model
    artifact: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    if requested == "auto":
        chosen, artifact = _pick_artifact_model()
        if chosen is None:
            chosen = "stub"
            reason = "brak jakiegokolwiek artefaktu w katalogu – serwowany jest stub"
    elif requested == "stub":
        chosen = "stub"
        # Artefakt stub może nieść dopasowane parametry (sigma/drift) – używamy
        # ich, jeśli istnieją, ale tryb pozostaje jawnie "stub".
        artifact, _ = load_artifact("stub")
    else:
        chosen = requested
        artifact, reason = load_artifact(chosen)

    forecaster, mode, fallback_reason = _build_forecaster(chosen, artifact, request.horizon)
    if reason and not fallback_reason:
        fallback_reason = reason

    prices = [price for _, price in history]
    result = forecaster.fit(prices).predict(horizon=request.horizon)
    forecast_dates = [last_day + timedelta(days=step) for step in range(1, request.horizon + 1)]

    quantiles = {
        f"p{int(round(level * 100))}": [round(float(v), 4) for v in values]
        for level, values in result.quantiles.items()
    }
    return PredictResponse(
        model=chosen,
        requested_model=requested,
        mode=mode,  # type: ignore[arg-type]
        stub=mode == "stub",
        as_of=last_day.isoformat(),
        horizon=request.horizon,
        context_len=int(min(forecaster.config.context_len, len(prices))),
        n_history=len(prices),
        last_price=round(float(last_price), 4),
        point_forecast=[
            ForecastPoint(date=day.isoformat(), value=round(float(value), 4))
            for day, value in zip(forecast_dates, result.p50)
        ],
        quantiles=quantiles,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        fallback_reason=fallback_reason,
        disclaimer=(
            "STUB: prognoza to baseline random-walk z empirycznymi kwantylami, "
            "nie wytrenowany model TSFM."
            if mode == "stub"
            else "Prognoza zero-shot z artefaktu modelu; nie stanowi porady inwestycyjnej."
        ),
    )
