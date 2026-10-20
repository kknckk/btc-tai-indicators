#!/usr/bin/env python3
"""Cloud Run Job: artefakt z GCS → prognoza → BigQuery.

Przepływ
--------
1. ``GCS_ARTIFACT_URI`` (np. ``gs://bucket/tsfm/model.json``) – konfiguracja
   modelu; brak artefaktu => jawny stub (random walk + empiryczne kwantyle).
2. Historia ceny BTC z BigQuery (``PRICE_TABLE``/``PRICE_COLUMN``), a przy
   braku dostępu – z lokalnego CSV w obrazie (``LOCAL_PRICE_CSV``).
3. Prognoza h kroków (p10/p50/p90); modele TSFM tylko gdy ``ALLOW_TSFM=1``
   (obraz z zainstalowanym ``chronos-forecasting``), inaczej stub.
4. Zapis wierszy do ``predictions`` + wpis w ``model_runs`` (idempotentnie po
   ``RUN_ID``: najpierw DELETE, potem INSERT).

``DRY_RUN=1`` wypisuje JSON na stdout i nie dotyka BigQuery – używane w testach
lokalnych i w CI.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tsfm_wrapper import (  # noqa: E402
    DependencyMissing,
    TSFMConfig,
    create_forecaster,
)

LOGGER = logging.getLogger("inference_job")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project")
DATASET = os.environ.get("BQ_DATASET", "btc_indicators")
PREDICTIONS_TABLE = os.environ.get("BQ_PREDICTIONS_TABLE", "predictions")
RUNS_TABLE = os.environ.get("BQ_MODEL_RUNS_TABLE", "model_runs")
ARTIFACT_URI = os.environ.get("GCS_ARTIFACT_URI", "")
PRICE_TABLE = os.environ.get("PRICE_TABLE", "coinmetrics_daily")
PRICE_COLUMN = os.environ.get("PRICE_COLUMN", "PriceUSD")
LOCAL_PRICE_CSV = os.environ.get("LOCAL_PRICE_CSV", str(REPO_ROOT / "api" / "csv" / "PriceUSD.csv"))
HORIZON = int(os.environ.get("HORIZON", "7"))
ALLOW_TSFM = os.environ.get("ALLOW_TSFM", "0") == "1"
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
RUN_ID_ENV = os.environ.get("RUN_ID", "")
GIT_COMMIT = os.environ.get("GIT_COMMIT", "")

_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Wejście: cena, artefakt
# ---------------------------------------------------------------------------
def load_price_from_bigquery(client: Any) -> List[Tuple[date, float]]:
    if not _COLUMN_RE.match(PRICE_COLUMN):
        raise ValueError(f"Niedozwolona nazwa kolumny: {PRICE_COLUMN}")
    table_id = f"{PROJECT_ID}.{DATASET}.{PRICE_TABLE}"
    query = (
        f"SELECT DATE(time) AS d, {PRICE_COLUMN} AS price FROM `{table_id}` "
        f"WHERE {PRICE_COLUMN} IS NOT NULL AND {PRICE_COLUMN} > 0 ORDER BY d"
    )
    rows = list(client.query(query).result())
    series = [(row["d"] if isinstance(row["d"], date) else row["d"].date(), float(row["price"])) for row in rows]
    LOGGER.info("BigQuery: %d obserwacji z %s", len(series), table_id)
    return series


def load_price_from_csv(path: str) -> List[Tuple[date, float]]:
    import csv

    series: Dict[date, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                day = datetime.strptime(str(row["time"])[:10], "%Y-%m-%d").date()
                value = float(row["PriceUSD"])
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0:
                series[day] = value
    LOGGER.info("CSV: %d obserwacji z %s", len(series), path)
    return sorted(series.items())


def load_artifact(uri: str) -> Optional[Dict[str, Any]]:
    """Artefakt z GCS (``gs://``) albo lokalnej ścieżki."""
    if not uri:
        return None
    if uri.startswith("gs://"):
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:  # pragma: no cover
            LOGGER.error("Brak google-cloud-storage – artefakt %s pominięty", uri)
            return None
        bucket_name, blob_name = uri[5:].split("/", 1)
        try:
            text = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()
        except Exception as exc:  # pragma: no cover - GCP
            LOGGER.error("Nie udało się pobrać %s: %s", uri, exc)
            return None
        return json.loads(text)
    path = Path(uri)
    if not path.exists():
        LOGGER.error("Brak pliku artefaktu: %s", uri)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Prognoza
# ---------------------------------------------------------------------------
def build_forecast(
    artifact: Optional[Dict[str, Any]],
    prices: Sequence[float],
    horizon: int,
) -> Tuple[Any, str, Optional[str]]:
    model = str((artifact or {}).get("model") or "stub").lower()
    params = dict((artifact or {}).get("params") or {})
    context_len = int((artifact or {}).get("context_len") or 256)
    quantiles = tuple((artifact or {}).get("quantiles") or (0.1, 0.5, 0.9))

    if model != "stub" and not ALLOW_TSFM:
        LOGGER.warning("Artefakt wymaga modelu '%s', ale ALLOW_TSFM=0 – stub", model)
        model, fallback = "stub", f"ALLOW_TSFM=0 – stub zamiast '{model}'"
    else:
        fallback = None

    config = TSFMConfig(
        model=model,
        context_len=context_len,
        horizon=horizon,
        quantiles=quantiles,
        model_id=(artifact or {}).get("model_id"),
        extra=params,
    )
    forecaster = create_forecaster(config)
    try:
        if hasattr(forecaster, "_load"):
            forecaster._load()  # type: ignore[attr-defined]
    except DependencyMissing as exc:
        LOGGER.warning("Fallback do stub: %s", exc)
        fallback = str(exc)
        forecaster = create_forecaster(
            TSFMConfig(model="stub", context_len=context_len, horizon=horizon, quantiles=quantiles)
        )
        model = "stub"
    result = forecaster.fit(list(prices)).predict(horizon=horizon)
    mode = "stub" if model == "stub" else "artifact"
    return result, mode, fallback


def build_rows(
    run_id: str,
    model: str,
    mode: str,
    as_of: date,
    last_price: float,
    forecast: Any,
    horizon: int,
) -> List[Dict[str, Any]]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    labels = {f"p{int(round(level * 100))}": values for level, values in forecast.quantiles.items()}
    rows = []
    for step in range(1, horizon + 1):
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "mode": mode,
                "as_of": as_of.isoformat(),
                "forecast_date": (as_of + timedelta(days=step)).isoformat(),
                "step": step,
                "point": float(forecast.p50[step - 1]),
                "p10": float(labels.get("p10", forecast.p50)[step - 1]),
                "p50": float(labels.get("p50", forecast.p50)[step - 1]),
                "p90": float(labels.get("p90", forecast.p50)[step - 1]),
                "last_price": float(last_price),
                "created_at": created_at,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Wyjście: BigQuery
# ---------------------------------------------------------------------------
def _table(name: str) -> str:
    return f"{PROJECT_ID}.{DATASET}.{name}"


def delete_previous_run(client: Any, run_id: str) -> None:
    for table in (PREDICTIONS_TABLE, RUNS_TABLE):
        client.query(
            f"DELETE FROM `{_table(table)}` WHERE run_id = @run_id",
            job_config=_query_config(run_id),
        ).result()


def _query_config(run_id: str) -> Any:
    from google.cloud import bigquery  # type: ignore

    return bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )


def write_bigquery(
    client: Any,
    rows: List[Dict[str, Any]],
    run_row: Dict[str, Any],
) -> None:
    delete_previous_run(client, str(run_row["run_id"]))
    errors = client.insert_rows_json(_table(PREDICTIONS_TABLE), rows)
    if errors:
        raise RuntimeError(f"BigQuery odrzucił {len(errors)} wierszy: {errors[:3]}")
    errors = client.insert_rows_json(_table(RUNS_TABLE), [run_row])
    if errors:
        raise RuntimeError(f"BigQuery odrzucił wpis model_runs: {errors[:3]}")
    LOGGER.info("Zapisano %d wierszy do %s", len(rows), _table(PREDICTIONS_TABLE))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started_at = datetime.now(timezone.utc)
    run_id = RUN_ID_ENV or f"infer-{started_at.strftime('%Y%m%dT%H%M%SZ')}"

    if not 1 <= HORIZON <= 90:
        LOGGER.error("HORIZON=%s poza zakresem 1-90", HORIZON)
        return 64

    client = None
    status = "ok"
    error: Optional[str] = None
    rows: List[Dict[str, Any]] = []
    run_row: Dict[str, Any] = {"run_id": run_id}

    try:
        if not DRY_RUN:
            from google.cloud import bigquery  # type: ignore

            client = bigquery.Client(project=PROJECT_ID)

        try:
            history = load_price_from_bigquery(client) if client is not None else []
        except Exception as exc:  # brak uprawnień/tabeli – fallback do CSV
            LOGGER.warning("BigQuery niedostępne (%s) – używam CSV", exc)
            history = []
        if not history:
            history = load_price_from_csv(LOCAL_PRICE_CSV)
        if len(history) < 32:
            raise RuntimeError(f"Za mało historii: {len(history)} obserwacji (minimum 32)")

        artifact = load_artifact(ARTIFACT_URI)
        if artifact is None:
            LOGGER.warning("Brak artefaktu (%s) – jawny stub", ARTIFACT_URI or "<puste GCS_ARTIFACT_URI>")

        prices = [price for _, price in history]
        forecast, mode, fallback = build_forecast(artifact, prices, HORIZON)
        as_of, last_price = history[-1]
        model = str((artifact or {}).get("model") or "stub")
        if mode == "stub" and model != "stub":
            model = "stub"

        rows = build_rows(run_id, model, mode, as_of, last_price, forecast, HORIZON)
        run_row = {
            "run_id": run_id,
            "model": model,
            "mode": mode,
            "artifact_uri": ARTIFACT_URI or None,
            "horizon": HORIZON,
            "as_of": as_of.isoformat(),
            "n_history": len(prices),
            "metrics_json": json.dumps({"fallback": fallback}) if fallback else None,
            "status": "ok",
            "error": None,
            "git_commit": GIT_COMMIT or None,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        if DRY_RUN:
            print(json.dumps({"run": run_row, "rows": rows}, indent=2, ensure_ascii=False))
            return 0

        assert client is not None
        write_bigquery(client, rows, run_row)
        LOGGER.info("Run %s zakończony: model=%s mode=%s h=%d", run_id, model, mode, HORIZON)
        return 0

    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Inferencja nie powiodła się")
        if client is not None:
            try:
                client.insert_rows_json(
                    _table(RUNS_TABLE),
                    [
                        {
                            "run_id": run_id,
                            "model": run_row.get("model", "unknown"),
                            "mode": run_row.get("mode", "stub"),
                            "artifact_uri": ARTIFACT_URI or None,
                            "horizon": HORIZON,
                            "as_of": run_row.get("as_of"),
                            "n_history": run_row.get("n_history"),
                            "metrics_json": None,
                            "status": status,
                            "error": error[:4000],
                            "git_commit": GIT_COMMIT or None,
                            "started_at": started_at.isoformat(timespec="seconds"),
                            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }
                    ],
                )
            except Exception:  # pragma: no cover - best effort
                LOGGER.exception("Nie udało się zapisać statusu failed do model_runs")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
