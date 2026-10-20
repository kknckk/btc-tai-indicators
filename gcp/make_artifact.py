#!/usr/bin/env python3
"""Buduje artefakt modelu (JSON) dla API i Cloud Run Joba.

Artefakt jest małym plikiem konfiguracyjnym – **nie** zawiera wag sieci. Dla
``--model stub`` dopasowuje ``sigma_daily`` (odchylenie log-zwrotów z ostatnich
``--sigma-window`` dni) i opcjonalny ``drift_daily``; dla modeli TSFM zapisuje
``model_id`` i ``context_len``, a parametry statystyczne zostają jako fallback.

Przykład:
    python gcp/make_artifact.py --metrics results/tsfm.json \\
        --model stub --context-len 256 --out models/artifacts/stub.json

Następnie:
    gcloud storage cp models/artifacts/stub.json gs://BUCKET/tsfm/model.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from models.tsfm_wrapper import DEFAULT_MODEL_IDS, SUPPORTED_MODELS  # noqa: E402

PRICE_CANDIDATES: Tuple[Path, ...] = (
    REPO_ROOT / "data_ingestion" / "csv" / "PriceUSD.csv",
    REPO_ROOT / "api" / "csv" / "PriceUSD.csv",
)


def load_prices(csv_path: Optional[str]) -> Tuple[np.ndarray, str, str]:
    import pandas as pd

    path = Path(csv_path) if csv_path else next((p for p in PRICE_CANDIDATES if p.exists()), None)
    if path is None or not path.exists():
        raise FileNotFoundError(
            "Brak CSV z ceną. Sprawdzone: " + ", ".join(str(p) for p in PRICE_CANDIDATES)
        )
    df = pd.read_csv(path)
    value_col = "PriceUSD" if "PriceUSD" in df.columns else df.columns[-1]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna().query(f"{value_col} > 0").drop_duplicates("time", keep="last").sort_values("time")
    prices = df[value_col].to_numpy(dtype=float)
    return prices, str(path), df["time"].iloc[-1].date().isoformat()


def fit_params(prices: np.ndarray, window: int, drift_mode: str) -> Dict[str, Any]:
    log_returns = np.diff(np.log(np.clip(prices[-(window + 1) :], 1e-12, None)))
    sigma = float(np.std(log_returns, ddof=1))
    drift = float(np.mean(log_returns)) if drift_mode == "mean" else 0.0
    return {
        "sigma_daily": round(sigma, 8),
        "drift_daily": round(drift, 10),
        "sigma_window": int(window),
        "method": "stub_random_walk_empirical_quantiles",
    }


def summarise_metrics(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {"source": str(metrics_path), "status": "missing"}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    horizons = payload.get("horizons") or {}
    return {
        "source": str(metrics_path),
        "mode": payload.get("mode"),
        "stub": payload.get("stub"),
        "generated_at": payload.get("generated_at"),
        "horizons": {
            key: {
                "rmse": row.get("metrics", {}).get("rmse"),
                "mae": row.get("metrics", {}).get("mae"),
                "dir_acc": row.get("metrics", {}).get("dir_acc"),
                "rmse_skill_vs_rw": row.get("metrics", {}).get("rmse_skill_vs_rw"),
                "n_origins": row.get("n_origins"),
            }
            for key, row in horizons.items()
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generuje artefakt JSON modelu prognostycznego")
    parser.add_argument("--model", default="stub", choices=list(SUPPORTED_MODELS))
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--sigma-window", type=int, default=90)
    parser.add_argument("--drift", choices=("zero", "mean"), default="zero")
    parser.add_argument("--metrics", default=str(REPO_ROOT / "results" / "tsfm.json"))
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None, help="domyślnie models/artifacts/<model>.json")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        prices, source, last_date = load_prices(args.csv)
    except FileNotFoundError as exc:
        print(f"BLAD: {exc}", file=sys.stderr)
        return 3

    quantiles: List[float] = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    artifact: Dict[str, Any] = {
        "artifact_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "model_id": DEFAULT_MODEL_IDS.get(args.model),
        "context_len": int(args.context_len),
        "quantiles": quantiles,
        "params": fit_params(prices, int(args.sigma_window), args.drift),
        "training": {
            "n_obs": int(len(prices)),
            "last_price": float(prices[-1]),
            "last_date": last_date,
            "data_source": source,
            "metrics": summarise_metrics(args.metrics),
        },
    }
    artifact["params"]["expected_quantiles"] = len(quantiles)

    out_path = Path(args.out) if args.out else REPO_ROOT / "models" / "artifacts" / f"{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Zapisano artefakt: {out_path}")
    print(json.dumps(artifact["params"], indent=2, ensure_ascii=False))
    print(f"n_obs={artifact['training']['n_obs']} last_price={artifact['training']['last_price']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
