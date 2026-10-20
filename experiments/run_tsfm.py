#!/usr/bin/env python3
"""Zero-shot ewaluacja modeli TSFM na realnych danych BTC (rolling origin).

Pipeline
--------
1. Wczytuje dzienną cenę BTC: najpierw próbuje ``experiments.data`` (moduł
   budowany równolegle przez innego agenta), a przy ``ImportError`` używa
   lokalnego loadera CSV (``data_ingestion/csv/*.csv`` + ``api/csv/PriceUSD.csv``).
2. Dla każdego horyzontu h ∈ {1, 7, 30} buduje rolling-origin folds: kontekst
   to ostatnie ``context_len`` obserwacji przed originem, prognoza dotyczy
   ``h`` kroków naprzód, a metryki liczone są na wartości końcowej (origin+h).
3. Liczy RMSE/MAE/MAPE/sMAPE, trafność kierunku oraz baseline random walk
   (predykcja = ostatnia cena) i momentum (kierunek ostatniej zmiany).
4. Zapisuje ``results/tsfm.json`` + ``results/tsfm.md``.

``--dry-run`` podstawia jawny model atrapę (``stub``: random walk + empiryczne
kwantyle) – cały pipeline działa bez ciężkich zależności, a wynik ma
``mode="dry-run"`` i ``stub=true``. Uruchomienie bez ``--dry-run`` wymaga
``chronos-forecasting`` / ``timesfm``; brak biblioteki kończy się czytelnym
komunikatem i plikiem ``results/tsfm_blocked.json`` (exit code 2).

Przykłady
---------
    /opt/data/venvs/btcai/bin/python experiments/run_tsfm.py --dry-run
    /opt/data/venvs/btcai/bin/python experiments/run_tsfm.py \\
        --model chronos-bolt --horizons 1,7,30 --origins 40
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tsfm_wrapper import (  # noqa: E402  (po ustawieniu sys.path)
    DEFAULT_MODEL_IDS,
    DependencyMissing,
    TSFMConfig,
    available_models,
    create_forecaster,
)

LOGGER = logging.getLogger("run_tsfm")
PRICE_CANDIDATES: Tuple[str, ...] = (
    "data_ingestion/csv/PriceUSD.csv",
    "api/csv/PriceUSD.csv",
)


# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------
def _load_price_csv(path: Path) -> "Any":
    """Wczytuje ``time,PriceUSD`` bez zależności od pandas (fallback)."""
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Brak pliku z ceną: {path}")
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"{path}: brak kolumny 'time'")
    value_col = "PriceUSD" if "PriceUSD" in df.columns else df.columns[-1]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["time", value_col])
    df = df[df[value_col] > 0]
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    series = df.set_index("time")[value_col].astype(float)
    series.name = "price"
    return series


def load_price_series(
    start: Optional[str] = None,
    end: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Tuple["Any", str]:
    """Zwraca ``(series, źródło)``; najpierw cudzy loader, potem lokalne CSV."""
    import pandas as pd

    if csv_path:
        path = Path(csv_path)
        return _load_price_csv(path).loc[start:end], str(path)

    # 1) Opcjonalny moduł innego agenta – tylko jeśli istnieje i ma loader.
    try:
        from experiments import data as ext_data  # type: ignore
    except ImportError:
        ext_data = None  # type: ignore[assignment]
    if ext_data is not None:
        loader = getattr(ext_data, "load_price_series", None) or getattr(
            ext_data, "load_prices", None
        )
        if callable(loader):
            try:
                series = loader(start=start, end=end)
                if series is not None and len(series) > 0:
                    series = series.astype(float).sort_index()
                    series = series[~series.index.duplicated(keep="last")]
                    LOGGER.info("Cena BTC z experiments.data (%d obserwacji)", len(series))
                    return series, "experiments.data"
            except Exception as exc:  # pragma: no cover - zależne od gałęzi
                LOGGER.warning("experiments.data nie dostarczył ceny (%s) – fallback CSV", exc)

    # 2) Lokalny fallback: data_ingestion/csv, potem api/csv.
    for candidate in PRICE_CANDIDATES:
        path = REPO_ROOT / candidate
        if path.exists():
            series = _load_price_csv(path)
            LOGGER.info("Cena BTC z %s (%d obserwacji)", candidate, len(series))
            return series.loc[start:end], candidate
    raise FileNotFoundError(
        "Nie znaleziono ceny BTC. Sprawdzone: " + ", ".join(PRICE_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Rolling origin + metryki
# ---------------------------------------------------------------------------
def build_origins(
    n_obs: int,
    context_len: int,
    horizon: int,
    max_origins: int,
    step: int = 1,
) -> np.ndarray:
    """Indeksy originów (ostatnia obserwacja kontekstu), rosnąco.

    Wymagania: ``origin >= context_len - 1`` oraz ``origin + horizon <= n_obs - 1``.
    """
    last_origin = n_obs - horizon - 1
    first_possible = context_len - 1
    if last_origin < first_possible:
        raise ValueError(
            f"Za mało danych: n={n_obs}, context_len={context_len}, h={horizon}"
        )
    step = max(1, int(step))
    first = max(first_possible, last_origin - step * (max_origins - 1))
    return np.arange(first, last_origin + 1, step, dtype=int)


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    last_observed: np.ndarray,
    previous_observed: np.ndarray,
) -> Dict[str, float]:
    """Metryki punktowe + kierunkowe + baseline'y (wszystko na origin+h)."""
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    errors = predictions - actuals
    naive_errors = last_observed - actuals

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / actuals)) * 100.0)
    denom = np.abs(predictions) + np.abs(actuals)
    smape = float(np.mean(np.where(denom > 0, 2 * np.abs(errors) / denom, 0.0)) * 100.0)

    dir_pred = np.sign(predictions - last_observed)
    dir_actual = np.sign(actuals - last_observed)
    dir_acc = float(np.mean(dir_pred == dir_actual))

    # Baseline kierunkowy: momentum (kierunek ostatniej zaobserwowanej zmiany).
    momentum = np.sign(last_observed - previous_observed)
    dir_acc_momentum = float(np.mean(momentum == dir_actual))

    rmse_naive = float(np.sqrt(np.mean(naive_errors**2)))
    mae_naive = float(np.mean(np.abs(naive_errors)))
    sse_model = float(np.sum(errors**2))
    sse_naive = float(np.sum(naive_errors**2))
    return {
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
        "smape_pct": smape,
        "dir_acc": dir_acc,
        "dir_acc_rw_momentum": dir_acc_momentum,
        "dir_edge_vs_rw": dir_acc - dir_acc_momentum,
        "rmse_naive_rw": rmse_naive,
        "mae_naive_rw": mae_naive,
        "rmse_skill_vs_rw": 1.0 - rmse / rmse_naive if rmse_naive > 0 else float("nan"),
        "r2_oos_vs_rw": 1.0 - sse_model / sse_naive if sse_naive > 0 else float("nan"),
    }


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover
        return None


def run_horizon(
    y: np.ndarray,
    forecaster: Any,
    horizon: int,
    context_len: int,
    max_origins: int,
    step: int,
    batch_size: int,
) -> Dict[str, Any]:
    origins = build_origins(len(y), context_len, horizon, max_origins, step)
    predictions: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    actuals: List[float] = []
    last_obs: List[float] = []
    prev_obs: List[float] = []

    for start in range(0, len(origins), batch_size):
        chunk = origins[start : start + batch_size]
        contexts = [y[o - context_len + 1 : o + 1] for o in chunk]
        results = forecaster.predict_batch(contexts, horizon=horizon)
        for origin, result in zip(chunk, results):
            predictions.append(float(result.p50[-1]))
            lows.append(float(result.p10[-1]))
            highs.append(float(result.p90[-1]))
            actuals.append(float(y[origin + horizon]))
            last_obs.append(float(y[origin]))
            prev_obs.append(float(y[origin - 1]))

    pred = np.asarray(predictions)
    act = np.asarray(actuals)
    last = np.asarray(last_obs)
    prev = np.asarray(prev_obs)
    lo = np.asarray(lows)
    hi = np.asarray(highs)

    metrics = compute_metrics(pred, act, last, prev)
    coverage = float(np.mean((act >= lo) & (act <= hi)))
    return {
        "horizon": horizon,
        "n_origins": int(len(origins)),
        "first_origin": str(_origin_date(y, origins[0])) if len(origins) else None,
        "last_origin": str(_origin_date(y, origins[-1])) if len(origins) else None,
        "metrics": metrics,
        "interval_coverage_p10_p90": coverage,
        "median_interval_width_pct": float(np.mean((hi - lo) / last) * 100.0)
        if len(last)
        else None,
    }


def _origin_date(y: np.ndarray, index: int) -> Any:
    """Data originu, jeśli seria ma DatetimeIndex (w innym razie numer)."""
    dates = getattr(y, "index", None)
    if dates is not None and index < len(dates):
        try:
            return dates[index].date().isoformat()
        except Exception:
            return dates[index]
    return int(index)


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    mode = payload["mode"]
    lines.append("# TSFM zero-shot – wyniki na BTC\n")
    lines.append(f"- **Tryb:** `{mode}`" + (" (ATRAPA – dane nie są wynikiem modelu TSFM)" if payload.get("stub") else ""))
    lines.append(f"- **Model:** `{payload['model']}` (`{payload.get('model_id')}`)")
    lines.append(f"- **Dane:** `{payload['data']['source']}` – {payload['data']['n_obs']} obserwacji, "
                 f"{payload['data']['start']} → {payload['data']['end']}")
    lines.append(
        f"- **Konfiguracja:** context_len={payload['config']['context_len']}, "
        f"origins/h={payload['config']['n_origins']}, step={payload['config']['step']}, "
        f"device={payload['config']['device']}"
    )
    lines.append(f"- **Czas:** {payload['runtime_seconds']:.1f} s, commit `{payload.get('git_commit')}`\n")

    lines.append("## Metryki (punktowo na origin+h)\n")
    header = (
        "| h | n | RMSE | MAE | MAPE % | sMAPE % | DirAcc | DirAcc RW-mom | "
        "RMSE RW | skill vs RW | R² OOS | pokrycie p10–p90 |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(payload["horizons"], key=lambda k: int(k)):
        row = payload["horizons"][key]
        m = row["metrics"]
        lines.append(
            f"| {row['horizon']} | {row['n_origins']} | {m['rmse']:.2f} | {m['mae']:.2f} | "
            f"{m['mape_pct']:.2f} | {m['smape_pct']:.2f} | {m['dir_acc']*100:.1f}% | "
            f"{m['dir_acc_rw_momentum']*100:.1f}% | {m['rmse_naive_rw']:.2f} | "
            f"{m['rmse_skill_vs_rw']*100:+.2f}% | {m['r2_oos_vs_rw']*100:+.2f}% | "
            f"{row['interval_coverage_p10_p90']*100:.1f}% |"
        )

    agg = payload["aggregate"]
    lines.append("\n## Agregat\n")
    lines.append(
        f"- Średnia DirAcc: **{agg['mean_dir_acc']*100:.1f}%** "
        f"(RW-momentum: {agg['mean_dir_acc_rw_momentum']*100:.1f}%), "
        f"średni RMSE skill vs RW: **{agg['mean_rmse_skill_vs_rw']*100:+.2f}%**"
    )
    lines.append(
        "- Interpretacja: dodatni `skill vs RW` = model bije random walk; "
        "ujemny = przegrywa z „jutro taka sama cena”. `DirAcc` liczone na zmianie "
        "ceny względem ostatniej obserwacji; `DirAcc RW-mom` to naiwny baseline "
        "kierunku (znak ostatniej dziennej zmiany)."
    )
    if payload.get("stub"):
        lines.append(
            "\n> **UWAGA:** tryb `--dry-run` używa atrapy `stub` (random walk + "
            "empiryczne kwantyle log-zwrotów). To test instalacji pipeline'u, "
            "**nie** wynik modelu TSFM."
        )
    if payload.get("warnings"):
        lines.append("\n## Ostrzeżenia\n")
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot TSFM na cenie BTC (rolling origin)")
    parser.add_argument("--model", default="chronos-bolt",
                        help="chronos | chronos-bolt | timesfm | moirai | stub")
    parser.add_argument("--horizons", default="1,7,30", help="lista horyzontów, np. 1,7,30")
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--origins", type=int, default=60, help="maks. liczba originów na horyzont")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", default=None, help="jawna ścieżka do CSV z ceną")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true", help="atrapa stub, bez ciężkich zależności")
    parser.add_argument("--list-models", action="store_true", help="pokaż status modeli i wyjdź")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list_models:
        for item in available_models():
            print(
                f"{item['model']:<13} available={str(item['available']):<5} "
                f"missing={item['missing_packages'] or '-'} hint={item['install_hint'] or '-'}"
            )
        return 0

    try:
        horizons = [int(x) for x in str(args.horizons).split(",") if str(x).strip()]
    except ValueError:
        LOGGER.error("--horizons musi być listą liczb, np. 1,7,30")
        return 64
    if not horizons or any(h < 1 for h in horizons):
        LOGGER.error("Horyzonty muszą być >= 1")
        return 64

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = "stub" if args.dry_run else args.model
    started = time.time()
    warnings: List[str] = []
    if args.dry_run and args.model != "stub":
        warnings.append(f"--dry-run nadpisał model '{args.model}' atrapą 'stub'")

    try:
        series, source = load_price_series(csv_path=args.csv)
    except Exception as exc:
        LOGGER.error("Nie udało się wczytać danych: %s", exc)
        return 3

    y = np.asarray(series.values, dtype=float)
    LOGGER.info("Dane: %s, %d obserwacji, %s → %s", source, len(y), series.index[0], series.index[-1])

    config = TSFMConfig(
        model=model,
        context_len=args.context_len,
        horizon=max(horizons),
        device=args.device,
        batch_size=args.batch_size,
    )

    payload: Dict[str, Any] = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "dry-run" if args.dry_run else "real",
        "stub": model == "stub",
        "model": model,
        "requested_model": args.model,
        "model_id": config.model_id or DEFAULT_MODEL_IDS.get(model),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "data": {
            "source": source,
            "n_obs": int(len(y)),
            "start": str(series.index[0].date()) if hasattr(series.index[0], "date") else str(series.index[0]),
            "end": str(series.index[-1].date()) if hasattr(series.index[-1], "date") else str(series.index[-1]),
        },
        "config": {
            "context_len": args.context_len,
            "horizons": horizons,
            "n_origins": args.origins,
            "step": args.step,
            "batch_size": args.batch_size,
            "device": args.device,
            "quantiles": list(config.quantiles),
        },
        "horizons": {},
        "warnings": warnings,
    }

    try:
        forecaster = create_forecaster(config)
        for horizon in horizons:
            t0 = time.time()
            payload["horizons"][str(horizon)] = run_horizon(
                y, forecaster, horizon, args.context_len, args.origins, args.step, args.batch_size
            )
            LOGGER.info(
                "h=%d: %d originów, RMSE=%.2f, DirAcc=%.1f%%, skill=%.2f%% (%.1f s)",
                horizon,
                payload["horizons"][str(horizon)]["n_origins"],
                payload["horizons"][str(horizon)]["metrics"]["rmse"],
                payload["horizons"][str(horizon)]["metrics"]["dir_acc"] * 100,
                payload["horizons"][str(horizon)]["metrics"]["rmse_skill_vs_rw"] * 100,
                time.time() - t0,
            )
    except DependencyMissing as exc:
        payload.update(
            {
                "status": "blocked",
                "error": str(exc),
                "missing_package": exc.package,
                "install_hint": exc.install_hint,
            }
        )
        payload["runtime_seconds"] = time.time() - started
        blocked_path = out_dir / "tsfm_blocked.json"
        blocked_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.error("ZABLOKOWANE: %s", exc)
        LOGGER.error("Szczegóły: %s", blocked_path)
        return 2
    except (ValueError, RuntimeError) as exc:
        payload.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        payload["runtime_seconds"] = time.time() - started
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tsfm_blocked.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.error("Błąd uruchomienia modelu: %s", exc)
        return 4

    metrics_list = [row["metrics"] for row in payload["horizons"].values()]
    payload["aggregate"] = {
        "mean_rmse": float(np.mean([m["rmse"] for m in metrics_list])),
        "mean_mae": float(np.mean([m["mae"] for m in metrics_list])),
        "mean_dir_acc": float(np.mean([m["dir_acc"] for m in metrics_list])),
        "mean_dir_acc_rw_momentum": float(np.mean([m["dir_acc_rw_momentum"] for m in metrics_list])),
        "mean_rmse_skill_vs_rw": float(np.mean([m["rmse_skill_vs_rw"] for m in metrics_list])),
    }
    payload["runtime_seconds"] = time.time() - started

    json_path = out_dir / "tsfm.json"
    md_path = out_dir / "tsfm.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, md_path)
    LOGGER.info("Zapisano %s oraz %s", json_path, md_path)

    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "model": payload["model"],
                "stub": payload["stub"],
                "horizons": {
                    h: {
                        "n": row["n_origins"],
                        "rmse": round(row["metrics"]["rmse"], 2),
                        "dir_acc": round(row["metrics"]["dir_acc"], 4),
                        "skill_vs_rw": round(row["metrics"]["rmse_skill_vs_rw"], 4),
                    }
                    for h, row in payload["horizons"].items()
                },
                "runtime_seconds": round(payload["runtime_seconds"], 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
