"""Conformal calibration report on walk-forward OOS residuals.

Uses expanding calibration strictly on past OOS rows (before each origin).
Writes ``results/calibration.md`` and ``results/calibration.json``.

Usage:
    python -m experiments.run_calibration
    python -m experiments.run_calibration --max-folds 8
    python -m experiments.run_calibration --tsfm-real
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.calibration import (
    calibrate_frame_cqr,
    calibrate_frame_point,
    coverage_width_curve,
    coverage_width_curve_cqr,
    raw_quantile_coverage,
)
from evaluation.walk_forward import run_walk_forward, sign_position
from experiments.baselines import NaiveRW, XGBForecaster
from experiments.config import ROOT, load_config
from experiments.data import build_features, load_panel

logger = logging.getLogger("calibration")
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
TARGET_ALPHAS = (0.20, 0.05)  # nominal 80% and 95%
NOMINAL = {0.20: "80%", 0.05: "95%"}


def _git_short() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _fmt_pct(x: float | None, nd: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.{nd}f}%"


def _fmt_ratio(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.2f}×"


def load_or_run_frame(
    panel: pd.DataFrame,
    cfg: dict,
    model: str,
    h: int,
    max_folds: int | None,
) -> pd.DataFrame:
    path = PREDS / f"{model}_h{h}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    wf = cfg["walk_forward"]
    ret = panel["logret_1"]
    factories = {
        "naive": lambda: NaiveRW(ret),
        "xgboost": lambda: XGBForecaster(cfg["xgboost"], cfg["seed"]),
    }
    if model not in factories:
        raise KeyError(model)
    r = run_walk_forward(
        panel,
        h,
        build_features,
        factories[model],
        min_train=wf["min_train"],
        embargo=wf["embargo"],
        step=wf["step"],
        position_fn=sign_position,
        costs_bps=tuple(cfg["costs_bps"]),
        seed=cfg["seed"],
        max_folds=max_folds,
        causal_check=False,
    )
    return r["frame"]


def tsfm_origin_frames(cfg: dict, horizons: list[int], stub: bool) -> dict[int, pd.DataFrame]:
    from experiments.run_tsfm import collect_horizon_frame, load_price_series
    from models.tsfm_wrapper import TSFMConfig, create_forecaster

    series, _ = load_price_series()
    y = series
    model = "stub" if stub else "timesfm"
    try:
        forecaster = create_forecaster(
            TSFMConfig(model=model, context_len=256, horizon=max(horizons), device="cpu", batch_size=16)
        )
    except Exception as exc:
        logger.warning("TSFM forecaster unavailable (%s) — skipping CQR block", exc)
        return {}

    frames: dict[int, pd.DataFrame] = {}
    for h in horizons:
        try:
            frames[h] = collect_horizon_frame(y, forecaster, h, 256, 40, 1, 16)
        except Exception as exc:
            logger.warning("TSFM h=%d failed: %s", h, exc)
    return frames


def _curve_snippet(curve: pd.DataFrame, nominal: float) -> str:
    sub = curve.iloc[(curve["nominal_coverage"] - nominal).abs().argsort()[:1]]
    if sub.empty:
        return "n/a"
    row = sub.iloc[0]
    return f"cov={_fmt_pct(row['empirical_coverage'])}, width={row['median_width']:.5f}"


def run(cfg: dict, max_folds: int | None, tsfm_stub: bool) -> dict:
    panel = load_panel()
    horizons = list(cfg["horizons"])
    models = ["naive", "xgboost"]
    min_cal = max(30, cfg["walk_forward"]["min_train"] // 40)

    payload: dict = {
        "git_commit": _git_short(),
        "panel": {"start": str(panel.index[0].date()), "end": str(panel.index[-1].date())},
        "min_cal": min_cal,
        "horizons": {},
        "tsfm_cqr": {},
    }

    for h in horizons:
        payload["horizons"][str(h)] = {"models": {}, "curve_xgboost_80": None}
        for model in models:
            frame = load_or_run_frame(panel, cfg, model, h, max_folds)
            entry: dict = {"n_obs": int(len(frame)), "levels": {}}
            for alpha in TARGET_ALPHAS:
                _, summ = calibrate_frame_point(frame, alpha=alpha, min_cal=min_cal)
                entry["levels"][NOMINAL[alpha]] = {
                    "empirical_coverage": summ.empirical_coverage,
                    "median_width": summ.median_width,
                    "median_width_rw": summ.median_width_rw,
                    "width_ratio_vs_rw": summ.width_ratio_vs_rw,
                    "n_eval": summ.n_eval,
                }
            if model == "xgboost":
                curve = coverage_width_curve(
                    frame["y_true"].to_numpy(),
                    frame["y_pred"].to_numpy(),
                    min_cal=min_cal,
                )
                payload["horizons"][str(h)]["curve_xgboost_80"] = _curve_snippet(curve, 0.80)
            payload["horizons"][str(h)]["models"][model] = entry

    tsfm_frames = tsfm_origin_frames(cfg, horizons, stub=tsfm_stub)
    for h, frame in tsfm_frames.items():
        raw = raw_quantile_coverage(frame)
        entry = {"n_origins": int(len(frame)), "raw_p10_p90_coverage": raw, "levels": {}}
        for alpha in TARGET_ALPHAS:
            _, summ = calibrate_frame_cqr(frame, alpha=alpha, min_cal=min(10, len(frame) // 3))
            entry["levels"][NOMINAL[alpha]] = {
                "empirical_coverage": summ.empirical_coverage,
                "median_width": summ.median_width,
                "median_width_rw": summ.median_width_rw,
                "width_ratio_vs_rw": summ.width_ratio_vs_rw,
            }
        curve = coverage_width_curve_cqr(
            frame["y_true"].to_numpy(),
            frame["q_lo"].to_numpy(),
            frame["q_hi"].to_numpy(),
            min_cal=min(10, len(frame) // 3),
        )
        entry["curve_near_80"] = _curve_snippet(curve, 0.80)
        payload["tsfm_cqr"][str(h)] = entry

    return payload


def _load_tsfm_raw_coverage() -> list[tuple[str, str, float, float]]:
    """(label, h, raw_coverage, width_pct) from committed full TSFM runs."""
    rows: list[tuple[str, str, float, float]] = []
    for label, sub in (
        ("TimesFM", "tsfm_full_timesfm"),
        ("Chronos-Bolt", "tsfm_full_chronos"),
    ):
        path = RESULTS / sub / "tsfm.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for h_key, block in data.get("horizons", {}).items():
            rows.append(
                (
                    label,
                    h_key,
                    float(block["interval_coverage_p10_p90"]),
                    float(block["median_interval_width_pct"]),
                )
            )
    return rows


def write_md(payload: dict, path: Path) -> None:
    lines = [
        "# Conformal calibration — BTC OOS intervals",
        "",
        f"Panel {payload['panel']['start']} → {payload['panel']['end']}. "
        f"Calibration scores use only OOS rows **before** each forecast origin "
        f"(expanding window, min_cal={payload['min_cal']}). "
        f"Commit `{payload.get('git_commit') or 'n/a'}`.",
        "",
        "Methods: **split-conformal** (symmetric residual bands on point forecasts) "
        "for walk-forward baselines; **CQR** on nominal p10–p90 TSFM intervals.",
        "",
        "Acceptance band for nominal 80%: empirical coverage **78–82%** per horizon.",
        "",
    ]

    raw_tsfm = _load_tsfm_raw_coverage()
    if raw_tsfm:
        lines.append("## TSFM — nominal p10–p90 before conformal (committed runs)")
        lines.append("")
        lines.append("| model | h | emp. coverage | median width (% of price) |")
        lines.append("|---|---|---|---|")
        for label, h_key, cov, w in raw_tsfm:
            lines.append(f"| {label} | {h_key} | {_fmt_pct(cov)} | {w:.2f}% |")
        lines.append("")
        lines.append(
            "TimesFM bands are too narrow (under ~80%); Chronos h=7 can be too wide (~100% on small origin counts). "
            "Rolling **CQR** below retargets the same quantiles to a chosen nominal level."
        )
        lines.append("")

    for h_key in sorted(payload["horizons"], key=int):
        block = payload["horizons"][h_key]
        lines.append(f"## Horizon h={h_key} (log-return targets)")
        lines.append("")
        lines.append(
            "| model | level | emp. coverage | median width | RW width | width / RW | n_eval |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for model, entry in block["models"].items():
            for level, row in entry["levels"].items():
                ok = ""
                if level == "80%":
                    c = row["empirical_coverage"]
                    if np.isfinite(c) and 0.78 <= c <= 0.82:
                        ok = " ✓"
                lines.append(
                    f"| {model} | {level} | {_fmt_pct(row['empirical_coverage'])}{ok} | "
                    f"{row['median_width']:.5f} | {row['median_width_rw']:.5f} | "
                    f"{_fmt_ratio(row['width_ratio_vs_rw'])} | {row['n_eval']} |"
                )
        if block.get("curve_xgboost_80"):
            lines.append("")
            lines.append(f"*XGBoost coverage–width near 80% nominal:* {block['curve_xgboost_80']}")
        lines.append("")

    if payload.get("tsfm_cqr"):
        lines.append("## TSFM price levels — CQR on p10–p90")
        lines.append("")
        lines.append("| h | n | raw p10–p90 cov. | CQR 80% cov. | CQR 95% cov. | width/RW @80% |")
        lines.append("|---|---|---|---|---|---|")
        for h_key in sorted(payload["tsfm_cqr"], key=int):
            e = payload["tsfm_cqr"][h_key]
            l80 = e["levels"]["80%"]
            l95 = e["levels"]["95%"]
            lines.append(
                f"| {h_key} | {e['n_origins']} | {_fmt_pct(e['raw_p10_p90_coverage'])} | "
                f"{_fmt_pct(l80['empirical_coverage'])} | {_fmt_pct(l95['empirical_coverage'])} | "
                f"{_fmt_ratio(l80['width_ratio_vs_rw'])} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Walk-forward frames from `results/preds/` when present; otherwise a fresh "
        "`run_walk_forward` (use `--max-folds` for a quick smoke run)."
    )
    lines.append(
        "- TSFM block uses `--tsfm-stub` by default (fast, tests pipeline); pass "
        "`--tsfm-real` to load TimesFM when installed."
    )
    lines.append(
        "- RW width benchmark: symmetric conformal interval around a **zero** log-return "
        "(or |y| score on past OOS) at the same nominal level."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Conformal calibration report")
    parser.add_argument("--max-folds", type=int, default=None, help="limit walk-forward folds when preds missing")
    parser.add_argument("--tsfm-real", action="store_true", help="use real TimesFM (slow; default is stub)")
    parser.add_argument("--out", default=str(RESULTS / "calibration.md"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config()
    t0 = time.time()
    stub = not args.tsfm_real
    payload = run(cfg, args.max_folds, tsfm_stub=stub)
    payload["runtime_s"] = round(time.time() - t0, 1)
    payload["tsfm_mode"] = "stub" if stub else "timesfm"

    out_md = Path(args.out)
    out_json = out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_md(payload, out_md)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s (%.1fs)", out_md, out_json, payload["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
