"""TRD-32 — NBP reference rate tier-I models (Taylor, forward linear, ordered logit/probit).

Usage:
  python -m experiments.run_sgh_nbp_tier1 [--demo] [--max-folds N] [--cache-only]

Writes ``results/sgh_nbp_rate_tier1.json`` and ``results/sgh_nbp_rate_tier1.md``.
Production runs require cached ``nbp_rate`` + GUS CPI/PMI (see ``data/macro_pl/SOURCES.md``).
``--demo`` uses a deterministic synthetic panel for smoke tests only.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.dm_test import dm_test
from evaluation.targets import TargetSpec
from evaluation.walk_forward import run_walk_forward, sign_position
from experiments.config import ROOT
from experiments.sgh.nbp_panel import build_nbp_monthly_panel, demo_nbp_panel, nbp_features
from experiments.sgh.tier1_nbp import model_factory

logger = logging.getLogger("sgh_nbp_tier1")
RESULTS = ROOT / "results"
MODELS = ("taylor_rule", "forward_linear", "ordered_logit", "ordered_probit")


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _target(h: int) -> TargetSpec:
    return TargetSpec(kind="level", price_col="nbp_rate", horizon=h, unit="month", naive_window=1)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, TargetSpec):
        return {"kind": obj.kind, "price_col": obj.price_col, "horizon": obj.horizon, "unit": obj.unit}
    return obj


def _naive_losses(panel: pd.DataFrame, spec: TargetSpec, index: pd.Index) -> np.ndarray:
    from evaluation.targets import benchmark_naive, forward_target

    price = panel["nbp_rate"]
    y = forward_target(price, spec).reindex(index)
    nv = benchmark_naive(price, spec, y)
    return ((y - nv) ** 2).to_numpy()


def run_fixed(cfg: dict, panel: pd.DataFrame, max_folds: int | None = None, demo: bool = False) -> dict:
    wf = cfg["walk_forward"]
    costs = tuple(cfg.get("costs_bps", (10, 20)))
    out: dict = {
        "series_id": cfg["series_id"],
        "demo_panel": demo,
        "panel": {
            "start": str(panel.index[0].date()),
            "end": str(panel.index[-1].date()),
            "n_obs": int(len(panel)),
        },
        "horizons": {},
    }
    for h in cfg["horizons"]:
        spec = _target(h)
        res_h: dict = {}
        frames: dict[str, pd.DataFrame] = {}
        price = panel["nbp_rate"]
        for name in MODELS:
            t0 = time.time()
            r = run_walk_forward(
                panel,
                h,
                nbp_features,
                model_factory(name, cfg),
                min_train=wf["min_train"],
                embargo=wf["embargo"],
                step=wf["step"],
                costs_bps=costs,
                seed=cfg["seed"],
                max_folds=max_folds,
                target=spec,
                series_id=cfg["series_id"],
                causal_check=not demo,
            )
            frames[name] = r["frame"]
            res_h[name] = {
                "metrics": r["metrics"],
                "n_folds": len(r["folds"]),
                "first_test": r["folds"][0]["test_start"],
                "last_test": r["folds"][-1]["test_end"],
                "runtime_s": round(time.time() - t0, 1),
            }
        idx = frames["taylor_rule"].index
        se = {m: ((f["y_true"] - f["y_pred"]) ** 2).reindex(idx).to_numpy() for m, f in frames.items()}
        naive_se = _naive_losses(panel, spec, idx)
        for m in MODELS:
            stat, p = dm_test(se[m], naive_se, h)
            res_h[m]["dm_mse_vs_naive"] = {"stat": stat, "p": p}
            rw = (frames[m]["y_true"] - frames[m]["price"]).to_numpy()
            rw_se = rw ** 2
            stat, p = dm_test(se[m], rw_se, h)
            res_h[m]["dm_mse_vs_rw"] = {"stat": stat, "p": p}
            vb = res_h[m]["metrics"].get("vs_benchmarks", {})
            mse_m = vb.get("mse_model", float("nan"))
            mse_rw = vb.get("mse_rw", float("nan"))
            mse_nv = vb.get("mse_naive", float("nan"))
            res_h[m]["skill_vs_rw"] = float(1.0 - mse_m / mse_rw) if mse_rw and mse_rw > 0 else None
            res_h[m]["skill_vs_naive"] = float(1.0 - mse_m / mse_nv) if mse_nv and mse_nv > 0 else None
        out["horizons"][str(h)] = res_h
    return out


def write_md(out: dict, path: Path) -> None:
    demo = out.get("demo_panel", False)
    lines = [
        "# TRD-32 — stopa referencyjna NBP, tier I",
        "",
        f"Seria: `{out['series_id']}` · panel {out['panel']['start']} → {out['panel']['end']} "
        f"({out['panel']['n_obs']} mies.)"
        + (" · **panel demo (--demo)**" if demo else ""),
        "",
        "Modele: reguła Taylora / reakcji, liniowy forward-looking (Ridge), ordered logit, ordered probit. "
        "Hiperparametry wybierane na końcówce train w każdym foldzie (bez dostrajania na teście). "
        "Harness: `evaluation/walk_forward.py` (sgh-2); dane: `experiments/macro_pl.py` (NBP + GUS CPI/PMI).",
        "",
        "| h (mies.) | model | RMSE | skill vs RW | skill vs naive | DM-p vs naive | DM-p vs RW |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for h, block in out["horizons"].items():
        for m in MODELS:
            row = block[m]
            pt = row["metrics"]["point"]
            dm_n = row.get("dm_mse_vs_naive") or {}
            dm_r = row.get("dm_mse_vs_rw") or {}
            lines.append(
                f"| {h} | {m} | {_f(pt.get('rmse'))} | {_f(row.get('skill_vs_rw'))} | "
                f"{_f(row.get('skill_vs_naive'))} | {_f(dm_n.get('p'))} | {_f(dm_r.get('p'))} |"
            )
    lines.extend(
        [
            "",
            "Skill = 1 − MSE_model/MSE_benchmark (wyżej = lepiej). DM: Diebold–Mariano na kwadratach błędu poziomu.",
            "",
            "Uruchomienie produkcyjne:",
            "```bash",
            "python -m experiments.macro_pl --check --series nbp_rate --series gus_cpi_yoy --series gus_pmi",
            "python -m experiments.run_sgh_nbp_tier1 --cache-only",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SGH TRD-32 NBP rate tier I")
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/configs/sgh_nbp_rate_tier1.yaml")
    ap.add_argument("--demo", action="store_true", help="synthetic monthly panel (smoke only)")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _load_cfg(args.config)
    if args.demo:
        panel = demo_nbp_panel()
    else:
        panel = build_nbp_monthly_panel(use_cache_only=args.cache_only)
    out = run_fixed(cfg, panel, max_folds=args.max_folds, demo=args.demo)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "sgh_nbp_rate_tier1.json"
    json_path.write_text(json.dumps(_clean(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out, RESULTS / "sgh_nbp_rate_tier1.md")
    logger.info("Wrote %s and %s", json_path, json_path.with_suffix(".md"))


if __name__ == "__main__":
    main()
