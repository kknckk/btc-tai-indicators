"""TRD-37 — consumer credit tier-II (VAR/BVAR, factors, FCI, LASSO/EN, ML, PKO proxy).

Usage:
  python -m experiments.run_sgh_credit_consumer_tier2 [--demo] [--max-folds N] [--cache-only]

Writes ``results/sgh_credit_consumer_tier2.json`` and ``results/sgh_credit_consumer_tier2.md``.
Compares skill to tier I when ``results/sgh_credit_consumer_tier1.json`` exists.
Production runs require cached NBP monetary + macro series; optional BIK CSV — **do weryfikacji**.
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
from evaluation.walk_forward import run_walk_forward
from experiments.config import ROOT
from experiments.sgh.credit_consumer_panel import (
    build_credit_consumer_monthly_panel,
    credit_consumer_tier2_features,
    demo_credit_consumer_panel,
)
from experiments.sgh.tier2_credit_consumer import model_factory

logger = logging.getLogger("sgh_credit_consumer_tier2")
RESULTS = ROOT / "results"
TIER1_MODELS = ("ardl", "ecm", "dynamic_regression", "logit_scorecard")
MODELS = (
    "var",
    "bvar",
    "factor_model",
    "fci",
    "lasso",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
    "pko_internal",
)
PRICE_COL = "credit_consumer"


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _target(h: int) -> TargetSpec:
    return TargetSpec(kind="level", price_col=PRICE_COL, horizon=h, unit="month", naive_window=1)


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

    price = panel[PRICE_COL]
    y = forward_target(price, spec).reindex(index)
    nv = benchmark_naive(price, spec, y)
    return ((y - nv) ** 2).to_numpy()


def _best_tier1_skill(horizon: str) -> tuple[str | None, float | None]:
    path = RESULTS / "sgh_credit_consumer_tier1.json"
    if not path.is_file():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("horizons", {}).get(horizon, {})
    best_name, best_skill = None, None
    for name in TIER1_MODELS:
        row = block.get(name, {})
        sk = row.get("skill_vs_naive")
        if sk is None or not np.isfinite(sk):
            continue
        if best_skill is None or sk > best_skill:
            best_skill, best_name = float(sk), name
    return best_name, best_skill


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
        "tier1_comparison": {},
    }
    for h in cfg["horizons"]:
        spec = _target(h)
        res_h: dict = {}
        frames: dict[str, pd.DataFrame] = {}
        for name in MODELS:
            t0 = time.time()
            r = run_walk_forward(
                panel,
                h,
                credit_consumer_tier2_features,
                model_factory(name, cfg, horizon=h),
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
        idx = frames["var"].index
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
        hs = str(h)
        out["horizons"][hs] = res_h
        t1_name, t1_skill = _best_tier1_skill(hs)
        best_t2, best_t2_skill = None, None
        for m in MODELS:
            sk = res_h[m].get("skill_vs_naive")
            if sk is None or not np.isfinite(sk):
                continue
            if best_t2_skill is None or sk > best_t2_skill:
                best_t2_skill, best_t2 = float(sk), m
        out["tier1_comparison"][hs] = {
            "best_tier1_model": t1_name,
            "best_tier1_skill_vs_naive": t1_skill,
            "best_tier2_model": best_t2,
            "best_tier2_skill_vs_naive": best_t2_skill,
            "tier2_beats_tier1_skill": (
                None
                if t1_skill is None or best_t2_skill is None
                else bool(best_t2_skill > t1_skill)
            ),
        }
    return out


def write_md(out: dict, path: Path) -> None:
    demo = out.get("demo_panel", False)
    lines = [
        "# TRD-37 — kredyty konsumpcyjne (NBP), tier II",
        "",
        f"Seria: `{out['series_id']}` · panel {out['panel']['start']} → {out['panel']['end']} "
        f"({out['panel']['n_obs']} mies.)"
        + (" · **panel demo (--demo)**" if demo else ""),
        "",
        "Modele: VAR, BVAR (ridge), PCA+AR (czynniki), FCI (warunki finansowe), LASSO, elastic net, "
        "random forest, gradient boosting, proxy modeli wewnętrznych PKO (ECM+FCI+momentum). "
        "Harness: `evaluation/walk_forward.py`; dane: NBP monetarna, GUS CPI, opcjonalnie BIK — **do weryfikacji**.",
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
    lines.extend(["", "## Tier II vs tier I (skill vs naive)", ""])
    for h, cmp in out.get("tier1_comparison", {}).items():
        beats = cmp.get("tier2_beats_tier1_skill")
        flag = "tak" if beats is True else ("nie" if beats is False else "n/a (brak tier I)")
        lines.append(
            f"- h={h}: najlepszy tier II `{cmp.get('best_tier2_model')}` "
            f"(skill {_f(cmp.get('best_tier2_skill_vs_naive'))}) vs tier I `{cmp.get('best_tier1_model')}` "
            f"(skill {_f(cmp.get('best_tier1_skill_vs_naive'))}) — tier II lepszy: **{flag}**"
        )
    lines.extend(
        [
            "",
            "Skill = 1 − MSE_model/MSE_benchmark. Złożoność (VAR/ML) nie gwarantuje poprawy względem tier I; "
            "sekcja powyżej porównuje najlepszy skill vs naive w każdym horyzoncie.",
            "",
            "Uruchomienie:",
            "```bash",
            "python -m experiments.macro_pl --check --series credit_consumer --series nbp_rate --series gus_cpi_yoy",
            "python -m experiments.run_sgh_credit_consumer_tier2 --cache-only",
            "pytest tests/test_sgh_credit_consumer_tier2.py -q",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SGH TRD-37 consumer credit tier II")
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/configs/sgh_credit_consumer_tier2.yaml")
    ap.add_argument("--demo", action="store_true", help="synthetic monthly panel (smoke only)")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _load_cfg(args.config)
    if args.demo:
        panel = demo_credit_consumer_panel()
    else:
        panel = build_credit_consumer_monthly_panel(use_cache_only=args.cache_only)
    out = run_fixed(cfg, panel, max_folds=args.max_folds, demo=args.demo)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "sgh_credit_consumer_tier2.json"
    json_path.write_text(json.dumps(_clean(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out, RESULTS / "sgh_credit_consumer_tier2.md")
    logger.info("Wrote %s and %s", json_path, json_path.with_suffix(".md"))


if __name__ == "__main__":
    main()
