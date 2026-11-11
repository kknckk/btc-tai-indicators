"""TRD-35 — PL bond yields 2/5/7/10Y tier-II (PCA/VAR/macro/AFNS/LSTM).

Usage:
  python -m experiments.run_sgh_bonds_tier2 [--demo] [--max-folds N] [--cache-only]

Writes ``results/sgh_bonds_pl_tier2.json`` and ``results/sgh_bonds_pl_tier2.md``.
Compares skill to tier I when ``results/sgh_bonds_pl_tier1.json`` exists.
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
from experiments.sgh.bonds_panel import (
    BOND_COLS,
    bonds_tier2_features,
    build_bonds_tier2_daily_panel,
    demo_bonds_panel,
)
from experiments.sgh.tier2_bonds import model_factory

logger = logging.getLogger("sgh_bonds_tier2")
RESULTS = ROOT / "results"
TIER1_MODELS = ("nelson_siegel", "svensson", "diebold_li")
MODELS = (
    "pca_curve",
    "var",
    "bvar",
    "varx",
    "macro_finance",
    "afns",
    "pca_lstm",
    "lstm",
)


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _target(h: int, tenor: str) -> TargetSpec:
    return TargetSpec(kind="level", price_col=tenor, horizon=h, unit="day", naive_window=1)


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

    price = panel[spec.price_col]
    y = forward_target(price, spec).reindex(index)
    nv = benchmark_naive(price, spec, y)
    return ((y - nv) ** 2).to_numpy()


def _best_tier1_skill(tenor: str, horizon: str) -> tuple[str | None, float | None]:
    path = RESULTS / "sgh_bonds_pl_tier1.json"
    if not path.is_file():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("tenors", {}).get(tenor, {}).get("horizons", {}).get(horizon, {})
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
    tenors = cfg.get("tenors", list(BOND_COLS))
    out: dict = {
        "series_id": cfg["series_id"],
        "demo_panel": demo,
        "panel": {
            "start": str(panel.index[0].date()),
            "end": str(panel.index[-1].date()),
            "n_obs": int(len(panel)),
        },
        "tenors": {},
        "tier1_comparison": {},
    }

    def features(p: pd.DataFrame) -> pd.DataFrame:
        return bonds_tier2_features(p, ns_lambda=float(cfg.get("ns_lambda", 0.07)))

    for tenor in tenors:
        out["tenors"][tenor] = {"horizons": {}}
        out["tier1_comparison"][tenor] = {}
        for h in cfg["horizons"]:
            spec = _target(h, tenor)
            res_h: dict = {}
            frames: dict[str, pd.DataFrame] = {}
            for name in MODELS:
                t0 = time.time()
                r = run_walk_forward(
                    panel,
                    h,
                    features,
                    model_factory(name, cfg, horizon=h, target_col=tenor),
                    min_train=wf["min_train"],
                    embargo=wf["embargo"],
                    step=wf["step"],
                    costs_bps=costs,
                    seed=cfg["seed"],
                    max_folds=max_folds,
            model_name=name,
                    target=spec,
                    series_id=f"{cfg['series_id']}/{tenor}",
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
            out["tenors"][tenor]["horizons"][hs] = res_h
            t1_name, t1_skill = _best_tier1_skill(tenor, hs)
            best_t2, best_t2_skill = None, None
            for m in MODELS:
                sk = res_h[m].get("skill_vs_naive")
                if sk is None or not np.isfinite(sk):
                    continue
                if best_t2_skill is None or sk > best_t2_skill:
                    best_t2_skill, best_t2 = float(sk), m
            out["tier1_comparison"][tenor][hs] = {
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
        "# TRD-35 — rentowności obligacji PL 2/5/7/10Y, tier II",
        "",
        f"Seria: `{out['series_id']}` · panel {out['panel']['start']} → {out['panel']['end']} "
        f"({out['panel']['n_obs']} dni)"
        + (" · **panel demo (--demo)**" if demo else ""),
        "",
        "Modele: PCA krzywej, VAR/BVAR/VARX na czynnikach NS, makrofinansowy Ridge, "
        "AFNS-lite (NS + VAR), PCA-LSTM, LSTM na lagach. Makro: `nbp_rate`, `gus_cpi_yoy` (ffill). "
        "Harness: `evaluation/walk_forward.py`; dane: MF/GPW SPW — **do weryfikacji**.",
        "",
        "| tenor | h (dni) | model | RMSE | skill vs RW | skill vs naive | DM-p vs naive | DM-p vs RW |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for tenor, tblock in out["tenors"].items():
        short = tenor.replace("bond_pl_", "")
        for h, block in tblock["horizons"].items():
            for m in MODELS:
                row = block[m]
                pt = row["metrics"]["point"]
                dm_n = row.get("dm_mse_vs_naive") or {}
                dm_r = row.get("dm_mse_vs_rw") or {}
                lines.append(
                    f"| {short} | {h} | {m} | {_f(pt.get('rmse'))} | {_f(row.get('skill_vs_rw'))} | "
                    f"{_f(row.get('skill_vs_naive'))} | {_f(dm_n.get('p'))} | {_f(dm_r.get('p'))} |"
                )
    lines.extend(["", "## Tier II vs tier I (skill vs naive)", ""])
    for tenor, cmp_block in out.get("tier1_comparison", {}).items():
        short = tenor.replace("bond_pl_", "")
        for h, cmp in cmp_block.items():
            beats = cmp.get("tier2_beats_tier1_skill")
            flag = "tak" if beats is True else ("nie" if beats is False else "n/a (brak tier I)")
            lines.append(
                f"- {short} h={h}: najlepszy tier II `{cmp.get('best_tier2_model')}` "
                f"(skill {_f(cmp.get('best_tier2_skill_vs_naive'))}) vs tier I `{cmp.get('best_tier1_model')}` "
                f"(skill {_f(cmp.get('best_tier1_skill_vs_naive'))}) — tier II lepszy: **{flag}**"
            )
    lines.extend(
        [
            "",
            "Skill = 1 − MSE_model/MSE_benchmark. Złożoność (VAR/LSTM) nie gwarantuje poprawy względem tier I; "
            "sekcja powyżej porównuje najlepszy skill vs naive per tenor/horyzont.",
            "",
            "Uruchomienie:",
            "```bash",
            "python -m experiments.macro_pl --check --series bond_pl_2y --series bond_pl_10y --series nbp_rate",
            "python -m experiments.run_sgh_bonds_tier2 --cache-only",
            "pytest tests/test_sgh_bonds_tier2.py -q",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SGH TRD-35 PL bond yields tier II")
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/configs/sgh_bonds_pl_tier2.yaml")
    ap.add_argument("--demo", action="store_true", help="synthetic daily panel (smoke only)")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _load_cfg(args.config)
    if args.demo:
        panel = demo_bonds_panel()
    else:
        panel = build_bonds_tier2_daily_panel(use_cache_only=args.cache_only)
    out = run_fixed(cfg, panel, max_folds=args.max_folds, demo=args.demo)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "sgh_bonds_pl_tier2.json"
    json_path.write_text(json.dumps(_clean(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out, RESULTS / "sgh_bonds_pl_tier2.md")
    logger.info("Wrote %s and %s", json_path, json_path.with_suffix(".md"))


if __name__ == "__main__":
    main()
