"""Merge per-model SGH run results written by different runners/configs into the
canonical results/sgh_<series>_tier<N>.json and regenerate the markdown report.

Usecase: czesc modeli liczona na CPU (np. xgboost/svm), czesc na GPU (bilstm/lstm/tft...)
— runner nadpisuje JSON tylko modelami z biezacego configu, wiec sklejamy po kluczach
horizons[h][model]. Pozniejszy plik ma pierwszenstwo dla powtarzajacego sie modelu.

Usage:
  python scripts/merge_sgh_results.py results/sgh_nbp_rate_tier2.json results/extra/nbp_tier2_gpu.json
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

RUNNER_BY_NAME = {
    "sgh_nbp_rate_tier1": "experiments.run_sgh_nbp_tier1",
    "sgh_nbp_rate_tier2": "experiments.run_sgh_nbp_tier2",
    "sgh_bitcoin_tier1": "experiments.run_sgh_bitcoin_tier1",
    "sgh_bitcoin_tier2": "experiments.run_sgh_bitcoin_tier2",
}


def merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    horizons = dict(base.get("horizons", {}))
    for h, block in extra.get("horizons", {}).items():
        cur = dict(horizons.get(h, {}))
        cur.update(block)
        horizons[h] = cur
    out["horizons"] = horizons
    if extra.get("tier1_comparison"):
        comp = dict(base.get("tier1_comparison", {}))
        comp.update(extra["tier1_comparison"])
        out["tier1_comparison"] = comp
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path, help="canonical results JSON (updated in place)")
    ap.add_argument("extra", type=Path, help="JSON from an additional run (e.g. GPU)")
    args = ap.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    extra = json.loads(args.extra.read_text(encoding="utf-8"))
    out = merge(base, extra)
    args.base.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = args.base.with_suffix(".md")
    module = importlib.import_module(RUNNER_BY_NAME[args.base.stem])
    module.write_md(out, md_path)
    print(f"merged {args.extra} -> {args.base}; md: {md_path}")


if __name__ == "__main__":
    main()
