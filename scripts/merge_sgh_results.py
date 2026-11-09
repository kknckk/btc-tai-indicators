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
    return out


def recompute_tier1_comparison(out: dict, runner) -> None:
    """Po scaleniu przelicza tier1_comparison z PELNEGO zbioru modeli tier2.

    Kopiowanie tego bloku z pliku czastkowego (np. GPU z 3 z 4 modeli)
    falszowaloby wniosek 'nailepszy model tier II' — patrz review f4e01bf.
    """
    if not hasattr(runner, "_best_tier1_skill"):
        return  # runner nie ma pojecia o tier1 — nie ma co przeliczac
    comp: dict = {}
    import math

    for hs, block in out.get("horizons", {}).items():
        t1_name, t1_skill = runner._best_tier1_skill(hs)
        best_t2, best_t2_skill = None, None
        for m, row in block.items():
            sk = row.get("skill_vs_naive")
            if sk is None or not math.isfinite(sk):
                continue
            if best_t2_skill is None or sk > best_t2_skill:
                best_t2_skill, best_t2 = float(sk), m
        comp[hs] = {
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
    if comp:
        out["tier1_comparison"] = comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path, help="canonical results JSON (updated in place)")
    ap.add_argument("extra", type=Path, help="JSON from an additional run (e.g. GPU)")
    args = ap.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    extra = json.loads(args.extra.read_text(encoding="utf-8"))
    out = merge(base, extra)
    recompute_tier1_comparison(out, importlib.import_module(RUNNER_BY_NAME[args.base.stem]))
    args.base.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = args.base.with_suffix(".md")
    module = importlib.import_module(RUNNER_BY_NAME[args.base.stem])
    module.write_md(out, md_path)
    print(f"merged {args.extra} -> {args.base}; md: {md_path}")


if __name__ == "__main__":
    main()
