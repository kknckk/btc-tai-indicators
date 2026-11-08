"""Unified economic backtest + MCS/DM on a common OOS forecast panel.

Loads walk-forward prediction frames from ``results/preds/{model}_h{h}.parquet`` (written by
``experiments.run_baselines`` and optional neural/tsfm runners). Falls back to committed
``results/baselines.json`` for economic tables when parquets are missing.

Usage:
    python -m experiments.run_economic
    python -m experiments.run_economic --preds-dir results/preds
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from evaluation.backtest import (
    align_forecast_frames,
    backtest_on_frame,
    buy_hold_position,
    squared_return_losses,
    vol_target_position,
)
from evaluation.mcs import mcs
from evaluation.walk_forward import sign_position
from experiments.config import ROOT, load_config

logger = logging.getLogger("economic")
RESULTS = ROOT / "results"

BASELINE_MODELS = ["naive", "har_rv", "garch", "xgboost"]
EXTRA_MODELS = ["moderntcn", "patchtst", "chronos", "timesfm"]
ALL_MODELS = BASELINE_MODELS + EXTRA_MODELS


def _position_fn(model: str, cfg: dict):
    vt = cfg["vol_target"]
    ann = cfg["annualization"]
    if model == "naive":
        return buy_hold_position
    if model in ("har_rv", "garch"):
        return vol_target_position(vt["annual_vol"], vt["max_leverage"], ann)
    return sign_position


def load_pred_frames(preds_dir: Path, h: int, models: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for m in models:
        p = preds_dir / f"{m}_h{h}.parquet"
        if p.is_file():
            frames[m] = pd.read_parquet(p)
    return frames


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not pd.notna(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_md(
    cfg: dict,
    panel: dict,
    horizons: dict,
    preds_dir: Path,
    path: Path,
) -> None:
    c = cfg
    lines = [
        "# Economic evaluation — common OOS panel",
        "",
        f"Panel {panel.get('start', '?')} → {panel.get('end', '?')}. "
        f"Expanding walk-forward (min_train={c['walk_forward']['min_train']}, "
        f"step={c['walk_forward']['step']}, purge=h). "
        "Positions at close of day *t* earn ``next_ret`` (day *t+1*); costs on |Δposition| "
        f"at {c['costs_bps']} bps.",
        "",
        "Position rules: naive = buy & hold; har_rv/garch = vol-target "
        f"({c['vol_target']['annual_vol']:.0%} ann. vol, max lev {c['vol_target']['max_leverage']}); "
        "others = sign(forecast) long/short/flat.",
        "",
        f"Forecast parquets: `{preds_dir}/` (MCS/DM on squared return loss uses the date intersection of loaded models).",
        "",
    ]
    for h, block in horizons.items():
        lines += [f"## Horizon h={h}", ""]
        econ = block.get("economic", {})
        if econ:
            lines += [
                "| model | Sharpe 10bps | Sharpe 20bps | MaxDD 20bps | Turnover ann. | Hit rate 20bps |",
                "|---|---|---|---|---|---|",
            ]
            for m in ALL_MODELS:
                if m not in econ:
                    continue
                e10, e20 = econ[m]["10bps"], econ[m]["20bps"]
                lines.append(
                    f"| {m} | {_f(e10.get('sharpe'), 2)} | {_f(e20.get('sharpe'), 2)} | "
                    f"{_f(e20.get('max_dd'), 2)} | {_f(e20.get('turnover_ann'), 2)} | "
                    f"{_f(e20.get('hit_rate'), 3)} |"
                )
            lines.append("")
        mcs_out = block.get("mcs")
        if mcs_out:
            lines += [
                "### Model Confidence Set (bootstrap, α=0.10, squared return loss)",
                "",
                f"Confidence set: **{', '.join(mcs_out['confidence_set'])}**",
                "",
                "Mean loss rank (lower is better):",
                "",
                "| model | mean SE | elimination p |",
                "|---|---|---|",
            ]
            order = sorted(mcs_out["mean_loss"], key=lambda k: mcs_out["mean_loss"][k])
            for m in order:
                lines.append(
                    f"| {m} | {mcs_out['mean_loss'][m]:.6f} | {_f(mcs_out['elimination_p'].get(m), 3)} |"
                )
            lines += ["", "Pairwise DM p-values (row vs column, squared error):", ""]
            dm_p = mcs_out.get("dm_p", {})
            names = order
            if dm_p and names:
                header = "| | " + " | ".join(names) + " |"
                lines += [header, "|---|" + "---|" * len(names)]
                for a in names:
                    row = [a]
                    for b in names:
                        p = dm_p.get(a, {}).get(b)
                        row.append(_f(p, 3) if a != b else "—")
                    lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        elif block.get("from_baselines"):
            lines += [
                "*MCS not computed (no aligned prediction parquets). Run "
                "`python -m experiments.run_baselines` then re-run this script.*",
                "",
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict, preds_dir: Path, mcs_bootstrap: int = 2000) -> dict:
    horizons_out: dict = {}
    panel_meta: dict = {}
    baselines_path = RESULTS / "baselines.json"
    baselines = json.loads(baselines_path.read_text()) if baselines_path.is_file() else {}
    panel_meta = baselines.get("panel", {})

    for h in cfg["horizons"]:
        hs = str(h)
        frames = load_pred_frames(preds_dir, h, ALL_MODELS)
        block: dict = {"economic": {}, "from_baselines": not frames}

        if frames:
            try:
                _, aligned = align_forecast_frames(frames)
            except ValueError as e:
                logger.warning("h=%s: %s", h, e)
                aligned = frames
            for m, f in aligned.items():
                block["economic"][m] = backtest_on_frame(f, _position_fn(m, cfg), tuple(cfg["costs_bps"]))
            if len(aligned) >= 2:
                losses = squared_return_losses(aligned)
                block["mcs"] = mcs(losses, n_bootstrap=mcs_bootstrap, seed=cfg["seed"], h=h)
            else:
                logger.warning(
                    "h=%s: tylko %d model(i) z ramkami predykcji — MCS/DM pominięte (wymagane >= 2)",
                    h, len(aligned),
                )
                block["mcs_skipped"] = "fewer than 2 aligned models"
            block["from_baselines"] = False
            block["n_common_days"] = len(next(iter(aligned.values())))
            block["models_loaded"] = sorted(aligned)
        elif hs in baselines.get("horizons", {}):
            rh = baselines["horizons"][hs]
            for m in BASELINE_MODELS:
                if m in rh:
                    block["economic"][m] = rh[m]["metrics"]["economic"]
            block["models_loaded"] = BASELINE_MODELS
        horizons_out[hs] = block

    return {"config": cfg, "panel": panel_meta, "preds_dir": str(preds_dir), "horizons": horizons_out}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preds-dir", type=Path, default=RESULTS / "preds")
    ap.add_argument("--mcs-bootstrap", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    out = run(cfg, args.preds_dir, args.mcs_bootstrap)
    (RESULTS / "economic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(cfg, out["panel"], out["horizons"], args.preds_dir, RESULTS / "economic.md")
    logger.info("wrote results/economic.json and results/economic.md")


if __name__ == "__main__":
    main()
