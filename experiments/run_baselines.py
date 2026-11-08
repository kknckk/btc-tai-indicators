"""Run naive / HAR-RV / GARCH / XGBoost through the walk-forward harness.

Usage: python -m experiments.run_baselines [--max-folds N]
Writes results/baselines.json, results/baselines.md and results/preds/*.parquet (ignored by git).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import metrics as M
from evaluation.dm_test import average_rank, dm_matrix, dm_test
from evaluation.backtest import buy_hold_position, vol_target_position
from evaluation.walk_forward import run_walk_forward, sign_position, summarize
from experiments.baselines import GarchVol, HarRV, NaiveRW, XGBForecaster
from experiments.config import ROOT, load_config
from experiments.data import build_features, load_panel

logger = logging.getLogger("baselines")
RESULTS = ROOT / "results"
MODELS = ["naive", "har_rv", "garch", "xgboost"]


def _buy_hold(pred, vol=None):
    return buy_hold_position(pred, vol)


def _vol_target(cfg):
    vt = cfg["vol_target"]
    return vol_target_position(vt["annual_vol"], vt["max_leverage"], cfg["annualization"])


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(cfg: dict, max_folds: int | None = None) -> dict:
    panel = load_panel()
    ret = panel["logret_1"]
    wf = cfg["walk_forward"]
    costs = tuple(cfg["costs_bps"])
    since = cfg["eval_start_subperiod"]
    out: dict = {"config": cfg, "panel": {"start": str(panel.index[0].date()), "end": str(panel.index[-1].date()),
                                           "n_cols": int(panel.shape[1])}, "horizons": {}}
    for h in cfg["horizons"]:
        garch_state: dict = {}
        factories = {
            "naive": (lambda: NaiveRW(ret), _buy_hold),
            "har_rv": (lambda h=h: HarRV(ret, h), _vol_target(cfg)),
            "garch": (lambda h=h: GarchVol(ret, h, garch_state, **cfg["garch"]), _vol_target(cfg)),
            "xgboost": (lambda: XGBForecaster(cfg["xgboost"], cfg["seed"]), sign_position),
        }
        res_h: dict = {}
        frames: dict[str, pd.DataFrame] = {}
        for name, (fac, pos_fn) in factories.items():
            t0 = time.time()
            r = run_walk_forward(panel, h, build_features, fac, min_train=wf["min_train"], embargo=wf["embargo"],
                                 step=wf["step"], position_fn=pos_fn, costs_bps=costs, seed=cfg["seed"],
                                 max_folds=max_folds)
            frames[name] = r["frame"]
            r["frame"].to_parquet(RESULTS / "preds" / f"{name}_h{h}.parquet")
            entry = {"metrics": r["metrics"], "metrics_since_2018": summarize(r["frame"], pos_fn, costs, since),
                     "n_folds": len(r["folds"]), "first_test": r["folds"][0]["test_start"],
                     "last_test": r["folds"][-1]["test_end"], "runtime_s": round(time.time() - t0, 1)}
            if name == "garch":
                entry["fit_failures"] = garch_state.get("failures", 0)
            if name == "har_rv":
                coefs = pd.DataFrame([f["diagnostics"]["coef"] for f in r["folds"]])
                entry["har_coef_mean"] = coefs.mean().to_dict()
            if name == "xgboost":
                g = pd.DataFrame([f["diagnostics"]["gain"] for f in r["folds"]]).fillna(0.0)
                entry["top10_features_gain"] = g.mean().sort_values(ascending=False).head(10).to_dict()
                entry["mean_best_iteration"] = float(np.mean([f["diagnostics"]["best_iteration"] for f in r["folds"]]))
            res_h[name] = entry
            logger.info("h=%d %-8s rmse=%.5f r2=%.4f %.0fs", h, name, entry["metrics"]["point"]["rmse"],
                        entry["metrics"]["point"]["r2_oos_vs_rw"], entry["runtime_s"])
        # Diebold-Mariano
        idx = frames["naive"].index
        se = {m: ((f["y_true"] - f["y_pred"]) ** 2).reindex(idx).to_numpy() for m, f in frames.items()}
        ql = {m: M.qlike(f["rv_true"].reindex(idx).to_numpy(), f["var_pred"].reindex(idx).to_numpy())
              for m, f in frames.items() if "var_pred" in f}
        for m in MODELS:
            res_h[m]["dm_ret_vs_naive"] = dict(zip(("stat", "p"), dm_test(se[m], se["naive"], h))) if m != "naive" else None
            res_h[m]["dm_qlike_vs_naive"] = dict(zip(("stat", "p"), dm_test(ql[m], ql["naive"], h))) if m in ql and m != "naive" else None
        st, pv = dm_matrix(ql, h)
        res_h["dm_matrix_qlike"] = {"stat": st.to_dict(), "p": pv.to_dict()}
        out["horizons"][str(h)] = res_h
        out.setdefault("_mean_se", {})[str(h)] = {m: float(np.mean(v)) for m, v in se.items()}
        out.setdefault("_mean_ql", {})[str(h)] = {m: float(np.mean(v)) for m, v in ql.items()}
    out["avg_rank_return_mse"] = average_rank(list(out["_mean_se"].values())).to_dict()
    out["avg_rank_qlike"] = average_rank(list(out["_mean_ql"].values())).to_dict()
    del out["_mean_se"], out["_mean_ql"]
    return out


def _f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_md(out: dict, path: Path) -> None:
    c = out["config"]
    lines = ["# Baselines — walk-forward results", "",
             f"Panel {out['panel']['start']} → {out['panel']['end']}, seed {c['seed']}, expanding window, "
             f"min_train={c['walk_forward']['min_train']}, step={c['walk_forward']['step']}, "
             f"purge=h, extra embargo={c['walk_forward']['embargo']}. Costs on |Δposition|.", "",
             "Positions: naive = buy&hold (zero-return forecast carries no signal, so B&H is the economic RW "
             "benchmark); har_rv/garch = long-only vol-targeting "
             f"({c['vol_target']['annual_vol']:.0%} ann. vol, max lev {c['vol_target']['max_leverage']}); "
             "xgboost = sign(forecast) long/short. DM-p (ret) = Diebold-Mariano (HLN) on squared return error vs "
             "naive; har_rv/garch forecast zero return, so it is identical to naive (n/a) — their skill is in "
             "the variance table.", "",
             "## Return forecasts (full OOS sample)", "",
             "| model | h | RMSE | MAPE price % | R² OOS vs RW | dir-acc | Sharpe net 10bps | Sharpe net 20bps | MaxDD 20bps | DM-p vs naive (ret) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, rh in out["horizons"].items():
        for m in MODELS:
            e = rh[m]["metrics"]; pt = e["point"]; ec = e["economic"]
            dm = rh[m]["dm_ret_vs_naive"]
            lines.append(f"| {m} | {h} | {pt['rmse']:.5f} | {pt['mape_price_pct']:.2f} | {_f(pt['r2_oos_vs_rw'])} | "
                         f"{_f(pt['dir_acc'], 3)} | {_f(ec['10bps']['sharpe'], 2)} | {_f(ec['20bps']['sharpe'], 2)} | "
                         f"{_f(ec['20bps']['max_dd'], 2)} | {_f(dm['p'] if dm else None, 3)} |")
    lines += ["", f"## Same, sub-period since {c['eval_start_subperiod']}", "",
              "| model | h | RMSE | R² OOS vs RW | dir-acc | Sharpe net 20bps | MaxDD 20bps |", "|---|---|---|---|---|---|---|"]
    for h, rh in out["horizons"].items():
        for m in MODELS:
            e = rh[m]["metrics_since_2018"]; pt = e["point"]; ec = e["economic"]
            lines.append(f"| {m} | {h} | {pt['rmse']:.5f} | {_f(pt['r2_oos_vs_rw'])} | {_f(pt['dir_acc'], 3)} | "
                         f"{_f(ec['20bps']['sharpe'], 2)} | {_f(ec['20bps']['max_dd'], 2)} |")
    lines += ["", "## Variance forecasts (target: mean daily r² over next h days)", "",
              "naive = trailing-30d mean r². Lower QLIKE is better; DM on QLIKE loss vs naive.", "",
              "| model | h | QLIKE | RMSE vol (daily) | DM stat vs naive | DM-p vs naive |", "|---|---|---|---|---|---|"]
    for h, rh in out["horizons"].items():
        for m in ("naive", "har_rv", "garch"):
            v = rh[m]["metrics"]["vol"]; dm = rh[m]["dm_qlike_vs_naive"]
            lines.append(f"| {m} | {h} | {v['qlike']:.4f} | {v['rmse_vol_daily']:.5f} | "
                         f"{_f(dm['stat'], 2) if dm else 'n/a'} | {_f(dm['p'] if dm else None, 4)} |")
    lines += ["", "## Average rank across horizons (1 = best)", "",
              "| model | avg rank (return MSE) | avg rank (QLIKE) |", "|---|---|---|"]
    r1, r2 = out["avg_rank_return_mse"]["avg_rank"], out["avg_rank_qlike"]["avg_rank"]
    for m in MODELS:
        lines.append(f"| {m} | {_f(r1.get(m), 2)} | {_f(r2.get(m), 2)} |")
    lines += ["", "(ties in return MSE: naive/har_rv/garch share identical zero forecasts, hence equal rank.)", ""]
    for h, rh in out["horizons"].items():
        lines += [f"## XGBoost top-10 features by mean normalised gain — h={h}", "", "| feature | gain share |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in rh["xgboost"]["top10_features_gain"].items()]
        lines.append("")
    lines += ["## Run info", ""]
    for h, rh in out["horizons"].items():
        lines.append(f"- h={h}: {rh['naive']['n_folds']} folds, test {rh['naive']['first_test']} → {rh['naive']['last_test']}, "
                     f"n_obs={rh['naive']['metrics']['n_obs']}, GARCH fit failures={rh['garch'].get('fit_failures', 0)}, "
                     f"XGB mean best_iter={rh['xgboost']['mean_best_iteration']:.0f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "preds").mkdir(exist_ok=True)
    out = _clean(run(cfg, args.max_folds))
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(out, RESULTS / "baselines.md")
    logger.info("wrote results/baselines.json and baselines.md")


if __name__ == "__main__":
    main()
