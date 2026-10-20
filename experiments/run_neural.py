"""Train & OOS-evaluate ModernTCN and PatchTST on BTC log-returns.

Usage:
    python experiments/run_neural.py [--config experiments/configs/neural.yaml]
                                      [--smoke] [--output-dir results]

Data loading and walk-forward splitting both prefer the implementations
being built in the parallel branch (`experiments/data.py`,
`evaluation/walk_forward.py`) and fall back to small local
implementations here if those modules are not yet available, so a later
merge only requires deleting the fallback branches.

Writes:
    results/neural.json  -- per model x horizon metrics (+ n_params, train_seconds)
    results/neural.md    -- human-readable Markdown table of the same
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moderntcn import ModernTCN, ModernTCNConfig
from models.patchtst import PatchTST, PatchTSTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_neural")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Data loading: prefer the parallel branch's experiments/data.py, else fall
# back to models/_local_data.py (small, self-contained merge logic).
# ---------------------------------------------------------------------------
try:
    from experiments.data import load_panel  # type: ignore

    _DATA_SOURCE = "experiments.data.load_panel"
except ImportError:
    from models._local_data import load_panel

    _DATA_SOURCE = "models._local_data.load_panel (fallback)"

# ---------------------------------------------------------------------------
# Walk-forward splitting: prefer evaluation/walk_forward.py if present, else
# use the expanding-window + purge/embargo splitter defined below.
# ---------------------------------------------------------------------------
try:
    from evaluation.walk_forward import make_folds as run_walk_forward  # type: ignore

    _WF_SOURCE = "evaluation.walk_forward.make_folds"
except ImportError:
    run_walk_forward = None
    _WF_SOURCE = "local expanding_window_splits (fallback)"



def expanding_window_splits(
    n: int, horizon: int, n_folds: int = 5, min_train_frac: float = 0.6, embargo: int = None
) -> List[Dict[str, Tuple[int, int]]]:
    """Expanding-window OOS folds with purge + embargo (Lopez de Prado-style).

    - purge: the last `horizon` rows of each training fold are dropped,
      because their (horizon-ahead) labels overlap into the test fold and
      would leak future information.
    - embargo: an additional `embargo` rows are skipped between the end of
      a test fold and the right edge used to size the *next* fold, so a
      fresh buffer separates consecutive folds beyond the purge alone.

    Returns a list of `{"train": (start, end), "test": (start, end)}`
    (end-exclusive) index ranges into the 0..n-1 row space.
    """
    embargo = horizon if embargo is None else embargo
    min_train = int(n * min_train_frac)
    test_region = max(0, n - min_train)
    fold_size = max(horizon + 1, test_region // max(1, n_folds))

    folds = []
    train_end = min_train
    while True:
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_start >= n or test_end - test_start < 1:
            break
        purged_train_end = max(1, train_end - horizon)
        folds.append({"train": (0, purged_train_end), "test": (test_start, test_end)})
        train_end = test_end + embargo
        if train_end >= n:
            break
    return folds


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _load_panel_frame(feature_columns=None, min_coverage=0.5) -> pd.DataFrame:
    """Load the merged panel through whichever ``load_panel`` is available.

    The merged branch (``experiments/data.py``) returns a DataFrame indexed by a
    DatetimeIndex named ``time`` and takes no feature arguments; the local
    fallback takes ``feature_columns``/``min_coverage``. Normalise both to a frame
    that has a ``time`` column so the rest of this module is source-agnostic.
    """
    try:
        panel = load_panel(feature_columns=feature_columns, min_coverage=min_coverage)
    except TypeError:
        panel = load_panel()
        if not isinstance(panel.index, pd.RangeIndex):
            panel = panel.reset_index().rename(columns={panel.index.name or "index": "time"})
    return panel


def build_feature_frame(cfg: dict) -> pd.DataFrame:
    """Load the on-chain + price panel and attach engineered columns.

    Returns a DataFrame sorted by time, index reset to 0..n-1, containing
    `time`, `PriceUSD`, the requested on-chain feature columns, `r_t`
    (daily log-return) and, for each horizon `h` in cfg["horizons"]`,
    a column `target_h{h}` = sum of log-returns from t to t+h (NaN near
    the tail where undefined).
    """
    data_cfg = cfg.get("data", {})
    feature_columns = data_cfg.get("feature_columns")
    min_coverage = data_cfg.get("min_coverage", 0.5)

    df = _load_panel_frame(feature_columns=feature_columns, min_coverage=min_coverage)
    df = df.sort_values("time").reset_index(drop=True)
    df["log_price"] = np.log(df["PriceUSD"].astype(float))
    df["r_t"] = df["log_price"].diff()

    # The merged data layer exposes raw levels under ``slog1p_<metric>``; accept either name.
    feature_columns = [
        c if c in df.columns else f"slog1p_{c}"
        for c in (feature_columns or [])
        if c in df.columns or f"slog1p_{c}" in df.columns
    ]
    feature_cols = list(feature_columns)

    if data_cfg.get("include_lagged_return", True):
        feature_cols = feature_cols + ["r_t"]

    required = ["time", "PriceUSD", "log_price"] + feature_cols
    df = df.dropna(subset=required).reset_index(drop=True)

    for h in cfg["horizons"]:
        df[f"target_h{h}"] = df["log_price"].shift(-h) - df["log_price"]

    df.attrs["feature_cols"] = feature_cols
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    sse_model = float(np.sum(err**2))
    sse_naive = float(np.sum(y_true**2))  # random-walk null: predicted return = 0
    r2_oos = float(1.0 - sse_model / sse_naive) if sse_naive > 0 else float("nan")
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2_OOS_vs_RW": r2_oos,
        "directional_accuracy": dir_acc,
        "n_test": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(name: str, model_cfg: dict, seed: int):
    if name == "moderntcn":
        cfg = ModernTCNConfig(**{**model_cfg, "seed": seed})
        return ModernTCN(cfg), cfg.window
    if name == "patchtst":
        cfg = PatchTSTConfig(**{**model_cfg, "seed": seed})
        return PatchTST(cfg), cfg.window
    raise ValueError(f"Unknown model name: {name}")


def run_one_fold(model_name, model_cfg, seed, X: pd.DataFrame, y: pd.Series, train_range, test_range, window):
    tr_start, tr_end = train_range
    te_start, te_end = test_range
    X_train = X.iloc[tr_start:tr_end]
    y_train = y.iloc[tr_start:tr_end]

    model, _ = make_model(model_name, model_cfg, seed)
    model.fit(X_train, y_train)

    # Give predict() enough left-context to form windows for the first test row.
    ctx_start = max(0, te_start - window + 1)
    X_ctx = X.iloc[ctx_start:te_end]
    preds_ctx = model.predict(X_ctx)
    preds_test = preds_ctx[-(te_end - te_start) :]

    y_test = y.iloc[te_start:te_end].to_numpy(dtype=float)
    valid = ~np.isnan(preds_test) & ~np.isnan(y_test)
    return y_test[valid], preds_test[valid], model.n_params, model.epochs_used_, model.train_seconds_


def evaluate_model_horizon(model_name, model_cfg, seeds, X, y, wf_folds, window):
    all_true, all_pred = [], []
    n_params = None
    epochs_list, seconds_list = [], []

    for fold in wf_folds:
        seed_preds = []
        y_test_fold = None
        for seed in seeds:
            y_test, preds, n_p, epochs_used, train_s = run_one_fold(
                model_name, model_cfg, seed, X, y, fold["train"], fold["test"], window
            )
            if len(preds) == 0:
                continue
            seed_preds.append(preds)
            y_test_fold = y_test
            n_params = n_p
            epochs_list.append(epochs_used)
            seconds_list.append(train_s)
        if not seed_preds or y_test_fold is None:
            continue
        min_len = min(len(p) for p in seed_preds)
        avg_pred = np.mean([p[:min_len] for p in seed_preds], axis=0)
        all_true.append(y_test_fold[:min_len])
        all_pred.append(avg_pred)

    if not all_true:
        return None

    y_true_cat = np.concatenate(all_true)
    y_pred_cat = np.concatenate(all_pred)
    metrics = calc_metrics(y_true_cat, y_pred_cat)
    metrics["n_parameters"] = int(n_params) if n_params is not None else None
    metrics["train_seconds_total"] = float(np.sum(seconds_list)) if seconds_list else None
    metrics["train_seconds_per_fit"] = float(np.mean(seconds_list)) if seconds_list else None
    metrics["epochs_used_avg"] = float(np.mean(epochs_list)) if epochs_list else None
    metrics["n_folds_used"] = len(all_true)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(cfg: dict) -> dict:
    smoke = cfg.get("smoke", {})
    for model_name in ("moderntcn", "patchtst"):
        cfg["models"][model_name]["max_epochs"] = smoke.get("max_epochs", 3)
        cfg["models"][model_name]["patience"] = smoke.get("patience", 2)
    cfg["walk_forward"]["n_folds"] = smoke.get("n_folds", 2)
    cfg["_subsample_rows"] = smoke.get("subsample_rows")
    return cfg


def to_markdown_table(results: dict) -> str:
    lines = [
        "| Model | Horizon (d) | RMSE | MAE | R2_OOS_vs_RW | Dir. Acc. | n_params | train_s/fit | folds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model_name, by_horizon in results["results"].items():
        for horizon, m in by_horizon.items():
            if m is None:
                lines.append(f"| {model_name} | {horizon} | - | - | - | - | - | - | 0 |")
                continue
            lines.append(
                f"| {model_name} | {horizon} | {m['RMSE']:.6f} | {m['MAE']:.6f} | "
                f"{m['R2_OOS_vs_RW']:.4f} | {m['directional_accuracy']:.4f} | "
                f"{m['n_parameters']} | {m['train_seconds_per_fit']:.2f} | {m['n_folds_used']} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "experiments", "configs", "neural.yaml"),
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny epochs + subsample, for CI/local sanity checks.")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    torch.set_num_threads(2)

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)

    logger.info("Data source: %s", _DATA_SOURCE)
    logger.info("Walk-forward source: %s", _WF_SOURCE)

    df = build_feature_frame(cfg)
    if args.smoke and cfg.get("_subsample_rows"):
        df = df.iloc[-cfg["_subsample_rows"] :].reset_index(drop=True)
        # targets referencing rows beyond the subsample tail are NaN again
        for h in cfg["horizons"]:
            df[f"target_h{h}"] = df["log_price"].shift(-h) - df["log_price"]

    feature_cols = df.attrs.get("feature_cols") or [
        c for c in cfg["data"]["feature_columns"] if c in df.columns
    ] + (["r_t"] if cfg["data"].get("include_lagged_return", True) else [])
    logger.info("Panel: %d rows, features=%s", len(df), feature_cols)

    wf_cfg = cfg["walk_forward"]
    seeds = cfg["seeds"]

    results = {
        "meta": {
            "data_source": _DATA_SOURCE,
            "walk_forward_source": _WF_SOURCE,
            "n_rows": int(len(df)),
            "feature_columns": feature_cols,
            "horizons": cfg["horizons"],
            "seeds": seeds,
            "smoke": args.smoke,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        "results": {"moderntcn": {}, "patchtst": {}},
    }

    X_full = df[feature_cols]
    start_all = time.perf_counter()

    for model_name in ("moderntcn", "patchtst"):
        model_cfg = dict(cfg["models"][model_name])
        window = model_cfg["window"]
        for h in cfg["horizons"]:
            y_full = df[f"target_h{h}"]
            n = len(df)

            if run_walk_forward is not None:
                try:
                    wf_folds = run_walk_forward(
                        n=n, horizon=h, n_folds=wf_cfg["n_folds"], min_train_frac=wf_cfg["min_train_frac"]
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning("evaluation.walk_forward.run_walk_forward failed (%s); using local splitter.", exc)
                    wf_folds = expanding_window_splits(
                        n,
                        horizon=h,
                        n_folds=wf_cfg["n_folds"],
                        min_train_frac=wf_cfg["min_train_frac"],
                        embargo=h * wf_cfg.get("embargo_multiplier", 1),
                    )
            else:
                wf_folds = expanding_window_splits(
                    n,
                    horizon=h,
                    n_folds=wf_cfg["n_folds"],
                    min_train_frac=wf_cfg["min_train_frac"],
                    embargo=h * wf_cfg.get("embargo_multiplier", 1),
                )

            logger.info("[%s] horizon=%d: %d walk-forward folds", model_name, h, len(wf_folds))
            t0 = time.perf_counter()
            metrics = evaluate_model_horizon(model_name, model_cfg, seeds, X_full, y_full, wf_folds, window)
            elapsed = time.perf_counter() - t0
            if metrics is None:
                logger.warning("[%s] horizon=%d produced no OOS predictions.", model_name, h)
            else:
                logger.info(
                    "[%s] horizon=%d RMSE=%.6f MAE=%.6f R2_OOS=%.4f DirAcc=%.4f n_params=%d (%.1fs)",
                    model_name,
                    h,
                    metrics["RMSE"],
                    metrics["MAE"],
                    metrics["R2_OOS_vs_RW"],
                    metrics["directional_accuracy"],
                    metrics["n_parameters"] or -1,
                    elapsed,
                )
            results["results"][model_name][str(h)] = metrics

    results["meta"]["total_runtime_seconds"] = float(time.perf_counter() - start_all)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "neural.json")
    md_path = os.path.join(args.output_dir, "neural.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ModernTCN / PatchTST -- BTC log-return OOS results\n\n")
        f.write(f"- data source: `{_DATA_SOURCE}`\n")
        f.write(f"- walk-forward source: `{_WF_SOURCE}`\n")
        f.write(f"- rows used: {len(df)}, smoke={args.smoke}\n\n")
        f.write(to_markdown_table(results))
        f.write("\n")

    logger.info("Wrote %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
