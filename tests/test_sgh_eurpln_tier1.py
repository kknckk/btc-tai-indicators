"""TRD-46 tier-I EUR/PLN models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_eurpln_tier1 import run_fixed
from experiments.sgh.eurpln_panel import PRICE_COL, demo_eurpln_panel, eurpln_features
from experiments.sgh.tier1_eurpln import PatchTSTLevelForecaster, StackingEnsembleForecaster
from evaluation.walk_forward import check_causal


def test_eurpln_features_causal():
    panel = demo_eurpln_panel(400)
    bad = check_causal(eurpln_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_patchtst_fits_demo():
    panel = demo_eurpln_panel(300)
    X = eurpln_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = PatchTSTLevelForecaster(
        {
            "window": 16,
            "patch_len": 4,
            "stride": 4,
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "max_epochs": 3,
            "patience": 2,
        },
        seed=42,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_stacking_fits_demo():
    panel = demo_eurpln_panel(280)
    X = eurpln_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    cfg = {
        "val_frac": 0.2,
        "xgb": {"n_estimators": 20, "max_depth": 3, "early_stopping_rounds": 5},
        "gru": {"window": 21, "hidden": 16, "max_epochs": 3, "patience": 2},
    }
    model = StackingEnsembleForecaster(cfg, seed=42, horizon=1, price_col=PRICE_COL)
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert np.isfinite(pred).all()


def test_run_sgh_eurpln_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_eurpln_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["patchtst"] = {
        "window": 16,
        "patch_len": 4,
        "stride": 4,
        "d_model": 16,
        "nhead": 2,
        "num_layers": 1,
        "max_epochs": 3,
        "patience": 2,
    }
    cfg["stacking"]["xgb"]["n_estimators"] = 30
    cfg["stacking"]["gru"]["max_epochs"] = 3
    cfg["pso_lstm"]["n_particles"] = 2
    cfg["pso_lstm"]["n_iters"] = 1
    cfg["pso_lstm"]["gru"]["max_epochs"] = 3
    panel = demo_eurpln_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in ("patchtst", "stacking", "pso_lstm"):
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
