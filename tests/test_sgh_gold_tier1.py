"""TRD-56 tier-I gold models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_gold_tier1 import run_fixed
from experiments.sgh.gold_panel import PRICE_COL, demo_gold_panel, gold_features
from experiments.sgh.tier1_gold import model_factory
from experiments.sgh.tier1_wheat import LSTMLevelForecaster, XGBLevelForecaster
from experiments.sgh.tier1_wig20 import ArchLevelForecaster
from evaluation.walk_forward import check_causal


def test_gold_features_causal():
    panel = demo_gold_panel(400)
    bad = check_causal(gold_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_garch_fits_demo():
    panel = demo_gold_panel(320)
    X = gold_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = ArchLevelForecaster(
        "garch",
        panel["logret_1"],
        PRICE_COL,
        7,
        val_frac=0.2,
        mean_lags_grid=(0, 1),
        dist_grid=("normal",),
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-20:])
    assert np.isfinite(pred).all()


def test_xgb_fits_demo():
    panel = demo_gold_panel(280)
    X = gold_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = XGBLevelForecaster(
        {
            "val_frac": 0.2,
            "n_estimators": 30,
            "max_depth_grid": (3,),
            "learning_rate_grid": (0.05,),
            "early_stopping_rounds": 5,
        },
        seed=42,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_lstm_fits_demo():
    panel = demo_gold_panel(300)
    X = gold_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = LSTMLevelForecaster(
        {
            "val_frac": 0.2,
            "window_grid": (21,),
            "hidden_grid": (16,),
            "lr_grid": (1e-3,),
            "max_epochs": 3,
            "patience": 2,
        },
        seed=42,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_model_factory_keys():
    cfg = {"seed": 42, "val_frac": 0.2, "garch": {}, "lstm": {}, "xgboost": {}}
    panel = demo_gold_panel(100)
    states: dict = {}
    for name in ("lstm", "garch", "xgboost"):
        factory = model_factory(
            name,
            cfg,
            horizon=1,
            price_col=PRICE_COL,
            returns=panel["logret_1"],
            states=states,
        )
        assert callable(factory)
        assert factory() is not None


def test_run_sgh_gold_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_gold_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["lstm"]["window_grid"] = [21]
    cfg["lstm"]["hidden_grid"] = [16]
    cfg["lstm"]["lr_grid"] = [0.001]
    cfg["lstm"]["max_epochs"] = 3
    cfg["lstm"]["patience"] = 2
    cfg["garch"]["mean_lags_grid"] = [0]
    cfg["garch"]["dist_grid"] = ["normal"]
    cfg["xgboost"]["n_estimators"] = 30
    cfg["xgboost"]["max_depth_grid"] = [3]
    cfg["xgboost"]["learning_rate_grid"] = [0.05]
    panel = demo_gold_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in ("lstm", "garch", "xgboost"):
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
