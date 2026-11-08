"""TRD-58 tier-I bitcoin models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import yaml

from experiments.config import ROOT
from experiments.run_sgh_bitcoin_tier1 import run_fixed
from experiments.sgh.bitcoin_panel import PRICE_COL, bitcoin_features, demo_bitcoin_panel
from experiments.sgh.tier1_bitcoin import BiLSTMLevelForecaster, model_factory
from experiments.sgh.tier1_wheat import XGBLevelForecaster
from evaluation.walk_forward import check_causal


def test_bitcoin_features_causal():
    panel = demo_bitcoin_panel(400)
    bad = check_causal(bitcoin_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_xgb_fits_demo():
    panel = demo_bitcoin_panel(280)
    X = bitcoin_features(panel).dropna()
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


def test_bilstm_fits_demo():
    panel = demo_bitcoin_panel(300)
    X = bitcoin_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = BiLSTMLevelForecaster(
        {
            "val_frac": 0.2,
            "window_grid": (30,),
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
    cfg = {"seed": 42, "val_frac": 0.2, "bilstm": {}, "xgboost": {}}
    for name in ("bilstm", "xgboost"):
        factory = model_factory(name, cfg, horizon=1, price_col=PRICE_COL)
        assert callable(factory)
        assert factory() is not None


def test_run_sgh_bitcoin_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_bitcoin_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["bilstm"]["window_grid"] = [30]
    cfg["bilstm"]["hidden_grid"] = [16]
    cfg["bilstm"]["lr_grid"] = [0.001]
    cfg["bilstm"]["max_epochs"] = 3
    cfg["bilstm"]["patience"] = 2
    cfg["xgboost"]["n_estimators"] = 30
    cfg["xgboost"]["max_depth_grid"] = [3]
    cfg["xgboost"]["learning_rate_grid"] = [0.05]
    panel = demo_bitcoin_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in ("bilstm", "xgboost"):
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
