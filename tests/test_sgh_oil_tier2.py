"""TRD-55 tier-II WTI oil models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_oil_tier2 import MODELS, run_fixed
from experiments.sgh.oil_panel import PRICE_COL, demo_oil_panel, oil_tier2_features
from experiments.sgh.tier2_oil import EMDLevelForecaster, OilStackingForecaster, model_factory
from evaluation.walk_forward import check_causal


def test_oil_tier2_features_causal():
    panel = demo_oil_panel(400)
    bad = check_causal(oil_tier2_features, panel, n_probes=3, seed=55)
    assert bad == []


def test_emd_fits_demo():
    panel = demo_oil_panel(320)
    X = oil_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = EMDLevelForecaster({"val_frac": 0.2, "ridge_alpha_grid": (1.0,)}, seed=42, price_col=PRICE_COL)
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_stacking_fits_demo():
    panel = demo_oil_panel(350)
    X = oil_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_oil_tier2.yaml").read_text(encoding="utf-8"))
    cfg["lstm_svr"]["window_grid"] = [21]
    cfg["lstm_svr"]["hidden_grid"] = [16]
    cfg["lstm_svr"]["lr_grid"] = [0.001]
    cfg["lstm_svr"]["max_epochs"] = 3
    cfg["lstm_svr"]["patience"] = 2
    cfg["lstm_svr"]["svr_c_grid"] = [1.0]
    cfg["lstm_svr"]["svr_gamma_grid"] = ["scale"]
    cfg["xgboost"]["n_estimators"] = 30
    cfg["xgboost"]["max_depth_grid"] = [3]
    cfg["xgboost"]["learning_rate_grid"] = [0.05]
    cfg["garch_midas"]["k_grid"] = [3]
    cfg["garch_midas"]["theta1_grid"] = [1.0]
    cfg["garch_midas"]["theta2_grid"] = [5.0]
    cfg["garch_midas"]["mean_lags_grid"] = [0]
    cfg["garch_midas"]["dist_grid"] = ["normal"]
    st = dict(cfg["stacking"])
    st["val_frac"] = 0.2
    st["lstm_svr"] = cfg["lstm_svr"]
    st["xgboost"] = cfg["xgboost"]
    st["garch_midas"] = cfg["garch_midas"]
    model = OilStackingForecaster(st, seed=42, horizon=1, price_col=PRICE_COL, returns=panel["logret_1"])
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-20:])
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_oil_tier2.yaml").read_text(encoding="utf-8"))
    panel = demo_oil_panel(200)
    for name in MODELS:
        if name == "stacking":
            model_factory(name, cfg, horizon=1, price_col=PRICE_COL, returns=panel["logret_1"])()
        else:
            model_factory(name, cfg, horizon=1, price_col=PRICE_COL)()


def test_run_sgh_oil_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_oil_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["emd"]["ridge_alpha_grid"] = [1.0]
    cfg["lstm_svr"]["window_grid"] = [21]
    cfg["lstm_svr"]["hidden_grid"] = [16]
    cfg["lstm_svr"]["lr_grid"] = [0.001]
    cfg["lstm_svr"]["max_epochs"] = 3
    cfg["lstm_svr"]["patience"] = 2
    cfg["lstm_svr"]["svr_c_grid"] = [1.0]
    cfg["lstm_svr"]["svr_gamma_grid"] = ["scale"]
    cfg["garch_midas"]["k_grid"] = [3]
    cfg["garch_midas"]["theta1_grid"] = [1.0]
    cfg["garch_midas"]["theta2_grid"] = [5.0]
    cfg["garch_midas"]["mean_lags_grid"] = [0]
    cfg["garch_midas"]["dist_grid"] = ["normal"]
    cfg["xgboost"]["n_estimators"] = 30
    cfg["xgboost"]["max_depth_grid"] = [3]
    cfg["xgboost"]["learning_rate_grid"] = [0.05]
    cfg["sparse_ae"]["max_epochs"] = 3
    cfg["sparse_ae"]["window"] = 21
    panel = demo_oil_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= set(MODELS)
    for name in MODELS:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
    assert "tier1_comparison" in out
