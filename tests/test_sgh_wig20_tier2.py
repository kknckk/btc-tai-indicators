"""TRD-51 tier-II WIG20 models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import yaml

from experiments.config import ROOT
from experiments.run_sgh_wig20_tier2 import MODELS, run_fixed
from experiments.sgh.tier2_wig20 import FeatureRidgeForecaster, StochasticVolLevelForecaster, model_factory
from experiments.sgh.wig20_panel import PRICE_COL, demo_wig20_panel, wig20_tier2_features
from evaluation.walk_forward import check_causal


def test_wig20_tier2_features_causal():
    panel = demo_wig20_panel(400)
    bad = check_causal(wig20_tier2_features, panel, n_probes=3, seed=51)
    assert bad == []


def test_har_rv_fits_demo():
    panel = demo_wig20_panel(320)
    X = wig20_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = FeatureRidgeForecaster(
        feature_cols=("rv_d", "rv_w", "rv_m", f"{PRICE_COL}_l1"),
        val_frac=0.2,
        alpha_grid=(0.1, 1.0),
        model_name="har_rv",
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-20:])
    assert np.isfinite(pred).all()


def test_stochastic_vol_fits_demo():
    panel = demo_wig20_panel(300)
    X = wig20_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = StochasticVolLevelForecaster(
        returns=panel["logret_1"],
        price_col=PRICE_COL,
        horizon=1,
        val_frac=0.2,
        phi_grid=(0.9,),
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_wig20_tier2.yaml").read_text(encoding="utf-8"))
    panel = demo_wig20_panel(200)
    for name in MODELS:
        model_factory(name, cfg, horizon=1, price_col=PRICE_COL, returns=panel["logret_1"])()


def test_run_sgh_wig20_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_wig20_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    rf = cfg["random_forest"]
    rf["n_estimators"] = [50]
    rf["max_depth"] = [4]
    gb = cfg["gradient_boosting"]
    gb["max_depth"] = [3]
    gb["learning_rate"] = [0.1]
    panel = demo_wig20_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in MODELS:
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
    assert "tier1_comparison" in out
