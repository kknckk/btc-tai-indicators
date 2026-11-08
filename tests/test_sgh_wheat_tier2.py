"""TRD-53 tier-II wheat models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import yaml

from experiments.config import ROOT
from experiments.run_sgh_wheat_tier2 import MODELS, run_fixed
from experiments.sgh.tier2_wheat import SarimaLSTMForecaster, model_factory
from experiments.sgh.wheat_panel import PRICE_COL, demo_wheat_panel, wheat_tier2_features
from evaluation.walk_forward import check_causal


def test_wheat_tier2_features_causal():
    panel = demo_wheat_panel(400)
    bad = check_causal(wheat_tier2_features, panel, n_probes=3, seed=53)
    assert bad == []


def test_sarima_lstm_fits_demo():
    panel = demo_wheat_panel(320)
    X = wheat_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_wheat_tier2.yaml").read_text(encoding="utf-8"))
    sc = dict(cfg["sarima_lstm"])
    sc["lstm"]["max_epochs"] = 3
    sc["lstm"]["window_grid"] = [21]
    sc["lstm"]["hidden_grid"] = [16]
    sc["lstm"]["lr_grid"] = [0.001]
    sc["seasonal_grid"] = [[0, 0, 0, 0]]
    model = SarimaLSTMForecaster(sc, seed=42, horizon=7, price_col=PRICE_COL)
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_wheat_tier2.yaml").read_text(encoding="utf-8"))
    for name in MODELS:
        model_factory(name, cfg, horizon=1, price_col=PRICE_COL)()


def test_run_sgh_wheat_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_wheat_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["sarima_lstm"]["lstm"]["max_epochs"] = 3
    cfg["sarima_lstm"]["lstm"]["window_grid"] = [21]
    cfg["sarima_lstm"]["seasonal_grid"] = [[0, 0, 0, 0]]
    cfg["fedformer"]["max_epochs"] = 3
    cfg["fedformer"]["window"] = 21
    cfg["gwrfr"]["n_estimators_grid"] = [50]
    cfg["gwrfr"]["max_depth_grid"] = [4]
    panel = demo_wheat_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= set(MODELS)
    for name in MODELS:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
    assert "tier1_comparison" in out
