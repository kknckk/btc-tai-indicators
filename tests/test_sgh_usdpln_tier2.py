"""TRD-49 tier-II USD/PLN models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import yaml

from experiments.config import ROOT
from experiments.run_sgh_usdpln_tier2 import MODELS, run_fixed
from experiments.sgh.tier2_eurpln import VMDCNNBiLSTMForecaster, model_factory
from experiments.sgh.usdpln_panel import PRICE_COL, demo_usdpln_panel, usdpln_tier2_features
from evaluation.walk_forward import check_causal


def test_usdpln_tier2_features_causal():
    panel = demo_usdpln_panel(400)
    bad = check_causal(usdpln_tier2_features, panel, n_probes=3, seed=4)
    assert bad == []


def test_vmd_cnn_bilstm_fits_demo():
    panel = demo_usdpln_panel(280)
    X = usdpln_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = VMDCNNBiLSTMForecaster(
        {"window": 21, "hidden": 16, "max_epochs": 3, "patience": 2},
        seed=42,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_usdpln_tier2.yaml").read_text(encoding="utf-8"))
    for name in MODELS:
        model_factory(name, cfg, horizon=1, price_col=PRICE_COL)()


def test_run_sgh_usdpln_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_usdpln_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["vmd_cnn_bilstm"]["max_epochs"] = 3
    cfg["vmd_cnn_bilstm"]["window"] = 21
    cfg["anomaly_transformer"]["max_epochs"] = 3
    cfg["anomaly_transformer"]["window"] = 21
    cfg["tft"]["max_epochs"] = 3
    cfg["tft"]["window"] = 21
    panel = demo_usdpln_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= set(MODELS)
    for name in MODELS:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
    assert "tier1_comparison" in out
