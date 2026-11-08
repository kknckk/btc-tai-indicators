"""TRD-44 tier-I TFI assets models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_tfi_assets_tier1 import run_fixed
from experiments.sgh.tfi_assets_panel import demo_tfi_assets_panel, tfi_assets_features
from experiments.sgh.tier1_tfi_assets import FlowPerformanceForecaster
from evaluation.walk_forward import check_causal


def test_tfi_features_causal():
    panel = demo_tfi_assets_panel(160)
    bad = check_causal(tfi_assets_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_flow_performance_fits_on_demo_panel():
    panel = demo_tfi_assets_panel(120)
    X = tfi_assets_features(panel).dropna()
    y = panel["tfi_assets"].shift(-1).reindex(X.index)
    m = y.notna()
    model = FlowPerformanceForecaster()
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_run_sgh_tfi_assets_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_tfi_assets_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 3]
    panel = demo_tfi_assets_panel(180)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= {"flow_performance", "panel_regression", "dynamic_regression"}
    for name in out["horizons"]["1"]:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
