"""TRD-42 tier-I household deposits models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_deposits_households_tier1 import run_fixed
from experiments.sgh.deposits_households_panel import (
    demo_deposits_households_panel,
    deposits_households_features,
)
from experiments.sgh.tier1_deposits_households import ARDLForecaster
from evaluation.walk_forward import check_causal


def test_deposits_features_causal():
    panel = demo_deposits_households_panel(160)
    bad = check_causal(deposits_households_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_ardl_fits_on_demo_panel():
    panel = demo_deposits_households_panel(120)
    X = deposits_households_features(panel).dropna()
    y = panel["deposits_households"].shift(-1).reindex(X.index)
    m = y.notna()
    model = ARDLForecaster()
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_run_sgh_deposits_households_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_deposits_households_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 3]
    panel = demo_deposits_households_panel(180)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= {"ardl", "ecm", "dynamic_regression"}
    for name in out["horizons"]["1"]:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
