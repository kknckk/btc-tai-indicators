"""TRD-38 tier-I mortgage credit models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_credit_mortgage_tier1 import run_fixed
from experiments.sgh.credit_mortgage_panel import credit_mortgage_features, demo_credit_mortgage_panel
from experiments.sgh.tier1_credit_mortgage import ARDLForecaster, VECMForecaster
from evaluation.walk_forward import check_causal


def test_mortgage_features_causal():
    panel = demo_credit_mortgage_panel(160)
    bad = check_causal(credit_mortgage_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_ardl_fits_on_demo_panel():
    panel = demo_credit_mortgage_panel(120)
    X = credit_mortgage_features(panel).dropna()
    y = panel["credit_mortgage"].shift(-1).reindex(X.index)
    m = y.notna()
    model = ARDLForecaster()
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_vecm_fits_on_demo_panel():
    panel = demo_credit_mortgage_panel(140)
    X = credit_mortgage_features(panel).dropna()
    y = panel["credit_mortgage"].shift(-3).reindex(X.index)
    m = y.notna()
    model = VECMForecaster(horizon=3, rank_grid=(0, 1), k_ar_grid=(1,))
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-8:])
    assert np.isfinite(pred).all()


def test_run_sgh_credit_mortgage_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_credit_mortgage_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 3]
    panel = demo_credit_mortgage_panel(180)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= {
        "ardl",
        "vecm",
        "dynamic_regression",
        "logit_scorecard",
    }
    for name in out["horizons"]["1"]:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
