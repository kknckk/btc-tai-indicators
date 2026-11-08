"""TRD-36 tier-I consumer credit models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_credit_consumer_tier1 import run_fixed
from experiments.sgh.credit_consumer_panel import credit_consumer_features, demo_credit_consumer_panel
from experiments.sgh.tier1_credit_consumer import ARDLForecaster
from evaluation.walk_forward import check_causal


def test_credit_features_causal():
    panel = demo_credit_consumer_panel(160)
    bad = check_causal(credit_consumer_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_ardl_fits_on_demo_panel():
    panel = demo_credit_consumer_panel(120)
    X = credit_consumer_features(panel).dropna()
    y = panel["credit_consumer"].shift(-1).reindex(X.index)
    m = y.notna()
    model = ARDLForecaster()
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_run_sgh_credit_consumer_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_credit_consumer_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 3]
    panel = demo_credit_consumer_panel(180)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= {
        "ardl",
        "ecm",
        "dynamic_regression",
        "logit_scorecard",
    }
    for name in out["horizons"]["1"]:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
