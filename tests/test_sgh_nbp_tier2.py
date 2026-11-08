"""TRD-33 tier-II NBP rate models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_nbp_tier2 import MODELS, run_fixed
from experiments.sgh.nbp_panel import demo_nbp_panel, nbp_tier2_features
from experiments.sgh.tier2_nbp import VARForecaster, model_factory
from evaluation.walk_forward import check_causal


def test_nbp_tier2_features_causal():
    panel = demo_nbp_panel(120)
    bad = check_causal(nbp_tier2_features, panel, n_probes=3, seed=2)
    assert bad == []


def test_var_fits_on_demo_panel():
    panel = demo_nbp_panel(100)
    X = nbp_tier2_features(panel).dropna()
    y = panel["nbp_rate"].shift(-1).reindex(X.index)
    m = y.notna()
    model = VARForecaster(horizon=1, p_grid=(1, 2))
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_nbp_rate_tier2.yaml").read_text(encoding="utf-8"))
    for name in MODELS:
        model_factory(name, cfg, horizon=1)()


def test_run_sgh_nbp_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_nbp_rate_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 3]
    panel = demo_nbp_panel(160)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= set(MODELS)
    for name in MODELS:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
    assert "tier1_comparison" in out
