"""TRD-35 tier-II bond yield models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_bonds_tier2 import MODELS, run_fixed
from experiments.sgh.bonds_panel import BOND_COLS, bonds_tier2_features, demo_bonds_panel
from experiments.sgh.tier2_bonds import VARFactorsForecaster, model_factory
from evaluation.walk_forward import check_causal


def test_bonds_tier2_features_causal():
    panel = demo_bonds_panel(400)
    bad = check_causal(bonds_tier2_features, panel, n_probes=3, seed=3)
    assert bad == []


def test_var_factors_fits_demo():
    panel = demo_bonds_panel(300)
    X = bonds_tier2_features(panel).dropna()
    y = panel["bond_pl_10y"].shift(-7).reindex(X.index)
    m = y.notna()
    model = VARFactorsForecaster(horizon=7, target_col="bond_pl_10y", p_grid=(1, 2))
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_bonds_pl_tier2.yaml").read_text(encoding="utf-8"))
    for name in MODELS:
        model_factory(name, cfg, horizon=1, target_col="bond_pl_10y")()


def test_run_sgh_bonds_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_bonds_pl_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["tenors"] = [BOND_COLS[0], BOND_COLS[-1]]
    cfg["walk_forward"]["min_train"] = 120
    cfg["lstm"]["max_epochs"] = 5
    cfg["pca_lstm"]["max_epochs"] = 5
    panel = demo_bonds_panel(500)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    for tenor in cfg["tenors"]:
        block = out["tenors"][tenor]["horizons"]["1"]
        assert set(block) >= set(MODELS)
        for name in MODELS:
            assert "dm_mse_vs_naive" in block[name]
            assert "skill_vs_rw" in block[name]
        assert "tier1_comparison" in out
        assert tenor in out["tier1_comparison"]
