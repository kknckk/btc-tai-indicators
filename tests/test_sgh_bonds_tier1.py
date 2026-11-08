"""TRD-34 tier-I bond yield models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_bonds_tier1 import run_fixed
from experiments.sgh.bonds_panel import BOND_COLS, demo_bonds_panel, bonds_features
from experiments.sgh.tier1_bonds import NelsonSiegelForecaster, _ns_loadings
from evaluation.walk_forward import check_causal


def test_ns_loadings_shape():
    L = _ns_loadings(np.array([2.0, 10.0]), 0.07)
    assert L.shape == (2, 3)
    assert np.allclose(L[:, 0], 1.0)


def test_bonds_features_causal():
    panel = demo_bonds_panel(400)
    bad = check_causal(bonds_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_nelson_siegel_fits_demo():
    panel = demo_bonds_panel(300)
    X = bonds_features(panel).dropna()
    y = panel["bond_pl_10y"].shift(-7).reindex(X.index)
    m = y.notna()
    model = NelsonSiegelForecaster(horizon=7, target_col="bond_pl_10y")
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_run_sgh_bonds_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_bonds_pl_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["tenors"] = [BOND_COLS[0], BOND_COLS[-1]]
    cfg["walk_forward"]["min_train"] = 120
    panel = demo_bonds_panel(500)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    for tenor in cfg["tenors"]:
        assert "horizons" in out["tenors"][tenor]
        block = out["tenors"][tenor]["horizons"]["1"]
        for name in ("nelson_siegel", "svensson", "diebold_li"):
            assert name in block
            assert "dm_mse_vs_naive" in block[name]
            assert "skill_vs_rw" in block[name]
