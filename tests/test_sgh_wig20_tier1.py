"""TRD-50 tier-I WIG20 models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.config import ROOT
from experiments.run_sgh_wig20_tier1 import run_fixed
from experiments.sgh.tier1_wig20 import ArchLevelForecaster, EventStudyLevelForecaster
from experiments.sgh.wig20_panel import PRICE_COL, demo_wig20_panel, wig20_features
from evaluation.walk_forward import check_causal


def test_wig20_features_causal():
    panel = demo_wig20_panel(400)
    bad = check_causal(wig20_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_garch_fits_demo():
    panel = demo_wig20_panel(320)
    X = wig20_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = ArchLevelForecaster(
        "garch",
        panel["logret_1"],
        PRICE_COL,
        horizon=7,
        val_frac=0.2,
        mean_lags_grid=(0,),
        dist_grid=("normal",),
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-20:])
    assert np.isfinite(pred).all()


def test_event_study_fits_demo():
    panel = demo_wig20_panel(280)
    X = wig20_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = EventStudyLevelForecaster(
        panel["logret_1"],
        PRICE_COL,
        horizon=1,
        quantile_grid=(0.85, 0.90),
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_run_sgh_wig20_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_wig20_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["garch"]["mean_lags_grid"] = [0]
    cfg["garch"]["dist_grid"] = ["normal"]
    cfg["egarch"]["mean_lags_grid"] = [0]
    cfg["egarch"]["dist_grid"] = ["normal"]
    cfg["gjr_garch"]["mean_lags_grid"] = [0]
    cfg["gjr_garch"]["dist_grid"] = ["normal"]
    cfg["event_study"]["quantile_grid"] = [0.85]
    panel = demo_wig20_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in ("garch", "egarch", "gjr_garch", "event_study"):
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
