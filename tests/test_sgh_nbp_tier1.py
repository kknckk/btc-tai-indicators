"""TRD-32 tier-I NBP rate models — smoke on demo panel, GUS CSV parser test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import macro_pl as mp
from experiments.config import ROOT
from experiments.run_sgh_nbp_tier1 import run_fixed
from experiments.sgh.nbp_panel import demo_nbp_panel, nbp_features
from experiments.sgh.tier1_nbp import TaylorRuleForecaster
from evaluation.targets import TargetSpec
from evaluation.walk_forward import check_causal


GUS_CSV = """time,gus_cpi_yoy,gus_pmi
2024-01-31,4.5,48.0
2024-02-29,4.2,49.1
2024-03-31,3.9,50.2
"""


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(mp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mp, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(mp, "META_PATH", tmp_path / "meta.json")
    return tmp_path


def test_parse_gus_macro_csv(cache_dir):
    mp.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = mp.RAW_DIR / "gus_macro.csv"
    path.write_text(GUS_CSV, encoding="utf-8")
    out = mp.fetch_gus_macro(refresh=True)
    assert len(out["gus_cpi_yoy"]) == 3
    assert out["gus_pmi"].iloc[-1] == pytest.approx(50.2)


def test_nbp_features_causal():
    panel = demo_nbp_panel(120)
    bad = check_causal(nbp_features, panel, n_probes=3, seed=1)
    assert bad == []


def test_taylor_fits_on_demo_panel():
    panel = demo_nbp_panel(100)
    X = nbp_features(panel).dropna()
    y = panel["nbp_rate"].shift(-1).reindex(X.index)
    m = y.notna()
    model = TaylorRuleForecaster()
    model.fit(X[m], y[m])
    pred = model.predict(X[m])
    assert len(pred) == int(m.sum())
    assert np.isfinite(pred).all()


def test_run_sgh_nbp_tier1_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_nbp_rate_tier1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 3]
    panel = demo_nbp_panel(160)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    assert set(out["horizons"]["1"]) >= {
        "taylor_rule",
        "forward_linear",
        "ordered_logit",
        "ordered_probit",
    }
    for name in out["horizons"]["1"]:
        assert "dm_mse_vs_naive" in out["horizons"]["1"][name]
        assert "skill_vs_rw" in out["horizons"]["1"][name]
