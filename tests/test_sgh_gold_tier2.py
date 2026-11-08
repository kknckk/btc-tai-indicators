"""TRD-57 tier-II gold models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import yaml

from experiments.config import ROOT
from experiments.run_sgh_gold_tier2 import MODELS, run_fixed
from experiments.sgh.gold_panel import PRICE_COL, demo_gold_panel, gold_tier2_features
from experiments.sgh.tier2_gold import (
    TimeGPTForecaster,
    TVPVARForecaster,
    VARLSTMForecaster,
    model_factory,
)
from evaluation.walk_forward import check_causal


def test_gold_tier2_features_causal():
    panel = demo_gold_panel(400)
    bad = check_causal(gold_tier2_features, panel, n_probes=3, seed=57)
    assert bad == []


def test_timegpt_fits_demo():
    panel = demo_gold_panel(320)
    X = gold_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = TimeGPTForecaster(
        {"val_frac": 0.2, "min_hist": 60, "ets_trend_grid": ("add",), "ets_seasonal_grid": (None,)},
        horizon=7,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-20:])
    assert np.isfinite(pred).all()


def test_var_lstm_fits_demo():
    panel = demo_gold_panel(350)
    X = gold_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_gold_tier2.yaml").read_text(encoding="utf-8"))
    vc = dict(cfg["var_lstm"])
    vc["val_frac"] = 0.2
    vc["p_grid"] = [1, 2]
    vc["lstm"]["window_grid"] = [21]
    vc["lstm"]["hidden_grid"] = [16]
    vc["lstm"]["lr_grid"] = [0.001]
    vc["lstm"]["max_epochs"] = 3
    vc["lstm"]["patience"] = 2
    model = VARLSTMForecaster(vc, seed=42, horizon=1, price_col=PRICE_COL)
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-20:])
    assert np.isfinite(pred).all()


def test_tvp_var_fits_demo():
    panel = demo_gold_panel(300)
    X = gold_tier2_features(panel).dropna()
    y = panel[PRICE_COL].shift(-7).reindex(X.index)
    m = y.notna()
    model = TVPVARForecaster(
        horizon=7,
        val_frac=0.2,
        p_grid=(1, 2),
        halflife_grid=(21.0, 63.0),
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_model_factory_names():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_gold_tier2.yaml").read_text(encoding="utf-8"))
    for name in MODELS:
        model_factory(name, cfg, horizon=1, price_col=PRICE_COL)()


def test_run_sgh_gold_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_gold_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["timegpt"]["min_hist"] = 60
    cfg["timegpt"]["ets_trend_grid"] = ["add"]
    cfg["timegpt"]["ets_seasonal_grid"] = [None]
    cfg["var_lstm"]["p_grid"] = [1, 2]
    cfg["var_lstm"]["lstm"]["window_grid"] = [21]
    cfg["var_lstm"]["lstm"]["hidden_grid"] = [16]
    cfg["var_lstm"]["lstm"]["lr_grid"] = [0.001]
    cfg["var_lstm"]["lstm"]["max_epochs"] = 3
    cfg["var_lstm"]["lstm"]["patience"] = 2
    cfg["tvp_var"]["p_grid"] = [1, 2]
    cfg["tvp_var"]["halflife_grid"] = [21, 63]
    panel = demo_gold_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in MODELS:
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
    assert "tier1_comparison" in out
