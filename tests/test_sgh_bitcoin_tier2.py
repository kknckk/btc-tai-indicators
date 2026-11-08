"""TRD-59 tier-II bitcoin models — smoke on demo panel."""
from __future__ import annotations

import numpy as np
import yaml

from experiments.config import ROOT
from experiments.run_sgh_bitcoin_tier2 import MODELS, run_fixed
from experiments.sgh.bitcoin_panel import PRICE_COL, bitcoin_features, demo_bitcoin_panel
from experiments.sgh.tier1_wheat import LSTMLevelForecaster
from experiments.sgh.tier2_bitcoin import HellFormerForecaster, SVMLevelForecaster, model_factory
from experiments.sgh.tier2_eurpln import TFTForecaster
from evaluation.walk_forward import check_causal


def test_bitcoin_features_causal():
    panel = demo_bitcoin_panel(400)
    bad = check_causal(bitcoin_features, panel, n_probes=3, seed=59)
    assert bad == []


def test_lstm_fits_demo():
    panel = demo_bitcoin_panel(300)
    X = bitcoin_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = LSTMLevelForecaster(
        {
            "val_frac": 0.2,
            "window_grid": (30,),
            "hidden_grid": (16,),
            "lr_grid": (1e-3,),
            "max_epochs": 3,
            "patience": 2,
        },
        seed=42,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_tft_fits_demo():
    panel = demo_bitcoin_panel(320)
    X = bitcoin_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = TFTForecaster(
        {"window": 30, "hidden": 16, "nhead": 2, "max_epochs": 3, "patience": 2, "lr": 1e-3},
        seed=42,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_svm_fits_demo():
    panel = demo_bitcoin_panel(280)
    X = bitcoin_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = SVMLevelForecaster({"val_frac": 0.2, "c_grid": (1.0, 10.0)}, seed=42, price_col=PRICE_COL)
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_hellformer_fits_demo():
    panel = demo_bitcoin_panel(350)
    X = bitcoin_features(panel).dropna()
    y = panel[PRICE_COL].shift(-1).reindex(X.index)
    m = y.notna()
    model = HellFormerForecaster(
        {
            "val_frac": 0.2,
            "window_grid": (30,),
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "kernel_sizes": [3, 7],
            "max_epochs": 3,
            "patience": 2,
        },
        seed=42,
        price_col=PRICE_COL,
    )
    model.fit(X[m], y[m])
    pred = model.predict(X[m].iloc[-15:])
    assert np.isfinite(pred).all()


def test_model_factory_keys():
    cfg = yaml.safe_load((ROOT / "experiments/configs/sgh_bitcoin_tier2.yaml").read_text(encoding="utf-8"))
    for name in MODELS:
        factory = model_factory(name, cfg, horizon=1, price_col=PRICE_COL)
        assert callable(factory)
        assert factory() is not None


def test_run_sgh_bitcoin_tier2_demo_smoke():
    cfg_path = ROOT / "experiments/configs/sgh_bitcoin_tier2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["horizons"] = [1, 7]
    cfg["walk_forward"]["min_train"] = 120
    cfg["lstm"]["window_grid"] = [30]
    cfg["lstm"]["hidden_grid"] = [16]
    cfg["lstm"]["lr_grid"] = [0.001]
    cfg["lstm"]["max_epochs"] = 3
    cfg["lstm"]["patience"] = 2
    cfg["tft"]["window"] = 30
    cfg["tft"]["hidden"] = 16
    cfg["tft"]["max_epochs"] = 3
    cfg["tft"]["patience"] = 2
    cfg["svm"]["c_grid"] = [1.0]
    cfg["hellformer"]["window_grid"] = [30]
    cfg["hellformer"]["d_model"] = 16
    cfg["hellformer"]["max_epochs"] = 3
    cfg["hellformer"]["patience"] = 2
    panel = demo_bitcoin_panel(550)
    out = run_fixed(cfg, panel, max_folds=2, demo=True)
    block = out["horizons"]["1"]
    for name in MODELS:
        assert name in block
        assert "dm_mse_vs_naive" in block[name]
        assert "skill_vs_rw" in block[name]
    assert "tier1_comparison" in out
