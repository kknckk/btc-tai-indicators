"""Tests for models/moderntcn.py and models/patchtst.py.

Uses small synthetic data (not the real CSVs) so the suite runs in a few
seconds on CPU; `experiments/run_neural.py --smoke` is the integration
check against real data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import NotFittedError, ModernTCN, ModernTCNConfig, PatchTST, PatchTSTConfig
from models.moderntcn import ModernTCNNet
from models.patchtst import PatchTSTNet


N_ROWS = 220
N_FEATURES = 4


def _make_synthetic_frame(n_rows=N_ROWS, n_features=N_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Target genuinely depends on the features (not pure noise) so a fitted
    # model can learn something non-trivial for the determinism/leakage checks.
    weights = rng.normal(size=n_features)
    y = pd.Series(X.to_numpy() @ weights * 0.1 + rng.normal(scale=0.01, size=n_rows))
    return X, y


FAST_MODERNTCN_CFG = dict(
    window=10,
    d_model=16,
    ffn_dim=32,
    kernel_size=5,
    num_blocks=2,
    max_epochs=5,
    patience=5,
    val_fraction=0.15,
    seed=42,
)

FAST_PATCHTST_CFG = dict(
    window=16,
    patch_len=4,
    stride=4,
    d_model=16,
    nhead=2,
    num_layers=1,
    dim_feedforward=32,
    max_epochs=5,
    patience=5,
    val_fraction=0.15,
    seed=42,
)


@pytest.fixture(scope="module")
def synthetic_data():
    return _make_synthetic_frame()


@pytest.mark.parametrize(
    "model_cls,cfg_cls,cfg_kwargs",
    [
        (ModernTCN, ModernTCNConfig, FAST_MODERNTCN_CFG),
        (PatchTST, PatchTSTConfig, FAST_PATCHTST_CFG),
    ],
)
class TestForecasterContract:
    def test_predict_before_fit_raises(self, model_cls, cfg_cls, cfg_kwargs, synthetic_data):
        X, _ = synthetic_data
        model = model_cls(cfg_cls(**cfg_kwargs))
        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_output_shape(self, model_cls, cfg_cls, cfg_kwargs, synthetic_data):
        X, y = synthetic_data
        model = model_cls(cfg_cls(**cfg_kwargs))
        model.fit(X, y)
        preds = model.predict(X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        window = cfg_kwargs["window"]
        # First `window - 1` rows cannot form a full window -> NaN padded.
        assert np.all(np.isnan(preds[: window - 1]))
        assert np.all(np.isfinite(preds[window - 1 :]))

    def test_determinism_with_fixed_seed(self, model_cls, cfg_cls, cfg_kwargs, synthetic_data):
        X, y = synthetic_data
        model_a = model_cls(cfg_cls(**cfg_kwargs))
        model_a.fit(X, y)
        preds_a = model_a.predict(X)

        model_b = model_cls(cfg_cls(**cfg_kwargs))
        model_b.fit(X, y)
        preds_b = model_b.predict(X)

        np.testing.assert_allclose(preds_a, preds_b, rtol=1e-5, atol=1e-6, equal_nan=True)

    def test_target_shift_changes_predictions(self, model_cls, cfg_cls, cfg_kwargs, synthetic_data):
        """Behavioural no-leakage check: training on a *different* target
        (y shifted by a few steps, i.e. a different horizon) must produce a
        differently-fitted model. If predictions were identical regardless
        of `y`, the model would be ignoring the label (e.g. a leak/bug that
        makes it depend only on X structure, or a frozen/non-learning net).
        """
        X, y = synthetic_data
        y_shifted = y.shift(-5).ffill()

        model_a = model_cls(cfg_cls(**cfg_kwargs))
        model_a.fit(X, y)
        preds_a = model_a.predict(X)

        model_b = model_cls(cfg_cls(**cfg_kwargs))
        model_b.fit(X, y_shifted)
        preds_b = model_b.predict(X)

        valid = ~np.isnan(preds_a) & ~np.isnan(preds_b)
        assert valid.sum() > 0
        assert not np.allclose(preds_a[valid], preds_b[valid], rtol=1e-3, atol=1e-4)


class TestParameterCountSanity:
    def test_moderntcn_default_param_count(self):
        net = ModernTCNNet(n_vars=8, cfg=ModernTCNConfig())
        n_params = sum(p.numel() for p in net.parameters())
        # Spec target: "~230k params at default config".
        assert 150_000 <= n_params <= 320_000

    def test_patchtst_default_param_count(self):
        net = PatchTSTNet(n_vars=8, cfg=PatchTSTConfig())
        n_params = sum(p.numel() for p in net.parameters())
        assert 10_000 <= n_params <= 600_000

    def test_moderntcn_n_params_property_requires_fit(self):
        model = ModernTCN(ModernTCNConfig(**FAST_MODERNTCN_CFG))
        with pytest.raises(NotFittedError):
            _ = model.n_params

    def test_patchtst_channel_independent_flag_changes_param_count(self):
        indep = PatchTSTNet(n_vars=5, cfg=PatchTSTConfig(channel_independent=True))
        joint = PatchTSTNet(n_vars=5, cfg=PatchTSTConfig(channel_independent=False))
        n_indep = sum(p.numel() for p in indep.parameters())
        n_joint = sum(p.numel() for p in joint.parameters())
        assert n_indep != n_joint
