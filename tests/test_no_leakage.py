"""Behavioural leakage tests: they inject future information and require the harness to notice."""
import numpy as np
import pandas as pd
import pytest

from evaluation.walk_forward import _fold_starts, check_causal, run_walk_forward
from experiments.data import build_features, forward_log_return, load_panel


@pytest.fixture(scope="module")
def panel():
    return load_panel(start="2013-01-01", end="2020-12-31")


def clean_features(p: pd.DataFrame) -> pd.DataFrame:
    r = p["logret_1"]
    return pd.DataFrame({"r1": r, "r5": r.rolling(5).sum(), "rv": p["rv_30"],
                         "tx": p["slog1p_TxCnt"].diff(7)}, index=p.index)


def leaky_shift(p):  # peeks next-day return
    F = clean_features(p)
    F["leak"] = p["logret_1"].shift(-1)
    return F


def leaky_centered(p):  # centred rolling window sees t+1..t+3
    F = clean_features(p)
    F["leak"] = p["logret_1"].rolling(7, center=True).mean()
    return F


def leaky_fullsample_scaling(p):  # statistic using the whole sample
    F = clean_features(p)
    F["leak"] = (p["logret_1"] - p["logret_1"].mean()) / p["logret_1"].std()
    return F


class RidgeLike:
    """Tiny deterministic model; scaler is fit inside fit() (train only)."""

    def fit(self, X, y):
        Xf = X.fillna(0.0)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        Z = ((Xf - self.mu) / self.sd).to_numpy()
        self.w = np.linalg.solve(Z.T @ Z + 10.0 * np.eye(Z.shape[1]), Z.T @ (y.to_numpy() - y.mean()))
        self.b = float(y.mean())
        return self

    def predict(self, X):
        return ((X.fillna(0.0) - self.mu) / self.sd).to_numpy() @ self.w + self.b


def test_clean_features_are_causal(panel):
    assert check_causal(clean_features, panel) == []
    assert check_causal(build_features, panel) == []


@pytest.mark.parametrize("fn", [leaky_shift, leaky_centered])
def test_causality_check_catches_injected_future(panel, fn):
    assert "leak" in check_causal(fn, panel)
    with pytest.raises(ValueError, match="not causal"):
        run_walk_forward(panel, 1, fn, RidgeLike, min_train=500, step=60, max_folds=2)


def test_causality_check_catches_fullsample_scaling(panel):
    assert "leak" in check_causal(leaky_fullsample_scaling, panel)


def test_leaky_feature_would_inflate_oos_score(panel):
    """Sanity: if the causal guard is bypassed the leak gives absurd OOS R^2 — proves the guard matters."""
    kw = dict(h=1, model_factory=RidgeLike, min_train=500, step=60, causal_check=False)
    clean = run_walk_forward(panel, feature_fn=clean_features, **kw)["metrics"]["point"]["r2_oos_vs_rw"]
    leak = run_walk_forward(panel, feature_fn=leaky_shift, **kw)["metrics"]["point"]["r2_oos_vs_rw"]
    assert clean < 0.1 and leak > 0.5


def _preds(p, fn, h=1):
    return run_walk_forward(p, h, fn, RidgeLike, min_train=500, step=30, causal_check=False)["frame"]["y_pred"]


@pytest.mark.parametrize("h", [1, 7])
def _shuffle_future(panel: pd.DataFrame, cut: pd.Timestamp, seed: int = 0, all_columns: bool = True) -> pd.DataFrame:
    """Copy of ``panel`` where every value after ``cut`` is randomly permuted in time and the
    price-derived columns are rebuilt from the permuted price (rows <= cut stay bit-identical)."""
    rng = np.random.default_rng(seed)
    p2 = panel.copy()
    fut = p2.index > cut
    cols = [c for c in p2.columns if all_columns or c == "PriceUSD"]
    for c in cols:
        v = p2.loc[fut, c].to_numpy().copy()
        rng.shuffle(v)
        p2.loc[fut, c] = v
    p2["log_price"] = np.log(p2["PriceUSD"])
    new_ret = p2["log_price"].diff()
    p2["logret_1"] = panel["logret_1"].where(~fut, new_ret)
    p2["rv_30"] = panel["rv_30"].where(~fut, new_ret.rolling(30).std() * np.sqrt(365))
    return p2


@pytest.mark.parametrize("h", [1, 7])
def test_shuffle_future_does_not_change_past_predictions(panel, h):
    """Permute everything after a cut date: forecasts made at or before the cut (whose training
    labels end before it) must be unchanged for a leak-free pipeline."""
    cut = pd.Timestamp("2017-06-30")
    p2 = _shuffle_future(panel, cut)
    a, b = _preds(panel, clean_features, h), _preds(p2, clean_features, h)
    safe = a.index[a.index <= cut - pd.Timedelta(days=h)]
    assert len(safe) > 200
    np.testing.assert_allclose(a.loc[safe].to_numpy(), b.loc[safe].to_numpy(), atol=1e-12)
    later = a.index[a.index > cut + pd.Timedelta(days=60)]
    assert np.abs(a.loc[later].to_numpy() - b.loc[later].to_numpy()).max() > 1e-4  # test has power


def test_shuffle_future_detects_leaky_pipeline(panel):
    cut = pd.Timestamp("2017-06-30")
    p2 = _shuffle_future(panel, cut, all_columns=False)
    a, b = _preds(panel, leaky_shift), _preds(p2, leaky_shift)
    edge = a.index[(a.index > cut - pd.Timedelta(days=3)) & (a.index <= cut)]  # origins whose t+1 lies after the cut
    assert len(edge) == 3
    assert np.abs(a.loc[edge].to_numpy() - b.loc[edge].to_numpy()).max() > 1e-6
    # ...while the causal pipeline is unaffected on exactly the same origins
    ca, cb = _preds(panel, clean_features), _preds(p2, clean_features)
    np.testing.assert_allclose(ca.loc[edge].to_numpy(), cb.loc[edge].to_numpy(), atol=1e-12)


@pytest.mark.parametrize("h,embargo", [(1, 0), (7, 0), (30, 0), (7, 5)])
def test_purge_and_embargo_geometry(panel, h, embargo):
    n = len(panel)
    res = run_walk_forward(panel, h, clean_features, RidgeLike, min_train=500, embargo=embargo, step=45, causal_check=False)
    for f in res["folds"]:
        te0, tr1 = pd.Timestamp(f["test_start"]), pd.Timestamp(f["train_end"])
        # last train label window ends at train_end + h  < test_start - embargo
        assert tr1 + pd.Timedelta(days=h + embargo) < te0
    ends = [pd.Timestamp(f["test_end"]) for f in res["folds"]]
    starts = [pd.Timestamp(f["test_start"]) for f in res["folds"]]
    assert all(s2 == e1 + pd.Timedelta(days=1) for e1, s2 in zip(ends[:-1], starts[1:]))  # contiguous, no overlap
    assert res["folds"][0]["n_train"] >= 500
    assert all(res["folds"][i]["n_train"] < res["folds"][i + 1]["n_train"] for i in range(len(res["folds"]) - 1))
    assert len(_fold_starts(n, h, 500, embargo, 45)) == len(res["folds"])


def test_target_never_equals_feature(panel):
    X = clean_features(panel)
    y = forward_log_return(panel["log_price"], 1)
    for c in X.columns:
        m = X[c].notna() & y.notna()
        assert not np.allclose(X.loc[m, c], y[m])
