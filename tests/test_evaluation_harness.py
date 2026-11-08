"""Generalized evaluation harness: targets, MCS, conformal, BTC regression."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import metrics as M
from evaluation.backtest import run_backtest
from evaluation.calibration import (
    calibrate_frame_cqr,
    calibrate_frame_point,
    cqr_interval,
    empirical_coverage,
    rolling_split_conformal,
    split_conformal_on_frame,
)
from evaluation.mcs import mcs, rank_by_mean_loss
from evaluation.targets import TargetSpec, forward_target, resolve_price
from evaluation.walk_forward import run_walk_forward, summarize, sign_position
from experiments.data import forward_log_return, load_panel


class RidgeLike:
    def fit(self, X, y):
        Xf = X.fillna(0.0)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        Z = ((Xf - self.mu) / self.sd).to_numpy()
        self.w = np.linalg.solve(Z.T @ Z + 10.0 * np.eye(Z.shape[1]), Z.T @ (y.to_numpy() - y.mean()))
        self.b = float(y.mean())
        return self

    def predict(self, X):
        return ((X.fillna(0.0) - self.mu) / self.sd).to_numpy() @ self.w + self.b


def _features(p):
    r = p["logret_1"]
    return pd.DataFrame({"r1": r, "r5": r.rolling(5).sum()}, index=p.index)


def test_btc_target_matches_forward_log_return():
    panel = load_panel(start="2015-01-01", end="2016-12-31")
    lp = panel["log_price"]
    for h in (1, 7, 30):
        a = forward_log_return(lp, h)
        b = forward_target(lp, TargetSpec.btc_log_return(h))
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_backtest_matches_economic_metrics():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    pos = pd.Series(np.sign(np.sin(np.arange(120))), index=idx)
    nr = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=120), index=idx)
    a = run_backtest(pos, nr, (10, 20))
    b = M.economic_metrics(pos, nr, (10, 20))
    assert a == b


def test_mcs_synthetic_ranking():
    rng = np.random.default_rng(42)
    T = 300
    losses = {
        "best": rng.normal(1.0, 0.05, T),
        "mid": rng.normal(2.0, 0.05, T),
        "worst": rng.normal(3.0, 0.05, T),
    }
    order = list(rank_by_mean_loss(losses).index)
    assert order == ["best", "mid", "worst"]
    out = mcs(losses, alpha=0.10, n_bootstrap=800, seed=7)
    assert "best" in out["confidence_set"]
    assert list(rank_by_mean_loss(losses).index)[0] == "best"


def test_cqr_interval_covers_on_synthetic():
    rng = np.random.default_rng(1)
    n = 200
    y = rng.normal(0, 1.0, n)
    q_lo = y - 0.05
    q_hi = y + 0.05
    cal, test = slice(0, 120), slice(120, None)
    res = cqr_interval(y[cal], q_lo[cal], q_hi[cal], q_lo[test], q_hi[test], alpha=0.10)
    covered = (y[test] >= res.lower) & (y[test] <= res.upper)
    assert empirical_coverage(covered) >= 0.85


def test_rolling_conformal_near_nominal_80():
    rng = np.random.default_rng(2)
    n = 500
    yhat = rng.normal(0, 0.02, n)
    y = yhat + rng.normal(0, 0.03, n)
    _, _, cov = rolling_split_conformal(y, yhat, alpha=0.20, min_cal=50)
    m = cov[50:]
    assert 0.75 <= empirical_coverage(m) <= 0.88


def test_calibrate_frame_point_on_walk_forward_frame(panel_small):
    from experiments.baselines import NaiveRW

    ret = panel_small["logret_1"]
    res = run_walk_forward(
        panel_small, 1, _features, lambda: NaiveRW(ret), min_train=500, step=60, max_folds=4, causal_check=False
    )
    _, summ = calibrate_frame_point(res["frame"], alpha=0.20, min_cal=30)
    assert summ.n_eval > 0
    assert 0.70 <= summ.empirical_coverage <= 0.90


def test_calibrate_frame_cqr_on_quantile_synthetic():
    rng = np.random.default_rng(4)
    n = 120
    y = rng.normal(100.0, 5.0, n)
    price = y + rng.normal(0, 1.0, n)
    q_lo = y - 2.0
    q_hi = y + 2.0
    frame = pd.DataFrame({"y_true": y, "q_lo": q_lo, "q_hi": q_hi, "price": price})
    _, summ = calibrate_frame_cqr(frame, alpha=0.20, min_cal=20)
    assert 0.75 <= summ.empirical_coverage <= 0.92


def test_split_conformal_coverage_on_linear_synthetic():
    rng = np.random.default_rng(0)
    n = 400
    x = np.linspace(0, 4, n)
    y = 2.0 * x + rng.normal(0, 0.5, n)
    yhat = 2.0 * x
    frame = pd.DataFrame({"y_true": y, "y_pred": yhat})
    _, test = split_conformal_on_frame(frame, cal_frac=0.25, alpha=0.10)
    cov = empirical_coverage(test["covered"].to_numpy())
    assert cov >= 0.85


def _buy_hold(pred, vol=None):
    return pd.Series(1.0, index=pred.index)


def test_naive_rw_benchmark_on_btc_panel(panel_small):
    """NaiveRW predicts zero return → OOS R² vs RW is exactly zero (baselines.json property)."""
    from experiments.baselines import NaiveRW

    ret = panel_small["logret_1"]
    naive = run_walk_forward(
        panel_small,
        1,
        _features,
        lambda: NaiveRW(ret),
        min_train=500,
        step=45,
        position_fn=_buy_hold,
        causal_check=False,
    )
    assert naive["metrics"]["point"]["r2_oos_vs_rw"] == 0.0


@pytest.mark.integration
def test_naive_rmse_matches_baselines_json():
    """Full-panel walk-forward with config from base.yaml; matches committed baselines.json."""
    golden_path = Path(__file__).resolve().parents[1] / "results" / "baselines.json"
    if not golden_path.exists():
        pytest.skip("results/baselines.json missing")
    from experiments.baselines import NaiveRW
    from experiments.config import load_config

    cfg = load_config()
    panel = load_panel()
    wf = cfg["walk_forward"]
    ret = panel["logret_1"]
    naive = run_walk_forward(
        panel,
        1,
        _features,
        lambda: NaiveRW(ret),
        min_train=wf["min_train"],
        step=wf["step"],
        embargo=wf["embargo"],
        costs_bps=tuple(cfg["costs_bps"]),
        seed=cfg["seed"],
        position_fn=_buy_hold,
        causal_check=False,
    )
    ref = json.loads(golden_path.read_text())["horizons"]["1"]["naive"]["metrics"]
    got = naive["metrics"]
    assert got["point"]["rmse"] == pytest.approx(ref["point"]["rmse"], rel=1e-4, abs=1e-6)
    assert got["point"]["mape_price_pct"] == pytest.approx(ref["point"]["mape_price_pct"], rel=1e-4)
    assert got["economic"]["20bps"]["sharpe"] == pytest.approx(ref["economic"]["20bps"]["sharpe"], rel=1e-4)


@pytest.fixture(scope="module")
def panel_small():
    return load_panel(start="2013-01-01", end="2020-12-31")


def test_level_target_spec(panel_small):
    spec = TargetSpec(kind="level", price_col="PriceUSD", horizon=7, unit="day")
    price = resolve_price(panel_small, spec)
    y = forward_target(price, spec)
    t = y.dropna().index[100]
    expect = panel_small["PriceUSD"].shift(-7).loc[t]
    assert y.loc[t] == pytest.approx(expect)


def test_no_look_ahead_uses_next_day_return():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    pred = pd.Series(np.linspace(-0.02, 0.02, 40), index=idx)
    same_day = pd.Series(np.random.default_rng(3).normal(0, 0.02, 40), index=idx)
    pos = sign_position(pred)
    causal = run_backtest(pos, same_day.shift(-1), (10,))
    leaky = run_backtest(pos, same_day, (10,))
    assert causal["10bps"]["sharpe"] != leaky["10bps"]["sharpe"]


def test_squared_losses_ranking_on_synthetic():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    y = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50), index=idx)
    frames = {
        "good": pd.DataFrame({"y_true": y, "y_pred": y * 0.1, "next_ret": y.shift(-1)}),
        "bad": pd.DataFrame({"y_true": y, "y_pred": y + 0.05, "next_ret": y.shift(-1)}),
    }
    from evaluation.backtest import squared_return_losses

    losses = squared_return_losses(frames)
    assert np.mean(losses["good"]) < np.mean(losses["bad"])


def test_hit_rate_in_economic_metrics():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    pos = pd.Series(1.0, index=idx)
    nr = pd.Series(0.001, index=idx)
    out = run_backtest(pos, nr, (10,))
    assert 0.0 <= out["10bps"]["hit_rate"] <= 1.0


def test_summarize_uses_backtest(panel_small):
    res = run_walk_forward(
        panel_small, 7, _features, RidgeLike, min_train=500, step=60, max_folds=2, causal_check=False
    )
    s = summarize(res["frame"], sign_position, (10, 20))
    direct = run_backtest(sign_position(res["frame"]["y_pred"]), res["frame"]["next_ret"], (10, 20))
    assert s["economic"] == direct
