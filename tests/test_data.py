import numpy as np
import pandas as pd
import pytest

from experiments.data import (FORWARD_LOOKING_COLUMNS, build_features, coverage_report,
                              detect_forward_looking_columns, forward_log_return, load_panel,
                              make_supervised, signed_log1p)


@pytest.fixture(scope="module")
def panel():
    return load_panel()


def test_index_monotone_daily_unique(panel):
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert (panel.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_expected_columns_and_coverage(panel):
    for c in ("PriceUSD", "logret_1", "rv_7", "rv_30", "log_price", "slog1p_TxCnt"):
        assert c in panel.columns
    cov = coverage_report(panel)
    assert len(cov) == panel.shape[1]  # nothing silently dropped
    assert "VelCur1yr_Naive" in cov.index and cov.loc["VelCur1yr_Naive", "n_obs"] == 0


def test_derived_columns_are_causal(panel):
    r = np.log(panel["PriceUSD"]).diff()
    pd.testing.assert_series_equal(panel["logret_1"], r, check_names=False)
    assert panel["rv_30"].first_valid_index() > panel["logret_1"].first_valid_index()


def test_signed_log1p_monotone():
    x = np.linspace(-1e6, 1e6, 1001)
    assert (np.diff(signed_log1p(x)) > 0).all()


@pytest.mark.parametrize("h", [1, 7, 30])
def test_target_definition_and_not_in_features(panel, h):
    X, y = make_supervised(panel, h=h)
    t = X.index[500]
    expect = np.log(panel["PriceUSD"].loc[t + pd.Timedelta(days=h)]) - np.log(panel["PriceUSD"].loc[t])
    assert y.loc[t] == pytest.approx(expect)
    assert not any(c.startswith("logret_fwd") for c in X.columns)
    assert not (set(X.columns) & FORWARD_LOOKING_COLUMNS)
    assert y.name not in X.columns
    assert X.index.is_monotonic_increasing and X.index.equals(y.index)
    assert y.notna().all()


def test_forward_target_uses_only_future_prices(panel):
    lp = panel["log_price"]
    y = forward_log_return(lp, 7)
    assert pd.isna(y.dropna().index.max() + pd.Timedelta(days=0)) is False
    assert y.loc[:lp.last_valid_index()].dropna().index.max() <= lp.last_valid_index() - pd.Timedelta(days=7)


def test_forward_looking_column_detected_and_rejected(panel):
    flagged = detect_forward_looking_columns(panel)
    assert "dir_t1" in flagged  # shipped label = 1[next-day return > 0]
    assert set(flagged) == {"dir_t1"}, f"unexpected forward-looking columns: {flagged}"
    with pytest.raises(ValueError):
        make_supervised(panel, h=1, features=["dir_t1"])


def test_default_features_exclude_price_derived(panel):
    F = build_features(panel)
    for c in ("dir_t1", "r_t", "rv_t", "rv_7d", "rv_30d", "PriceUSD", "log_price"):
        assert c not in F.columns
