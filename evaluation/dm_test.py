"""Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Two-sided DM test of equal expected loss. d_t = loss_a - loss_b.

    Negative statistic => model A has lower loss. Long-run variance uses h-1 autocovariances
    (h-step forecasts are MA(h-1)); the HLN factor sqrt((n+1-2h+h(h-1)/n)/n) is applied and the
    statistic is compared with Student-t(n-1). Returns (stat, p); (nan, nan) if degenerate.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0.0):
        return float("nan"), float("nan")
    dc = d - d.mean()
    var = float(dc @ dc / n)
    for k in range(1, h):
        var += 2.0 * float(dc[k:] @ dc[:-k] / n)
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def dm_matrix(losses: dict[str, np.ndarray], h: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise DM statistics (row vs column; negative = row better) and p-values."""
    names = list(losses)
    stat = pd.DataFrame(np.nan, index=names, columns=names)
    pval = stat.copy()
    for a in names:
        for b in names:
            if a != b:
                stat.loc[a, b], pval.loc[a, b] = dm_test(losses[a], losses[b], h)
    return stat, pval


def average_rank(mean_losses: list[dict[str, float]]) -> pd.DataFrame:
    """Average rank (1 = lowest loss) of each model across a list of loss tables
    (e.g. one table per horizon / loss function). Models absent from a table are skipped there."""
    ranks = pd.DataFrame([pd.Series(t).rank(method="average") for t in mean_losses])
    return pd.DataFrame({"avg_rank": ranks.mean(), "n_tables": ranks.notna().sum()}).sort_values("avg_rank")
