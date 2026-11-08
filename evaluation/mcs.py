"""Model Confidence Set (bootstrap) and pairwise Diebold–Mariano tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.dm_test import dm_matrix, dm_test


def _loss_matrix(losses: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(losses)
    aligned = [np.asarray(losses[k], dtype=float) for k in names]
    n = min(len(x) for x in aligned)
    L = np.column_stack([x[:n] for x in aligned])
    m = np.all(np.isfinite(L), axis=1)
    return names, L[m]


def mcs(
    losses: dict[str, np.ndarray],
    alpha: float = 0.10,
    n_bootstrap: int = 2000,
    block_size: int | None = None,
    seed: int = 0,
    h: int = 1,
) -> dict:
    """Bootstrap Model Confidence Set (Hansen–Lunde–Nason style sequential elimination).

    Returns the set of models not rejected at level ``alpha``, bootstrap p-values for elimination,
    and the pairwise DM matrix on the same aligned sample.

    ``h`` is the forecast horizon of the losses: the pairwise DM tests are evaluated with
    ``h - 1`` autocovariances and the HLN correction (h-step forecast errors behave like
    MA(h-1)), and the bootstrap block is never shorter than ``h`` so that overlapping
    multi-day targets are not chopped mid-window. Limitations: the elimination statistic
    bootstraps the worst-vs-current-best loss difference (not the max-statistic over the
    full active set as in the canonical HLN-MCS), so p-values are indicative, not an exact
    level-alpha procedure under temporal dependence.
    """
    names, L = _loss_matrix(losses)
    if int(h) < 1:
        raise ValueError(f"forecast horizon h must be >= 1, got {h}")
    if L.shape[0] < 20 or L.shape[1] < 2:
        return {
            "models": names,
            "confidence_set": names,
            "elimination_p": {k: None for k in names},
            "mean_loss": {k: float(np.mean(L[:, i])) for i, k in enumerate(names)},
            "dm_stat": {},
            "dm_p": {},
        }
    rng = np.random.default_rng(seed)
    T, M = L.shape
    block = block_size or max(1, int(np.ceil(T ** (1 / 3))))
    block = max(block, int(h))
    active = list(range(M))
    elim_p: dict[str, float | None] = {names[i]: None for i in range(M)}

    while len(active) > 1:
        La = L[:, active]
        mean_a = La.mean(axis=0)
        best = int(np.argmin(mean_a))
        rel = La - La[:, [best]]
        t_stats = rel.mean(axis=0) / np.sqrt(np.maximum(rel.var(axis=0, ddof=1) / T, 1e-18))
        t_stats[best] = -np.inf
        worst_local = int(np.argmax(t_stats))
        worst = active[worst_local]
        d = rel[:, worst_local] - rel[:, worst_local].mean()
        boot_max = []
        for _ in range(n_bootstrap):
            idx = _block_bootstrap_indices(T, block, rng)
            db = d[idx]
            boot_max.append(db.mean() / np.sqrt(np.maximum(db.var(ddof=1) / T, 1e-18)))
        stat = float(t_stats[worst_local])
        p = float(np.mean(np.asarray(boot_max) >= stat))
        prev = elim_p[names[worst]]
        elim_p[names[worst]] = p if prev is None else max(prev, p)
        if p <= alpha:
            active = [i for i in active if i != worst]
        else:
            break

    conf = [names[i] for i in active]
    dm_stat, dm_p = dm_matrix({names[i]: L[:, i] for i in range(M)}, h=h)
    return {
        "models": names,
        "confidence_set": conf,
        "elimination_p": elim_p,
        "mean_loss": {names[i]: float(L[:, i].mean()) for i in range(M)},
        "dm_stat": dm_stat.to_dict(),
        "dm_p": dm_p.to_dict(),
    }


def _block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    while len(out) < n:
        start = int(rng.integers(0, max(1, n - block + 1)))
        out.extend(range(start, min(start + block, n)))
    return np.asarray(out[:n])


def dm_pairwise(losses: dict[str, np.ndarray], h: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise DM tests (re-export for MCS reporting pipelines)."""
    return dm_matrix(losses, h=h)


def rank_by_mean_loss(losses: dict[str, np.ndarray]) -> pd.Series:
    names, L = _loss_matrix(losses)
    return pd.Series(L.mean(axis=0), index=names).sort_values()
