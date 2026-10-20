"""Neural forecasting models (ModernTCN, PatchTST) for BTC on-chain data.

Both forecasters share the same scikit-style contract so they can be
used interchangeably by `experiments/run_neural.py` (and, once merged,
by `evaluation/walk_forward.py`):

    model.fit(X: pd.DataFrame, y: pd.Series) -> self
    model.predict(X: pd.DataFrame) -> np.ndarray  # len(X), NaN-padded
"""
from models._common import NotFittedError
from models.moderntcn import ModernTCN, ModernTCNConfig
from models.patchtst import PatchTST, PatchTSTConfig

__all__ = [
    "ModernTCN",
    "ModernTCNConfig",
    "PatchTST",
    "PatchTSTConfig",
    "NotFittedError",
]
