"""ModernTCN forecaster (Saidd 2026, arXiv:2603.16886; architecture after
Luo & Wang, ICLR 2024 "ModernTCN: A Modern Pure Convolution Structure for
General Time Series Analysis").

Depthwise-separable large-kernel temporal convolution blocks (a depthwise
conv over the time axis followed by a pointwise Conv-FFN) stacked on top of
a RevIN-normalised, linearly embedded multivariate window. Single scalar
regression head (horizon-ahead log return).

Public contract (compatible with the `experiments/data.py` /
`evaluation/walk_forward.py` interfaces expected from the parallel branch):

    model = ModernTCN(ModernTCNConfig(...))
    model.fit(X: pd.DataFrame, y: pd.Series) -> self
    model.predict(X: pd.DataFrame) -> np.ndarray
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from models._common import NotFittedError, RevIN, make_windows, set_seed, timed_train_loop

logger = logging.getLogger(__name__)


@dataclass
class ModernTCNConfig:
    window: int = 60
    d_model: int = 128
    ffn_dim: int = 288
    kernel_size: int = 25
    num_blocks: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    val_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"
    use_amp: bool = False


class _DepthwiseSeparableBlock(nn.Module):
    """Large-kernel depthwise conv (temporal mixing) + pointwise Conv-FFN
    (channel mixing), each with its own residual connection -- the core
    ModernTCN block."""

    def __init__(self, d_model: int, ffn_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model
        )
        self.norm1 = nn.BatchNorm1d(d_model)
        self.pw1 = nn.Conv1d(d_model, ffn_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(ffn_dim, d_model, kernel_size=1)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model, seq_len)
        residual = x
        y = self.depthwise(x)
        y = self.norm1(y)
        x = residual + self.dropout(y)

        residual = x
        y = self.pw1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.pw2(y)
        y = self.norm2(y)
        x = residual + self.dropout(y)
        return x


class ModernTCNNet(nn.Module):
    def __init__(self, n_vars: int, cfg: ModernTCNConfig):
        super().__init__()
        self.revin = RevIN(n_vars)
        self.embed = nn.Linear(n_vars, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                _DepthwiseSeparableBlock(cfg.d_model, cfg.ffn_dim, cfg.kernel_size, cfg.dropout)
                for _ in range(cfg.num_blocks)
            ]
        )
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars)
        x = self.revin(x, mode="norm")
        x = self.embed(x)  # (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)  # temporal pooling -> (batch, d_model)
        return self.head(x).squeeze(-1)


class ModernTCN:
    """Scikit-style forecaster wrapping `ModernTCNNet`."""

    def __init__(self, cfg: Optional[ModernTCNConfig] = None):
        self.cfg = cfg or ModernTCNConfig()
        self._model: Optional[ModernTCNNet] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_columns: Optional[List[str]] = None
        self._fitted = False
        self.epochs_used_: Optional[int] = None
        self.train_seconds_: Optional[float] = None

    @property
    def n_params(self) -> int:
        if self._model is None:
            raise NotFittedError("Model has no parameters before fit().")
        return sum(p.numel() for p in self._model.parameters())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModernTCN":
        set_seed(self.cfg.seed)
        self._feature_columns = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy(dtype=np.float64)

        self._scaler = StandardScaler().fit(X_arr)
        X_scaled = self._scaler.transform(X_arr)

        Xw, yw, _ = make_windows(X_scaled, y_arr, self.cfg.window)
        mask = ~np.isnan(yw)
        Xw, yw = Xw[mask], yw[mask]
        if len(Xw) < 4:
            raise ValueError(
                f"Not enough samples ({len(Xw)}) after windowing with window={self.cfg.window}."
            )

        n = len(Xw)
        val_size = max(1, int(n * self.cfg.val_fraction))
        val_size = min(val_size, n - 1) if n > 1 else 0
        train_size = n - val_size
        X_tr, y_tr = Xw[:train_size], yw[:train_size]
        if val_size > 0:
            X_val, y_val = Xw[train_size:], yw[train_size:]
        else:
            X_val, y_val = X_tr, y_tr

        n_vars = X_arr.shape[1]
        model = ModernTCNNet(n_vars, self.cfg)
        model, epochs_used, elapsed = timed_train_loop(model, X_tr, y_tr, X_val, y_val, self.cfg)

        self._model = model
        self.epochs_used_ = epochs_used
        self.train_seconds_ = elapsed
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._model is None or self._scaler is None:
            raise NotFittedError("Call fit() before predict().")
        X_arr = X.to_numpy(dtype=np.float64)
        X_scaled = self._scaler.transform(X_arr)
        Xw, _, idx = make_windows(X_scaled, None, self.cfg.window)

        out = np.full(len(X_arr), np.nan, dtype=np.float64)
        if len(idx) == 0:
            return out

        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(Xw, dtype=torch.float32)).numpy()
        out[idx] = preds
        return out
