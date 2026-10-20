"""PatchTST forecaster (Nie et al., ICLR 2023, "A Time Series is Worth 64
Words"; benchmarked again in Saidd 2026, arXiv:2603.16886).

Patchifies each (RevIN-normalised) input window, embeds patches linearly,
adds a learnable positional encoding, and runs a standard
`nn.TransformerEncoder`. A `channel_independent` flag switches between the
paper's channel-independence design (shared encoder weights applied per
variable, representations averaged across channels at the head) and a
plain multivariate-patch variant (all channels embedded jointly per patch).

Single scalar regression head (horizon-ahead log return), matching the
same `fit`/`predict` contract as `models.moderntcn.ModernTCN`.
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
class PatchTSTConfig:
    window: int = 128
    patch_len: int = 16
    stride: int = 16
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    channel_independent: bool = True
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    val_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"
    use_amp: bool = False

    @property
    def num_patches(self) -> int:
        return (self.window - self.patch_len) // self.stride + 1


class PatchTSTNet(nn.Module):
    def __init__(self, n_vars: int, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_vars = n_vars
        self.channel_independent = cfg.channel_independent
        self.num_patches = cfg.num_patches
        if self.num_patches < 1:
            raise ValueError("window must be >= patch_len for at least one patch.")

        self.revin = RevIN(n_vars)

        patch_input_dim = cfg.patch_len if self.channel_independent else cfg.patch_len * n_vars
        self.patch_embed = nn.Linear(patch_input_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, cfg.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        flat_dim = self.num_patches * cfg.d_model
        if self.channel_independent:
            self.channel_head = nn.Linear(flat_dim, cfg.d_model)
            self.out_head = nn.Linear(cfg.d_model, 1)
        else:
            self.out_head = nn.Linear(flat_dim, 1)

    def _make_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars) -> (batch, n_vars, num_patches, patch_len)
        x = x.transpose(1, 2)  # (batch, n_vars, seq_len)
        return x.unfold(dimension=2, size=self.cfg.patch_len, step=self.cfg.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode="norm")
        patches = self._make_patches(x)  # (B, C, P, patch_len)
        b, c, p, pl = patches.shape

        if self.channel_independent:
            z = patches.reshape(b * c, p, pl)
            z = self.patch_embed(z) + self.pos_embed
            z = self.encoder(z)
            z = z.reshape(b * c, -1)
            z = self.channel_head(z)  # (B*C, d_model)
            z = z.reshape(b, c, -1).mean(dim=1)  # cross-channel aggregation -> (B, d_model)
            return self.out_head(z).squeeze(-1)

        z = patches.permute(0, 2, 1, 3).reshape(b, p, c * pl)  # (B, P, C*patch_len)
        z = self.patch_embed(z) + self.pos_embed
        z = self.encoder(z)
        z = z.reshape(b, -1)
        return self.out_head(z).squeeze(-1)


class PatchTST:
    """Scikit-style forecaster wrapping `PatchTSTNet`."""

    def __init__(self, cfg: Optional[PatchTSTConfig] = None):
        self.cfg = cfg or PatchTSTConfig()
        self._model: Optional[PatchTSTNet] = None
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PatchTST":
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
        model = PatchTSTNet(n_vars, self.cfg)
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
