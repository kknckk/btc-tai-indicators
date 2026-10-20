"""Shared utilities for the neural forecasters in this package.

Kept deliberately framework-light: seeding, sliding-window construction,
a RevIN (Reversible Instance Normalisation) layer, and a generic
early-stopping training loop reused by both :mod:`models.moderntcn` and
:mod:`models.patchtst`.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class NotFittedError(RuntimeError):
    """Raised when `predict()` (or similar) is called before `fit()`."""


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs for deterministic runs (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_windows(
    X: np.ndarray, y: Optional[np.ndarray], window: int
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """Build sliding windows of length `window` over the time axis of `X`.

    For row index ``i`` (``i >= window - 1``) the window is
    ``X[i - window + 1 : i + 1]`` and, if `y` is given, the associated
    label is ``y[i]`` (i.e. labels must already be aligned/shifted by the
    caller -- no look-ahead is introduced here).

    Returns `(X_windows, y_windows_or_None, row_indices)`. `row_indices`
    maps each window back to its right-edge position in the original
    array, which callers use to left-pad predictions with NaN.
    """
    n = len(X)
    if n < window:
        empty_x = np.empty((0, window, X.shape[1]), dtype=X.dtype)
        empty_y = None if y is None else np.empty((0,), dtype=np.float64)
        return empty_x, empty_y, []

    idx = list(range(window - 1, n))
    Xw = np.stack([X[i - window + 1 : i + 1] for i in idx]).astype(np.float32)
    if y is None:
        return Xw, None, idx
    yw = np.array([y[i] for i in idx], dtype=np.float64)
    return Xw, yw, idx


class RevIN(nn.Module):
    """Reversible Instance Normalisation (Kim et al., 2022).

    Normalises each (batch, channel) instance over the time axis using its
    own mean/std, optionally with a learnable affine transform. `denorm`
    is provided for completeness/tests even though the forecasters here
    only use the `norm` direction (the regression target is not the same
    physical quantity as the normalised input channels).
    """

    def __init__(self, n_vars: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_vars))
            self.bias = nn.Parameter(torch.zeros(n_vars))
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        # x: (batch, seq_len, n_vars)
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, unbiased=False, keepdim=True) + self.eps
            out = (x - self._mean) / self._std
            if self.affine:
                out = out * self.weight + self.bias
            return out
        if mode == "denorm":
            if self._mean is None or self._std is None:
                raise NotFittedError("RevIN.forward(mode='denorm') called before 'norm'.")
            out = x
            if self.affine:
                out = (out - self.bias) / (self.weight + self.eps)
            return out * self._std + self._mean
        raise ValueError(f"Unknown RevIN mode: {mode}")


@dataclass
class TrainConfig:
    """Fields every model config must expose for `train_loop`."""

    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    use_amp: bool = False
    device: str = "cpu"


def train_loop(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[nn.Module, int]:
    """Generic Adam + ReduceLROnPlateau + early-stopping training loop.

    Mirrors the pattern already used in
    `analytical_pipeline/run_paper6_dl_v2.py::train_single_model`.
    Uses `torch.amp.autocast` with mixed precision only when CUDA is
    available and `cfg.use_amp` is set; otherwise trains in plain fp32
    on CPU.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, cfg.patience // 2)
    )
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    patience_cnt = 0
    epochs_used = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(bx)
                    loss = criterion(pred, by)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step(val_loss)
        epochs_used = epoch + 1

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            patience_cnt = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.to("cpu")
    return model, epochs_used


def timed_train_loop(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[nn.Module, int, float]:
    """`train_loop` wrapper that also returns wall-clock train seconds."""
    start = time.perf_counter()
    model, epochs_used = train_loop(model, X_tr, y_tr, X_val, y_val, cfg)
    elapsed = time.perf_counter() - start
    return model, epochs_used, elapsed
