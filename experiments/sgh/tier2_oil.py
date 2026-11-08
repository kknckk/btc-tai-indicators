"""Tier-II WTI oil forecasters: EMD, stacking (tier-I), sparse autoencoder (TRD-55)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.sgh.oil_panel import PRICE_COL
from experiments.sgh.tier1_oil import GarchMidasLevelForecaster, LSTMSVRLevelForecaster
from experiments.sgh.tier1_wheat import XGBLevelForecaster, _split_train_val, _val_mse
from models._common import TrainConfig, make_windows, set_seed


class _SparseAENet(nn.Module):
    """Sparse autoencoder bottleneck on the last window step → level head."""

    def __init__(self, n_feat: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)

    def encode_last(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x[:, -1, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_last(x)).squeeze(-1)


@dataclass
class _SparseAETrainer:
    window: int
    hidden: int
    latent: int
    lr: float
    max_epochs: int
    patience: int
    seed: int
    sparsity_weight: float
    _model: _SparseAENet | None = field(default=None, init=False, repr=False)
    _x_scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _y_mu: float = field(default=0.0, init=False)
    _y_sd: float = field(default=1.0, init=False)

    def fit_arrays(self, X_arr: np.ndarray, y_arr: np.ndarray) -> None:
        set_seed(self.seed)
        self._x_scaler = StandardScaler().fit(X_arr)
        xs = self._x_scaler.transform(X_arr)
        self._y_mu = float(np.nanmean(y_arr))
        self._y_sd = float(np.nanstd(y_arr)) or 1.0
        ys = (y_arr - self._y_mu) / self._y_sd
        Xw, yw, _ = make_windows(xs, ys, self.window)
        mask = np.isfinite(yw)
        Xw, yw = Xw[mask], yw[mask]
        if len(Xw) < 8:
            raise ValueError("not enough sparse-AE windows")
        n = len(Xw)
        val_n = max(1, int(n * 0.15))
        X_tr, y_tr = Xw[:-val_n], yw[:-val_n]
        X_val, y_val = Xw[-val_n:], yw[-val_n:]
        model = _SparseAENet(X_arr.shape[1], self.hidden, self.latent)
        cfg = TrainConfig(
            lr=self.lr,
            batch_size=min(64, max(1, len(X_tr))),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        sw = self.sparsity_weight

        def loss_fn(pred: torch.Tensor, target: torch.Tensor, batch_x: torch.Tensor) -> torch.Tensor:
            mse = nn.functional.mse_loss(pred, target)
            latent = model.encode_last(batch_x)
            l1 = latent.abs().mean()
            return mse + sw * l1

        device = torch.device("cpu")
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        best = float("inf")
        best_state = None
        wait = 0
        for _ in range(cfg.max_epochs):
            model.train()
            bx = torch.tensor(X_tr, dtype=torch.float32, device=device)
            by = torch.tensor(y_tr, dtype=torch.float32, device=device)
            opt.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by, bx)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                vx = torch.tensor(X_val, dtype=torch.float32, device=device)
                vy = torch.tensor(y_val, dtype=torch.float32, device=device)
                vpred = model(vx)
                vloss = float(loss_fn(vpred, vy, vx))
            if vloss < best:
                best = vloss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        model.to("cpu")
        self._model = model

    def predict_arrays(self, X_arr: np.ndarray) -> np.ndarray:
        if self._model is None or self._x_scaler is None:
            raise RuntimeError("sparse AE not fitted")
        xs = self._x_scaler.transform(X_arr)
        Xw, _, idx = make_windows(xs, None, self.window)
        out = np.full(len(X_arr), np.nan, dtype=float)
        if not len(idx):
            return out
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.tensor(Xw, dtype=torch.float32)).numpy()
        out[np.array(idx)] = pred * self._y_sd + self._y_mu
        return out


class EMDLevelForecaster:
    """MA-EMD components + trend → per-channel Ridge → meta Ridge (validation only)."""

    IMF_COLS = ("emd_hf", "emd_mf", "emd_lf")

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.val_frac = float(cfg.get("val_frac", 0.2))
        self.alpha_grid = tuple(cfg.get("ridge_alpha_grid", (0.1, 1.0, 10.0)))
        self._channels: dict[str, Ridge] = {}
        self._meta: Ridge | None = None
        self._diag: dict = {}

    def _channel_matrix(self, X: pd.DataFrame, imf: str) -> pd.DataFrame:
        cols = [imf, f"{imf}_l1", f"{imf}_l2"]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(f"EMD features missing {missing}")
        return X[cols]

    def _trend_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        ma = f"{self.price_col}_ma63"
        if ma not in X.columns:
            raise KeyError(f"missing trend column {ma}")
        return pd.DataFrame(
            {
                "trend": X[ma],
                "emd_trend_l1": X.get("emd_trend_l1", X[ma].shift(1)),
                "emd_trend_l2": X.get("emd_trend_l2", X[ma].shift(2)),
            },
            index=X.index,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EMDLevelForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        best_alpha = float(self.alpha_grid[0])
        best_stack_mse = float("inf")

        for alpha in self.alpha_grid:
            preds_va: list[np.ndarray] = []
            try:
                for imf in self.IMF_COLS:
                    feats = self._channel_matrix(Xf, imf)
                    ridge = Ridge(alpha=float(alpha))
                    ridge.fit(feats.iloc[tr_sl], yf.iloc[tr_sl])
                    preds_va.append(ridge.predict(feats.iloc[va_sl]))
                trend = self._trend_matrix(Xf)
                tr_ridge = Ridge(alpha=float(alpha))
                tr_ridge.fit(trend.iloc[tr_sl], yf.iloc[tr_sl])
                preds_va.append(tr_ridge.predict(trend.iloc[va_sl]))
            except KeyError:
                continue
            stack_va = np.column_stack(preds_va)
            if not np.isfinite(stack_va).all():
                continue
            meta = Ridge(alpha=float(alpha))
            meta.fit(stack_va, y_va.to_numpy())
            mse = _val_mse(y_va, meta.predict(stack_va))
            if mse < best_stack_mse:
                best_stack_mse = mse
                best_alpha = float(alpha)

        if not np.isfinite(best_stack_mse):
            self._meta = Ridge(alpha=1.0)
            spot = Xf[self.price_col].iloc[va_sl].to_numpy()
            self._meta.fit(np.ones((len(spot), 1)), spot)
            self._channels = {}
            self._diag = {"meta_fallback": True}
            return self

        self._channels = {}
        for imf in self.IMF_COLS:
            feats = self._channel_matrix(Xf, imf)
            ridge = Ridge(alpha=float(best_alpha))
            ridge.fit(feats, yf)
            self._channels[imf] = ridge
        trend = self._trend_matrix(Xf)
        tr_ridge = Ridge(alpha=float(best_alpha))
        tr_ridge.fit(trend, yf)
        self._channels["trend"] = tr_ridge
        stack_full = np.column_stack(
            [self._channels[c].predict(self._channel_matrix(Xf, c)) for c in self.IMF_COLS]
            + [self._channels["trend"].predict(trend)]
        )
        self._meta = Ridge(alpha=float(best_alpha))
        self._meta.fit(stack_full, yf.to_numpy())
        self._diag = {"ridge_alpha": best_alpha, "val_mse": best_stack_mse}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._meta is not None
        if not self._channels:
            return X[self.price_col].to_numpy(dtype=float)
        stack = np.column_stack(
            [self._channels[c].predict(self._channel_matrix(X, c)) for c in self.IMF_COLS]
            + [self._channels["trend"].predict(self._trend_matrix(X))]
        )
        pred = self._meta.predict(stack)
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "emd", **self._diag}


class OilStackingForecaster:
    """Stack tier-I oil models (LSTM-SVR, XGBoost, GARCH-MIDAS) with Ridge meta-learner."""

    def __init__(
        self,
        cfg: dict,
        seed: int,
        horizon: int,
        price_col: str,
        returns: pd.Series,
    ) -> None:
        self.cfg = cfg
        self.seed = seed
        self.h = horizon
        self.price_col = price_col
        self.returns = returns
        self.val_frac = float(cfg.get("val_frac", 0.2))
        vf = self.val_frac
        lc = dict(cfg.get("lstm_svr", {}))
        lc["val_frac"] = vf
        xc = dict(cfg.get("xgboost", {}))
        xc["val_frac"] = vf
        gm = dict(cfg.get("garch_midas", {}))
        self._lstm = LSTMSVRLevelForecaster(lc, seed, price_col)
        self._xgb = XGBLevelForecaster(xc, seed, price_col)
        self._garch = GarchMidasLevelForecaster(
            returns,
            price_col,
            horizon,
            val_frac=vf,
            k_grid=tuple(gm.get("k_grid", (3, 4, 6))),
            theta1_grid=tuple(gm.get("theta1_grid", (1.0, 2.0))),
            theta2_grid=tuple(gm.get("theta2_grid", (5.0, 10.0))),
            mean_lags_grid=tuple(gm.get("mean_lags_grid", (0, 1))),
            dist_grid=tuple(gm.get("dist_grid", ("normal", "t"))),
            rv_window=int(gm.get("rv_window", 22)),
            state={},
        )
        self._meta: Ridge | None = None
        self._diag: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> OilStackingForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        tr_sl, va_sl = _split_train_val(len(Xf), self.val_frac)
        X_va, y_va = Xf.iloc[va_sl], yf.iloc[va_sl]

        self._lstm.fit(Xf, yf)
        self._xgb.fit(Xf, yf)
        self._garch.returns = self.returns.reindex(Xf.index)
        self._garch.fit(Xf, yf)

        stack_va = np.column_stack(
            [
                self._lstm.predict(X_va),
                self._xgb.predict(X_va),
                self._garch.predict(X_va),
            ]
        )
        ok = np.isfinite(stack_va).all(axis=1) & y_va.notna().to_numpy()
        self._meta = Ridge(alpha=float(self.cfg.get("ridge_alpha", 1.0)))
        if ok.sum() < 5:
            self._meta.fit(np.ones((len(y_va), 1)), y_va.to_numpy())
            self._diag = {"meta_fallback": True}
            return self
        self._meta.fit(stack_va[ok], y_va.to_numpy()[ok])
        self._diag = {
            "meta_coef": self._meta.coef_.tolist(),
            "val_mse": _val_mse(y_va, self._meta.predict(stack_va)),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._meta is not None
        stack = np.column_stack(
            [
                self._lstm.predict(X),
                self._xgb.predict(X),
                self._garch.predict(X),
            ]
        )
        pred = self._meta.predict(stack)
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {"model": "stacking", **self._diag}


class SparseAutoencoderForecaster:
    """Windowed sparse autoencoder representation → level forecast."""

    def __init__(self, cfg: dict, seed: int, price_col: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.price_col = price_col
        self.window = int(cfg.get("window", 42))
        self.hidden = int(cfg.get("hidden", 32))
        self.latent = int(cfg.get("latent", 8))
        self.lr = float(cfg.get("lr", 1e-3))
        self.max_epochs = int(cfg.get("max_epochs", 45))
        self.patience = int(cfg.get("patience", 8))
        self.sparsity_weight = float(cfg.get("sparsity_weight", 0.01))
        self._trainer: _SparseAETrainer | None = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SparseAutoencoderForecaster:
        m = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[m], y.loc[m]
        prefer = [c for c in Xf.columns if c.startswith(("emd_", "ret_z"))]
        rest = [c for c in Xf.columns if c not in prefer]
        self._cols = prefer + rest
        tr = _SparseAETrainer(
            window=self.window,
            hidden=self.hidden,
            latent=self.latent,
            lr=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
            sparsity_weight=self.sparsity_weight,
        )
        tr.fit_arrays(Xf[self._cols].to_numpy(dtype=float), yf.to_numpy(dtype=float))
        self._trainer = tr
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._trainer is not None
        pred = self._trainer.predict_arrays(X[self._cols].to_numpy(dtype=float))
        spot = X[self.price_col].to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, spot)

    def diagnostics(self) -> dict:
        return {
            "model": "sparse_ae",
            "window": self.window,
            "latent": self.latent,
            "sparsity_weight": self.sparsity_weight,
        }


def model_factory(
    name: str,
    cfg: dict,
    *,
    horizon: int,
    price_col: str,
    returns: pd.Series | None = None,
) -> Callable[[], object]:
    seed = int(cfg.get("seed", 42))
    vf = float(cfg.get("val_frac", 0.2))
    if name == "emd":
        ec = dict(cfg.get("emd", {}))
        ec["val_frac"] = vf
        return lambda: EMDLevelForecaster(ec, seed, price_col)
    if name == "stacking":
        if returns is None:
            raise ValueError("stacking requires returns series")
        st = dict(cfg.get("stacking", {}))
        st["val_frac"] = vf
        st.setdefault("lstm_svr", cfg.get("lstm_svr", {}))
        st.setdefault("xgboost", cfg.get("xgboost", {}))
        st.setdefault("garch_midas", cfg.get("garch_midas", {}))
        return lambda: OilStackingForecaster(st, seed, horizon, price_col, returns)
    if name == "sparse_ae":
        sa = dict(cfg.get("sparse_ae", {}))
        return lambda: SparseAutoencoderForecaster(sa, seed, price_col)
    raise KeyError(f"unknown tier-II oil model {name!r}")
