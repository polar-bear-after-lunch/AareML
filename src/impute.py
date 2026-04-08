"""
AareML — Self-attention time-series imputer (SAITS-inspired).

Lightweight alternative to the full SAITS model (Du et al. 2023).
Uses a single self-attention layer to impute missing values by attending
over observed timesteps. Fitted on training data, applied to lookback
windows at inference time.

Reference:
    Du et al. (2023). SAITS: Self-Attention-based Imputation for Time Series.
    Expert Systems with Applications, 219, 119619.
    https://doi.org/10.1016/j.eswa.2023.119619
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class AttentionImputer(nn.Module):
    """
    Single-layer self-attention imputer for multivariate time series.

    For each timestep with a missing value, attends over observed timesteps
    in the same window and produces a weighted combination as the imputed value.

    Parameters
    ----------
    n_feat   : number of features (channels)
    d_model  : attention embedding dimension
    n_heads  : number of attention heads
    dropout  : dropout on attention weights
    """

    def __init__(self, n_feat: int, d_model: int = 32,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_feat, d_model)
        self.attn        = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.output_proj = nn.Linear(d_model, n_feat)
        self.norm        = nn.LayerNorm(d_model)
        self.drop        = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [batch, seq_len, n_feat]  — with NaN replaced by 0 as placeholder
        mask : [batch, seq_len]          — True where values are observed (not missing)

        Returns
        -------
        x_imp : [batch, seq_len, n_feat] — imputed (missing timesteps filled)
        """
        B, L, F = x.shape

        # Project to attention space
        h = self.input_proj(x)                          # [B, L, d_model]

        # Key/value mask: attend only over observed positions
        # attn key_padding_mask: True = IGNORE that position
        key_mask = ~mask                                 # [B, L]

        # Self-attention
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_mask)
        h = self.norm(h + self.drop(attn_out))          # residual + norm

        # Project back to feature space
        x_rec = self.output_proj(h)                     # [B, L, n_feat]

        # Only replace missing positions; keep observed values as-is
        obs_mask = mask.unsqueeze(-1).expand_as(x)      # [B, L, n_feat]
        x_imp = torch.where(obs_mask, x, x_rec)
        return x_imp


class SATSImputer:
    """
    Scikit-learn-style wrapper around AttentionImputer.

    Usage
    -----
    imputer = SATSImputer(n_feat=4)
    imputer.fit(X_train)          # X_train: [N, lookback, n_feat], may have NaN
    X_imp = imputer.transform(X)  # fills NaN in X
    """

    def __init__(self, n_feat: int, d_model: int = 32, n_heads: int = 4,
                 dropout: float = 0.1, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 128, device: Optional[str] = None):
        self.n_feat     = n_feat
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_: Optional[AttentionImputer] = None
        self._col_means: Optional[np.ndarray]   = None

    def fit(self, X: np.ndarray, verbose: bool = False) -> "SATSImputer":
        """
        Train the imputer using observed values as supervision.

        Strategy: for each window, randomly mask out 15% of *observed*
        timesteps, ask the model to reconstruct them, and use MSE loss.
        Column means are stored as fallback for fully-missing columns.

        Parameters
        ----------
        X : float32 [N, lookback, n_feat]  — training windows, may contain NaN
        """
        N, L, F = X.shape
        self._col_means = np.nanmean(X.reshape(-1, F), axis=0)
        # Replace remaining NaN in col_means with 0
        self._col_means = np.where(np.isfinite(self._col_means),
                                   self._col_means, 0.0)

        self.model_ = AttentionImputer(
            F, self.d_model, self.n_heads, self.dropout
        ).to(self.device)
        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Build observed mask [N, L, F]
        obs_mask_np = np.isfinite(X).astype(np.float32)      # 1 = observed
        # Fill NaN with col means for model input
        X_filled = np.where(np.isfinite(X), X, self._col_means[None, None, :])
        X_t   = torch.from_numpy(X_filled.astype(np.float32))
        obs_t = torch.from_numpy(obs_mask_np)

        rng = np.random.default_rng(42)

        for epoch in range(self.epochs):
            idx = rng.permutation(N)
            total_loss = 0.0
            n_batches  = 0

            for start in range(0, N, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb  = X_t[batch_idx].to(self.device)    # [B, L, F]
                ob  = obs_t[batch_idx].to(self.device)  # [B, L, F]

                # Create artificial masks: hide 15% of observed timesteps per sample
                # Use per-timestep mask (all features at a timestep masked together)
                obs_ts  = ob[:, :, 0].bool()             # [B, L] timestep observed
                rand_m  = torch.rand(obs_ts.shape, device=self.device) > 0.15  # keep 85%
                train_mask = obs_ts & rand_m             # timesteps model can attend to
                target_mask = obs_ts & ~rand_m           # timesteps to reconstruct

                if target_mask.sum() == 0:
                    continue

                xb_masked = torch.where(
                    train_mask.unsqueeze(-1).expand_as(xb), xb,
                    torch.zeros_like(xb)
                )

                optimiser.zero_grad()
                x_imp = self.model_(xb_masked, train_mask)

                # Loss only on artificially masked positions
                loss_mask = target_mask.unsqueeze(-1).expand_as(xb)
                loss = criterion(x_imp[loss_mask], xb[loss_mask])
                loss.backward()
                optimiser.step()

                total_loss += loss.item()
                n_batches  += 1

            if verbose and (epoch + 1) % 10 == 0:
                avg = total_loss / max(n_batches, 1)
                print(f"  SATSImputer epoch {epoch+1:3d}/{self.epochs}  loss={avg:.5f}")

        self.model_.eval()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in X.

        Parameters
        ----------
        X : float32 [N, lookback, n_feat]

        Returns
        -------
        X_imp : float32 [N, lookback, n_feat]  — NaN replaced
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before transform().")

        N, L, F = X.shape
        obs_mask = np.isfinite(X).astype(np.float32)
        X_filled = np.where(np.isfinite(X), X,
                             self._col_means[None, None, :])

        X_t   = torch.from_numpy(X_filled.astype(np.float32))
        obs_t = torch.from_numpy(obs_mask[:, :, 0])  # [N, L]

        results = []
        with torch.no_grad():
            for start in range(0, N, 256):
                xb  = X_t[start:start+256].to(self.device)
                ob  = obs_t[start:start+256].to(self.device)
                x_imp = self.model_(xb, ob.bool())
                results.append(x_imp.cpu().numpy())

        return np.concatenate(results, axis=0).astype(np.float32)

    def fit_transform(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        return self.fit(X, verbose=verbose).transform(X)
