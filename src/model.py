"""
AareML — Seq2Seq LSTM model, dataset, training loop, and inference.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from .config import LOOKBACK, HORIZON, N_FEAT, N_TGT, TARGETS, SEED

# ── CPU performance settings ──────────────────────────────────────────────
# PyTorch CPU performance is best with 4-6 threads — more threads cause
# contention and are actually slower due to synchronisation overhead.
# The notebook cell sets OMP_NUM_THREADS before import; we just clamp here.
import os as _os
_n_logical = _os.cpu_count() or 1
# Use at most 6 threads — empirically optimal for LSTM on macOS
_n_threads = min(_n_logical, 6)
torch.set_num_threads(_n_threads)


# ── Dataset ───────────────────────────────────────────────────────────────

class RiverDataset(Dataset):
    """
    Sliding-window PyTorch dataset built from pre-windowed numpy arrays.

    Parameters
    ----------
    X : float32 [N, lookback, n_feat]  — already imputed, scaled
    y : float32 [N, horizon,  n_tgt]   — scaled targets
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 3, f"RiverDataset: X must be 3D [N, L, F], got {X.shape}"
        assert y.ndim == 3, f"RiverDataset: y must be 3D [N, H, T], got {y.shape}"
        assert not np.isnan(X).any(), "RiverDataset: NaN in X — impute before creating dataset"
        assert not np.isnan(y).any(), "RiverDataset: NaN in y — targets must be fully observed"
        assert X.dtype == np.float32, f"RiverDataset: X dtype should be float32, got {X.dtype}"
        assert y.dtype == np.float32, f"RiverDataset: y dtype should be float32, got {y.dtype}"
        if __debug__:
            print(f"[model] RiverDataset: {len(X)} samples, X={X.shape}, y={y.shape}")
        # B1 fix: store as single stacked tensors (not per-sample list) for speed
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────

class Seq2SeqLSTM(nn.Module):
    """
    Encoder-decoder LSTM for multi-step time-series forecasting.

    Architecture
    ------------
    Encoder LSTM reads the lookback window and passes its final hidden
    state to the decoder. The decoder generates predictions one step at
    a time (auto-regressive). Teacher forcing is applied during training
    and annealed to zero by mid-training.

    Parameters
    ----------
    n_feat   : number of input features
    n_tgt    : number of output targets
    hidden   : LSTM hidden size
    n_layers : number of stacked LSTM layers
    dropout  : dropout probability (applied between LSTM layers and before fc)
    """

    def __init__(self, n_feat: int = N_FEAT, n_tgt: int = N_TGT,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_tgt   = n_tgt
        self.horizon = HORIZON

        enc_drop = dropout if n_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=n_feat, hidden_size=hidden,
            num_layers=n_layers, dropout=enc_drop, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=n_tgt, hidden_size=hidden,
            num_layers=n_layers, dropout=enc_drop, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_tgt)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        y_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                     : [batch, lookback, n_feat]
        teacher_forcing_ratio : probability of using ground truth as decoder input
        y_target              : [batch, horizon, n_tgt] — required if tf_ratio > 0

        Returns
        -------
        [batch, horizon, n_tgt]
        """
        _, (h, c) = self.encoder(x)

        batch = x.size(0)
        dec_in = torch.zeros(batch, 1, self.n_tgt, device=x.device)

        outputs = []
        for t in range(self.horizon):
            out, (h, c) = self.decoder(dec_in, (h, c))
            pred = self.fc(self.drop(out))          # [batch, 1, n_tgt]
            outputs.append(pred)

            use_tf = (
                teacher_forcing_ratio > 0.0
                and y_target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            dec_in = y_target[:, t : t + 1, :] if use_tf else pred.detach()

        return torch.cat(outputs, dim=1)            # [batch, horizon, n_tgt]



# ── Seq2Seq GRU (for ablation study — same interface as Seq2SeqLSTM) ──────────

class Seq2SeqGRU(nn.Module):
    """
    Encoder-decoder GRU — drop-in replacement for Seq2SeqLSTM.

    Identical interface to Seq2SeqLSTM; GRU hidden state is a single tensor
    (not a tuple), so the forward pass is slightly different internally.
    Used in the ablation study (notebook 11) to compare GRU vs LSTM.
    """

    def __init__(self, n_feat: int = N_FEAT, n_tgt: int = N_TGT,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_tgt   = n_tgt
        self.horizon = HORIZON

        enc_drop = dropout if n_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=n_feat, hidden_size=hidden,
            num_layers=n_layers, dropout=enc_drop, batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=n_tgt, hidden_size=hidden,
            num_layers=n_layers, dropout=enc_drop, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_tgt)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        y_target: Optional[torch.Tensor] = None,
        **kwargs,                                   # absorb unused LSTM kwargs
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                     : [batch, lookback, n_feat]
        teacher_forcing_ratio : probability of using ground truth as decoder input
        y_target              : [batch, horizon, n_tgt] — required if tf_ratio > 0

        Returns
        -------
        [batch, horizon, n_tgt]
        """
        _, h = self.encoder(x)                      # GRU: hidden only (no cell state)

        batch = x.size(0)
        dec_in = torch.zeros(batch, 1, self.n_tgt, device=x.device)

        outputs = []
        for t in range(self.horizon):
            out, h = self.decoder(dec_in, h)        # GRU: single hidden tensor
            pred = self.fc(self.drop(out))          # [batch, 1, n_tgt]
            outputs.append(pred)

            use_tf = (
                teacher_forcing_ratio > 0.0
                and y_target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            dec_in = y_target[:, t : t + 1, :] if use_tf else pred.detach()

        return torch.cat(outputs, dim=1)            # [batch, horizon, n_tgt]


# ── Entity-Aware LSTM (I6) ─────────────────────────────────────────────────

class EALSTMCell(nn.Module):
    """
    Entity-Aware LSTM cell (Kratzert et al. 2019).

    Static catchment attributes modulate the input gate (i) and cell gate (g)
    via a learned embedding. The forget gate (f) and output gate (o) remain
    sequence-driven as in standard LSTM.

    Parameters
    ----------
    input_size   : number of dynamic input features per timestep
    hidden_size  : LSTM hidden / cell state size
    static_size  : number of static catchment attributes
    """

    def __init__(self, input_size: int, hidden_size: int, static_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Standard gates for forget and output (sequence-driven)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

        # Input and cell gates — additionally conditioned on static embedding
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)

        # Static attribute embedding (projects catchment attributes → hidden_size)
        # Used to scale the input and cell gate activations
        self.W_s_i = nn.Linear(static_size, hidden_size)
        self.W_s_g = nn.Linear(static_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,       # [batch, input_size]
        h: torch.Tensor,       # [batch, hidden_size]
        c: torch.Tensor,       # [batch, hidden_size]
        s: torch.Tensor,       # [batch, static_size]  — static attributes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_new, c_new)."""
        xh = torch.cat([x, h], dim=-1)             # [batch, input+hidden]

        f = torch.sigmoid(self.W_f(xh))
        o = torch.sigmoid(self.W_o(xh))

        # Static-modulated input and cell gates
        i = torch.sigmoid(self.W_i(xh) + self.W_s_i(s))
        g = torch.tanh(  self.W_g(xh) + self.W_s_g(s))

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class EASeq2SeqLSTM(nn.Module):
    """
    Entity-Aware Sequence-to-Sequence LSTM (I6).

    Extends Seq2SeqLSTM by incorporating static catchment attributes into
    the encoder's LSTM gating mechanism (Kratzert et al. 2019). The decoder
    uses a standard LSTM. This allows a single model trained across multiple
    gauges to adapt its internal dynamics to each catchment.

    Parameters
    ----------
    n_feat      : number of dynamic input features
    n_tgt       : number of output targets
    static_size : number of static catchment attributes
    hidden      : LSTM hidden size
    n_layers    : number of encoder layers (only first layer is EA; rest standard)
    dropout     : dropout probability
    """

    def __init__(
        self,
        n_feat:      int,
        n_tgt:       int,
        static_size: int,
        hidden:      int = 64,
        n_layers:    int = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.n_tgt      = n_tgt
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.horizon    = HORIZON

        # First encoder layer is EA-LSTM
        self.ea_cell = EALSTMCell(n_feat, hidden, static_size)

        # Remaining encoder layers (standard LSTM) if n_layers > 1
        if n_layers > 1:
            enc_drop = dropout if n_layers > 2 else 0.0
            self.encoder_upper = nn.LSTM(
                input_size=hidden, hidden_size=hidden,
                num_layers=n_layers - 1, dropout=enc_drop, batch_first=True,
            )
        else:
            self.encoder_upper = None

        # Decoder is standard LSTM
        dec_drop = dropout if n_layers > 1 else 0.0
        self.decoder = nn.LSTM(
            input_size=n_tgt, hidden_size=hidden,
            num_layers=n_layers, dropout=dec_drop, batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_tgt)

        # Note: static_proj removed — static attributes are passed directly
        # to EALSTMCell which handles projection internally via W_s_i / W_s_g.

    def _encode(
        self,
        x: torch.Tensor,    # [batch, lookback, n_feat]
        s: torch.Tensor,    # [batch, static_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run EA encoder, return (h, c) for decoder initialisation."""
        B, L, _ = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        c = torch.zeros(B, self.hidden, device=x.device)

        ea_outputs = []
        for t in range(L):
            h, c = self.ea_cell(x[:, t, :], h, c, s)
            ea_outputs.append(h.unsqueeze(1))

        ea_seq = torch.cat(ea_outputs, dim=1)          # [B, L, hidden]
        h_last = h.unsqueeze(0)                        # [1, B, hidden]
        c_last = c.unsqueeze(0)                        # [1, B, hidden]

        if self.encoder_upper is not None:
            upper_out, (h_last, c_last) = self.encoder_upper(
                ea_seq, (
                    h_last.expand(self.n_layers - 1, -1, -1).contiguous(),
                    c_last.expand(self.n_layers - 1, -1, -1).contiguous(),
                )
            )
            h_final = h_last
            c_final = c_last
        else:
            h_final = h_last
            c_final = c_last

        # Take last layer's hidden state for decoder init
        h_last_layer = h_final[-1:, :, :]  # [1, B, hidden]
        c_last_layer = c_final[-1:, :, :]  # [1, B, hidden]
        h_dec = h_last_layer.expand(self.n_layers, -1, -1).contiguous()
        c_dec = c_last_layer.expand(self.n_layers, -1, -1).contiguous()
        return h_dec, c_dec

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        y_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                     : [batch, lookback, n_feat]  dynamic inputs
        s                     : [batch, static_size]       catchment attributes
        teacher_forcing_ratio : float
        y_target              : [batch, horizon, n_tgt]

        Returns
        -------
        [batch, horizon, n_tgt]
        """
        h, c = self._encode(x, s)

        batch   = x.size(0)
        dec_in  = torch.zeros(batch, 1, self.n_tgt, device=x.device)
        outputs = []

        for t in range(self.horizon):
            out, (h, c) = self.decoder(dec_in, (h, c))
            pred = self.fc(self.drop(out))
            outputs.append(pred)

            use_tf = (
                teacher_forcing_ratio > 0.0
                and y_target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            dec_in = y_target[:, t:t+1, :] if use_tf else pred.detach()

        return torch.cat(outputs, dim=1)               # [batch, horizon, n_tgt]


class EARiverDataset(Dataset):
    """
    Extended RiverDataset that also carries a static catchment attribute vector.

    Parameters
    ----------
    X : float32 [N, lookback, n_feat]  — scaled dynamic inputs
    y : float32 [N, horizon,  n_tgt]   — scaled targets
    s : float32 [N, static_size]       — static attributes (same for all
                                         windows of the same gauge)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        assert X.shape[0] == y.shape[0] == s.shape[0]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.s = torch.from_numpy(s)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]


def train_ea_model(
    model: EASeq2SeqLSTM,
    dl_train: DataLoader,
    dl_val: DataLoader,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 12,
    teacher_forcing_start: float = 0.5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[EASeq2SeqLSTM, Dict]:
    """
    Train EASeq2SeqLSTM. DataLoaders must yield (x, y, s) triples
    from EARiverDataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val   = np.inf
    best_state = None
    wait       = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        tf = teacher_forcing_start * max(0.0, 1.0 - epoch / (epochs * 0.5))

        model.train()
        train_loss = 0.0
        for xb, yb, sb in dl_train:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            optimiser.zero_grad()
            pred = model(xb, sb, teacher_forcing_ratio=tf, y_target=yb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(dl_train.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, sb in dl_val:
                xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                val_loss += criterion(model(xb, sb), yb).item() * len(xb)
        val_loss /= len(dl_val.dataset)

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  EA early stop at epoch {epoch}  (best val={best_val:.5f})")
                break

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"tf={tf:.2f}  lr={optimiser.param_groups[0]['lr']:.2e}")

    if best_state is None:
        raise RuntimeError("Training diverged: validation loss never improved. Check for NaN inputs or bad learning rate.")
    model.load_state_dict(best_state)
    return model, history


# ── Training ──────────────────────────────────────────────────────────────

def train_model(
    model: Seq2SeqLSTM,
    dl_train: DataLoader,
    dl_val: DataLoader,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 12,
    teacher_forcing_start: float = 0.5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[Seq2SeqLSTM, Dict]:
    """
    Train with AdamW + ReduceLROnPlateau + early stopping.

    Teacher forcing ratio is linearly annealed from `teacher_forcing_start`
    to 0 over the first half of training, then held at 0.

    Returns the model loaded with the best-validation-loss weights, and a
    history dict with 'train' and 'val' loss lists.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(dl_train.dataset) > 0, "train_model: training DataLoader is empty"
    assert len(dl_val.dataset)   > 0, "train_model: validation DataLoader is empty"
    if len(dl_train) == 0:
        raise ValueError(f"Training DataLoader is empty — dataset has {len(dl_train.dataset)} samples but batch_size={dl_train.batch_size} with drop_last=True yields 0 batches. Reduce batch_size or increase data.")
    assert epochs > 0,   f"train_model: epochs must be > 0, got {epochs}"
    assert patience > 0, f"train_model: patience must be > 0, got {patience}"
    if __debug__:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] train_model: {n_params:,} trainable params, "
              f"device={device}, epochs={epochs}, patience={patience}, lr={lr}")

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=5
    )

    def nse_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                     alpha: float = 0.5) -> torch.Tensor:
        """
        Combined NSE + MSE loss.
        alpha=0.5 balances distributional fit (NSE) and point accuracy (MSE).
        NSE loss = 1 - NSE, so lower is better and range is (-inf, 1].
        Using mean across targets so multi-target models are weighted equally.
        """
        mse = torch.mean((pred - target) ** 2)
        var = torch.var(target, unbiased=False).clamp(min=1e-8)
        nse_loss = mse / var  # equivalent to 1 - NSE (without the constant 1)
        return alpha * nse_loss + (1.0 - alpha) * mse

    best_val   = np.inf
    best_state = None
    wait       = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        tf = teacher_forcing_start * max(0.0, 1.0 - epoch / (epochs * 0.5))

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb, teacher_forcing_ratio=tf, y_target=yb)
            loss = nse_mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(dl_train.dataset)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += nse_mse_loss(model(xb), yb).item() * len(xb)
        val_loss /= len(dl_val.dataset)

        assert np.isfinite(train_loss), \
            f"train_model: train loss is {train_loss} at epoch {epoch} — check for NaN inputs or exploding gradients"
        assert np.isfinite(val_loss), \
            f"train_model: val loss is {val_loss} at epoch {epoch} — check for NaN in validation data"

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  (best val={best_val:.5f})")
                break

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"tf={tf:.2f}  lr={optimiser.param_groups[0]['lr']:.2e}")

    if best_state is None:
        raise RuntimeError("Training diverged: validation loss never improved. Check for NaN inputs or bad learning rate.")
    model.load_state_dict(best_state)
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────

def predict(
    model: Seq2SeqLSTM,
    dataset: Dataset,
    tgt_scaler,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run inference and inverse-transform to original target units.

    Returns
    -------
    float32 array [N, horizon, n_tgt] in original (physical) units.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            out = model(xb.to(device)).cpu().numpy()
            preds.append(out)
    preds = np.concatenate(preds, axis=0)           # [N, H, n_tgt]
    assert preds.ndim == 3, f"predict: output should be 3D [N, H, T], got {preds.shape}"
    if np.isnan(preds).any():
        nan_frac = np.isnan(preds).mean()
        import warnings
        warnings.warn(f"predict: {nan_frac:.1%} NaN in predictions — model may have diverged")
    if __debug__:
        print(f"[model] predict: {preds.shape[0]} samples, "
              f"DO range [{preds[:,:,0].min():.2f}, {preds[:,:,0].max():.2f}] mg/L (scaled)")
    N, H, T = preds.shape
    return tgt_scaler.inverse_transform(
        preds.reshape(-1, T)
    ).reshape(N, H, T).astype(np.float32)


def get_y_true(dataset: Dataset, tgt_scaler) -> np.ndarray:
    """
    Extract ground-truth targets from a RiverDataset and inverse-transform.

    Returns
    -------
    float32 array [N, horizon, n_tgt] in original units.
    """
    # B2 fix: access stored tensor directly instead of looping sample-by-sample
    ys = dataset.y.numpy()
    N, H, T = ys.shape
    return tgt_scaler.inverse_transform(
        ys.reshape(-1, T)
    ).reshape(N, H, T).astype(np.float32)


# ── Checkpoint helpers ────────────────────────────────────────────────────

def save_checkpoint(path, model: Seq2SeqLSTM, best_params: dict,
                    feat_scaler, tgt_scaler):
    torch.save({
        "model_state":      model.state_dict(),
        "best_params":      best_params,
        "feat_scaler_mean":  feat_scaler.mean_,
        "feat_scaler_scale": feat_scaler.scale_,
        "tgt_scaler_mean":   tgt_scaler.mean_,
        "tgt_scaler_scale":  tgt_scaler.scale_,
    }, path)
    if __debug__:
        import os
        size_kb = os.path.getsize(path) / 1024
        print(f"[model] save_checkpoint: saved to {path} ({size_kb:.1f} KB)")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cpu")
    # weights_only=False is safe here — we only load checkpoints we generated ourselves.
    # If loading checkpoints from untrusted sources, set weights_only=True and use
    # a safetensors format instead.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    required_keys = {"model_state", "feat_scaler_mean", "tgt_scaler_mean"}
    missing = required_keys - set(ckpt.keys())
    assert not missing, f"load_checkpoint: checkpoint missing keys {missing}"
    if __debug__:
        print(f"[model] load_checkpoint: loaded from {path}, keys={list(ckpt.keys())}")
    return ckpt


def reconstruct_scalers(ckpt: dict):
    """
    F5: Rebuild StandardScaler objects from a saved checkpoint dict.

    Returns
    -------
    feat_scaler, tgt_scaler  — ready-to-use StandardScaler instances.
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    def _make_scaler(mean_, scale_):
        sc = StandardScaler()
        sc.mean_  = np.array(mean_)
        sc.scale_ = np.array(scale_)
        sc.var_   = sc.scale_ ** 2
        sc.n_features_in_ = len(mean_)
        # S-U3 fix: use a large representative number instead of 1.
        # Setting n_samples_seen_ = 1 would cause silently wrong running
        # statistics if partial_fit() is ever called on this reconstructed
        # scaler. 10000 is a safe approximation; persist the actual count
        # in the checkpoint if exact reproducibility of partial_fit is needed.
        sc.n_samples_seen_ = 10000
        return sc

    feat_scaler = _make_scaler(ckpt["feat_scaler_mean"], ckpt["feat_scaler_scale"])
    tgt_scaler  = _make_scaler(ckpt["tgt_scaler_mean"],  ckpt["tgt_scaler_scale"])
    return feat_scaler, tgt_scaler


def predict_single_window(
    model: "Seq2SeqLSTM",
    x_raw: np.ndarray,
    feat_scaler,
    tgt_scaler,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    F1: Forecast from a single raw input window.

    Parameters
    ----------
    x_raw       : float32 array [lookback, n_feat] in original (physical) units
    feat_scaler : fitted StandardScaler for features
    tgt_scaler  : fitted StandardScaler for targets

    Returns
    -------
    float32 array [horizon, n_tgt] in original (physical) units
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L, F = x_raw.shape
    x_scaled = feat_scaler.transform(x_raw).astype(np.float32)      # [L, F]
    x_tensor = torch.from_numpy(x_scaled).unsqueeze(0).to(device)   # [1, L, F]

    model.eval().to(device)
    with torch.no_grad():
        pred_scaled = model(x_tensor).squeeze(0).cpu().numpy()       # [H, T]

    return tgt_scaler.inverse_transform(pred_scaled).astype(np.float32)  # [H, T]
