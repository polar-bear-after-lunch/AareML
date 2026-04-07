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
        self.X = [torch.tensor(X[i]) for i in range(len(X))]
        self.y = [torch.tensor(y[i]) for i in range(len(y))]

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

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

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
            loss = criterion(pred, yb)
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
                val_loss += criterion(model(xb), yb).item() * len(xb)
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
                    print(f"  Early stop at epoch {epoch}  (best val={best_val:.5f})")
                break

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"tf={tf:.2f}  lr={optimiser.param_groups[0]['lr']:.2e}")

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
    ys = torch.stack([dataset[i][1] for i in range(len(dataset))]).numpy()
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


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    return ckpt
