"""
AareML — evaluation metrics.

All functions accept arrays shaped [N, horizon, n_targets] unless noted.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List

from .config import TARGETS, TARGET_LABELS


# ── Point metrics ─────────────────────────────────────────────────────────

def rmse_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """RMSE at each forecast step → [horizon, n_tgt]."""
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))


def mae_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAE at each forecast step → [horizon, n_tgt]."""
    return np.abs(y_true - y_pred).mean(axis=0)


def mean_rmse(y_true: np.ndarray, y_pred: np.ndarray,
              targets: List[str] = TARGETS) -> Dict[str, float]:
    """Mean RMSE across all horizon steps, per target."""
    assert y_true.shape == y_pred.shape, \
        f"metrics: y_true {y_true.shape} and y_pred {y_pred.shape} shape mismatch"
    assert y_true.ndim == 3, \
        f"metrics: expected 3D arrays [N, H, T], got {y_true.ndim}D"
    assert not np.isnan(y_true).any(), "metrics: NaN in y_true"
    per = rmse_per_step(y_true, y_pred)          # [H, n_tgt]
    return {t: float(per[:, i].mean()) for i, t in enumerate(targets)}


def mean_mae(y_true: np.ndarray, y_pred: np.ndarray,
             targets: List[str] = TARGETS) -> Dict[str, float]:
    """Mean MAE across all horizon steps, per target."""
    assert y_true.shape == y_pred.shape, \
        f"metrics: y_true {y_true.shape} and y_pred {y_pred.shape} shape mismatch"
    assert y_true.ndim == 3, \
        f"metrics: expected 3D arrays [N, H, T], got {y_true.ndim}D"
    assert not np.isnan(y_true).any(), "metrics: NaN in y_true"
    per = mae_per_step(y_true, y_pred)
    return {t: float(per[:, i].mean()) for i, t in enumerate(targets)}


def nse(y_true: np.ndarray, y_pred: np.ndarray,
        targets: List[str] = TARGETS) -> Dict[str, float]:
    """
    Nash-Sutcliffe Efficiency per target (flattened over all windows+steps).
    NSE = 1 → perfect; NSE = 0 → as good as the mean; NSE < 0 → worse.
    """
    result = {}
    for i, t in enumerate(targets):
        obs = y_true[:, :, i].ravel()
        sim = y_pred[:, :, i].ravel()
        ss_res = np.sum((obs - sim) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        result[t] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return result


def kge(y_true: np.ndarray, y_pred: np.ndarray,
        targets: List[str] = TARGETS) -> Dict[str, float]:
    """
    Kling-Gupta Efficiency (Gupta et al. 2009) per target.
    KGE = 1 → perfect; KGE < -0.41 → worse than mean flow.
    Decomposes into correlation (r), bias ratio (β), variability ratio (γ).
    """
    result = {}
    for i, t in enumerate(targets):
        obs = y_true[:, :, i].ravel()
        sim = y_pred[:, :, i].ravel()
        # B6 fix: guard all three KGE components against degenerate predictions
        # (flat predictions produce nan correlation; use r=0, b=1, g=1 fallbacks)
        corr_mat = np.corrcoef(obs, sim)
        r = float(corr_mat[0, 1]) if np.isfinite(corr_mat[0, 1]) else 0.0
        b = float(sim.mean() / obs.mean()) if obs.mean() != 0 else 1.0
        if obs.std() > 0 and obs.mean() != 0 and sim.mean() != 0 and sim.std() > 0:
            g = float((sim.std() / sim.mean()) / (obs.std() / obs.mean()))
        else:
            g = 1.0
        kge_val = 1 - np.sqrt((r - 1)**2 + (b - 1)**2 + (g - 1)**2)
        result[t] = float(kge_val)
    return result


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                targets: List[str] = TARGETS) -> Dict[str, Dict[str, float]]:
    """
    Compute RMSE, MAE, NSE, KGE for all targets.
    Returns dict: {target: {metric: value}}.
    """
    r = mean_rmse(y_true, y_pred, targets)
    m = mean_mae(y_true, y_pred, targets)
    n = nse(y_true, y_pred, targets)
    k = kge(y_true, y_pred, targets)
    return {t: {"RMSE": r[t], "MAE": m[t], "NSE": n[t], "KGE": k[t]}
            for t in targets}


# ── Confidence intervals via temporal block bootstrap ─────────────────────

def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 500,
    block_size: int = 30,
    ci: float = 0.95,
    seed: int = 42,
    targets: List[str] = TARGETS,
) -> Dict[str, Dict[str, tuple]]:
    """
    Temporal block bootstrap confidence intervals for a scalar metric function.

    Parameters
    ----------
    y_true, y_pred : [N, horizon, n_tgt]
    metric_fn      : function(y_true, y_pred, targets) → {target: float}
    n_boot         : bootstrap replicates
    block_size     : contiguous block length (days) — preserves temporal autocorrelation
    ci             : confidence level (default 0.95)

    Returns
    -------
    {target: {"mean": float, "lo": float, "hi": float}}
    """
    rng   = np.random.default_rng(seed)
    N     = len(y_true)
    alpha = (1 - ci) / 2

    boot_scores = {t: [] for t in targets}

    for _ in range(n_boot):
        # Draw block-start indices with replacement
        n_blocks = int(np.ceil(N / block_size))
        starts   = rng.integers(0, max(1, N - block_size), size=n_blocks)
        idx      = np.concatenate([
            np.arange(s, min(s + block_size, N)) for s in starts
        ])[:N]
        scores = metric_fn(y_true[idx], y_pred[idx], targets)
        for t in targets:
            boot_scores[t].append(scores[t])

    result = {}
    for t in targets:
        arr = np.array(boot_scores[t])
        result[t] = {
            "mean": float(arr.mean()),
            "lo":   float(np.quantile(arr, alpha)),
            "hi":   float(np.quantile(arr, 1 - alpha)),
        }
    return result


def metrics_table(models: dict, y_true: np.ndarray,
                  targets: List[str] = TARGETS,
                  n_boot: int = 300) -> "pd.DataFrame":
    """
    Build a tidy comparison DataFrame for multiple models.

    Parameters
    ----------
    models : {model_name: y_pred array}
    y_true : ground-truth array [N, H, n_tgt]
    n_boot : bootstrap replicates for CIs (set 0 to skip)

    Returns
    -------
    DataFrame with columns: Model, Target, RMSE, MAE, NSE, KGE,
    and optionally RMSE_lo / RMSE_hi for bootstrap CIs.
    """
    import pandas as pd

    rows = []
    for name, y_pred in models.items():
        m = all_metrics(y_true, y_pred, targets)
        if n_boot > 0:
            ci_rmse = block_bootstrap_ci(
                y_true, y_pred, mean_rmse, n_boot=n_boot, targets=targets
            )
            ci_mae = block_bootstrap_ci(
                y_true, y_pred, mean_mae, n_boot=n_boot, targets=targets
            )
        for t in targets:
            row = {
                "Model":  name,
                "Target": TARGET_LABELS.get(t, t),
                "RMSE":   round(m[t]["RMSE"], 4),
                "MAE":    round(m[t]["MAE"],  4),
                "NSE":    round(m[t]["NSE"],  3),
                "KGE":    round(m[t]["KGE"],  3),
            }
            if n_boot > 0:
                row["RMSE_lo"] = round(ci_rmse[t]["lo"], 4)
                row["RMSE_hi"] = round(ci_rmse[t]["hi"], 4)
                # F2: also compute MAE bootstrap CIs for consistency
                row["MAE_lo"]  = round(ci_mae[t]["lo"],  4)
                row["MAE_hi"]  = round(ci_mae[t]["hi"],  4)
            rows.append(row)

    df = pd.DataFrame(rows)
    assert len(df) > 0, "metrics_table: result DataFrame is empty"
    if __debug__:
        for name in models:
            print(f"[metrics] metrics_table for '{name}':")
            print(df[df["Model"] == name].to_string(index=False))
    return df
