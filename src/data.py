"""
AareML — shared data loading, preprocessing, and windowing utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from .config import (
    DAILY_DIR, METADATA_FILE,
    FEATURES, TARGETS,
    LOOKBACK, HORIZON,
    TRAIN_END, VAL_END,
)


# ── Load ───────────────────────────────────────────────────────────────────

def load_gauge(gauge_id: str | int, daily_dir: Path = DAILY_DIR) -> pd.DataFrame:
    """Load a single gauge's daily sensor CSV and return a date-indexed DataFrame."""
    path = daily_dir / f"camels_ch_chem_daily_{gauge_id}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return df


def load_metadata() -> pd.DataFrame:
    """Load gauge metadata table."""
    return pd.read_csv(METADATA_FILE)


def list_available_gauges(daily_dir: Path = DAILY_DIR) -> List[str]:
    """Return sorted list of all gauge IDs with a daily sensor file."""
    return sorted(
        p.stem.replace("camels_ch_chem_daily_", "")
        for p in daily_dir.glob("camels_ch_chem_daily_*.csv")
    )


def do_coverage(gauge_id: str, daily_dir: Path = DAILY_DIR) -> float:
    """Return fraction of non-null DO observations for a gauge (0–1)."""
    df = load_gauge(gauge_id, daily_dir)
    if "O2C_sensor" not in df.columns:
        return 0.0
    return float(df["O2C_sensor"].notna().mean())


def gauges_with_do(min_coverage: float = 0.10,
                   daily_dir: Path = DAILY_DIR) -> List[str]:
    """Return gauge IDs where DO coverage >= min_coverage."""
    return [
        g for g in list_available_gauges(daily_dir)
        if do_coverage(g, daily_dir) >= min_coverage
    ]


# ── Preprocess ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """
    Reindex to a daily DatetimeIndex and linearly interpolate gaps ≤ max_gap.
    Longer gaps remain as NaN.
    """
    df = df.copy()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    for col in df.columns:
        df[col] = df[col].interpolate(method="linear", limit=max_gap)
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split into train / val / test.
    Uses .iloc[1:] to avoid a one-row overlap at each boundary.
    """
    train = df.loc[:train_end]
    val   = df.loc[train_end:val_end].iloc[1:]
    test  = df.loc[val_end:].iloc[1:]
    return train, val, test


# ── Windowing ─────────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    train_means: pd.Series,
    lookback: int = LOOKBACK,
    horizon:  int = HORIZON,
    features: List[str] = FEATURES,
    targets:  List[str] = TARGETS,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build sliding-window (X, y) arrays.

    Parameters
    ----------
    df          : DataFrame with feature and target columns, daily DatetimeIndex.
    train_means : Series of column means from the *training* split — used for
                  NaN imputation to prevent data leakage.
    lookback    : Number of input days.
    horizon     : Number of forecast days.
    features    : Input feature column names.
    targets     : Target column names.

    Returns
    -------
    X     : float32 array [N, lookback, n_feat]
    y     : float32 array [N, horizon,  n_tgt]
    dates : DatetimeIndex of length N (date of first forecast day)
    """
    # Impute NaN with training means (no leakage)
    # Deduplicate columns (features and targets may overlap, e.g. temp_sensor / O2C_sensor)
    all_cols = list(dict.fromkeys(features + targets))
    df_imp = df[all_cols].copy()

    # Build a clean {col: scalar} dict from train_means (handles duplicated index)
    if hasattr(train_means, 'index'):
        means_dict = (
            train_means
            .groupby(level=0).first()   # collapse duplicates, keep first value
            .to_dict()
        )
    else:
        means_dict = {}

    for col in df_imp.columns:
        fill_val = means_dict.get(col, None)
        if fill_val is None or (hasattr(fill_val, '__len__')):
            fill_val = float(df_imp[col].mean()) if df_imp[col].notna().any() else 0.0
        df_imp[col] = df_imp[col].fillna(float(fill_val))

    feat_arr = df_imp[features].values.astype(np.float32)
    tgt_arr  = df_imp[targets].values.astype(np.float32)
    raw_tgt  = df[targets].values.astype(np.float32)   # pre-imputation for mask

    X_list, y_list, date_list = [], [], []
    n = len(df)
    for i in range(lookback, n - horizon + 1):
        y_raw_win = raw_tgt[i : i + horizon]
        if np.isnan(y_raw_win).any():
            continue                               # drop if any target is NaN
        X_list.append(feat_arr[i - lookback : i])
        y_list.append(tgt_arr[i : i + horizon])
        date_list.append(df.index[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, pd.DatetimeIndex(date_list)
