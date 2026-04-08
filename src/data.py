"""
AareML — shared data loading, preprocessing, and windowing utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from .config import (
    DAILY_DIR, METADATA_FILE, NAWAF_DIR,
    FEATURES, TARGETS, NAWAF_FEATURES,
    LOOKBACK, HORIZON,
    TRAIN_END, VAL_END,
    SEED,
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

    # B4 fix: catch empty result early with a clear error message
    if len(X_list) == 0:
        raise ValueError(
            f"make_windows produced 0 valid windows. "
            f"Check NaN coverage — the target columns may be entirely missing "
            f"or the DataFrame is shorter than lookback + horizon = "
            f"{lookback + horizon} days."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, pd.DatetimeIndex(date_list)


# ── NAWA FRACHT (monthly chemistry grab samples) ───────────────────────────

def load_nawaf(gauge_id: str, nawaf_dir: "Path" = NAWAF_DIR) -> pd.DataFrame:
    """
    Load NAWA FRACHT monthly chemistry grab samples for a gauge.
    Returns a date-indexed DataFrame. Returns empty DataFrame if no file found.
    """
    import glob as _glob
    pattern = str(nawaf_dir / f"*_{gauge_id}_*.csv")
    files = _glob.glob(pattern)
    if not files:
        # Try alternate naming
        pattern2 = str(nawaf_dir / f"*{gauge_id}*.csv")
        files = _glob.glob(pattern2)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


def merge_nawaf_features(
    daily: pd.DataFrame,
    nawaf: pd.DataFrame,
    nawaf_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    I7: Merge monthly NAWA FRACHT chemistry features into the daily sensor DataFrame.

    Missing months are forward-filled (last known value carried forward) then
    backward-filled for the start of the record. This gives a slowly-varying
    covariate without leaking future chemistry information.

    Parameters
    ----------
    daily      : daily sensor DataFrame (date-indexed)
    nawaf      : NAWA FRACHT DataFrame (monthly, date-indexed)
    nawaf_cols : columns to include; defaults to NAWAF_FEATURES

    Returns
    -------
    DataFrame with daily index and extra chemistry columns appended.
    """
    from .config import NAWAF_FEATURES
    if nawaf_cols is None:
        nawaf_cols = NAWAF_FEATURES
    if nawaf.empty or not nawaf_cols:
        return daily

    # Keep only requested columns that actually exist
    available = [c for c in nawaf_cols if c in nawaf.columns]
    if not available:
        return daily

    nawaf_sel = nawaf[available].copy()
    # Resample to daily, forward-fill then backward-fill gaps
    nawaf_daily = (
        nawaf_sel
        .reindex(daily.index)
        .ffill()
        .bfill()
    )
    return pd.concat([daily, nawaf_daily], axis=1)


# ── Multi-site helper ──────────────────────────────────────────────────────

def score_gauge(
    gauge_id: str,
    feat_scaler,
    tgt_scaler,
    predict_fn,
    train_means_ref: "pd.Series",
    daily_dir: "Path" = DAILY_DIR,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    features: "List[str]" = FEATURES,
    targets: "List[str]" = TARGETS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    F4: Full pipeline for one gauge — load, preprocess, split, window,
    scale, and return (X_test_scaled, y_test_scaled) ready for predict().

    Parameters
    ----------
    gauge_id        : gauge identifier string
    feat_scaler     : fitted StandardScaler for features (from focus gauge)
    tgt_scaler      : fitted StandardScaler for targets  (from focus gauge)
    predict_fn      : callable(dataset) -> y_pred [N, H, T] in physical units
    train_means_ref : training means Series from the focus gauge (for imputation)

    Returns
    -------
    y_pred : float32 [N, H, n_tgt]  in physical units
    y_true : float32 [N, H, n_tgt]  in physical units
    """
    from sklearn.preprocessing import StandardScaler
    from .model import RiverDataset, get_y_true

    raw  = load_gauge(gauge_id, daily_dir)
    data = preprocess(raw)
    train, val, test = train_val_test_split(data, train_end, val_end)

    # Use gauge-specific training means for imputation (no leakage)
    g_means = (
        pd.concat([train[features].mean(), train[targets].mean()])
        .groupby(level=0).first()
    )

    X_test, y_test, _ = make_windows(test, g_means,
                                     features=features, targets=targets)

    # Scale with provided scalers (zero-shot) or refit per gauge
    N, L, F = X_test.shape
    _, H, T  = y_test.shape
    Xs = feat_scaler.transform(X_test.reshape(-1, F)).reshape(N, L, F).astype(np.float32)
    ys = tgt_scaler.transform(y_test.reshape(-1, T)).reshape(N, H, T).astype(np.float32)

    ds = RiverDataset(Xs, ys)
    y_pred = predict_fn(ds)
    y_true = get_y_true(ds, tgt_scaler)
    return y_pred, y_true
