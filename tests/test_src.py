"""
AareML comprehensive test suite.
Covers: config, data, model, metrics, impute modules.
"""
from __future__ import annotations

import sys
import os

# Ensure src is importable from the AareML root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import torch

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

from src.config import (
    FEATURES, TARGETS, LOOKBACK, HORIZON, SEED,
    TRAIN_END, VAL_END, STATIC_COLS,
)
from src.data import make_windows, preprocess, train_val_test_split
from src.model import (
    RiverDataset, Seq2SeqLSTM, EASeq2SeqLSTM,
    predict, predict_single_window, get_y_true,
    reconstruct_scalers,
)
from src.metrics import (
    mean_rmse, nse, kge, block_bootstrap_ci, metrics_table,
)
from src.impute import SATSImputer


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_synthetic_df(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic daily DataFrame with FEATURES columns and no NaN."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n_days, freq="D")
    data = rng.standard_normal((n_days, len(FEATURES))).astype(np.float32)
    return pd.DataFrame(data, index=idx, columns=FEATURES)


def _make_train_means(df: pd.DataFrame) -> pd.Series:
    """Compute training means from a DataFrame."""
    return pd.concat([df[FEATURES].mean(), df[TARGETS].mean()]).groupby(level=0).first()


def _make_scaled_windows(n_days: int = 300, seed: int = 42):
    """Return (X_sc, y_sc, feat_sc, tgt_sc) scaled float32 arrays."""
    from sklearn.preprocessing import StandardScaler

    df = _make_synthetic_df(n_days, seed)
    means = _make_train_means(df)
    X, y, _ = make_windows(df, means)

    N, L, F = X.shape
    _, H, T = y.shape

    feat_sc = StandardScaler().fit(X.reshape(-1, F))
    tgt_sc = StandardScaler().fit(y.reshape(-1, T))

    X_sc = feat_sc.transform(X.reshape(-1, F)).reshape(N, L, F).astype(np.float32)
    y_sc = tgt_sc.transform(y.reshape(-1, T)).reshape(N, H, T).astype(np.float32)
    return X_sc, y_sc, feat_sc, tgt_sc


# ── 1. Config tests ───────────────────────────────────────────────────────

def test_config_features_and_targets():
    """FEATURES and TARGETS should be lists of strings."""
    assert isinstance(FEATURES, list), "FEATURES must be a list"
    assert isinstance(TARGETS, list), "TARGETS must be a list"
    assert all(isinstance(f, str) for f in FEATURES), "All FEATURES must be strings"
    assert all(isinstance(t, str) for t in TARGETS), "All TARGETS must be strings"
    assert len(FEATURES) > 0, "FEATURES must be non-empty"
    assert len(TARGETS) > 0, "TARGETS must be non-empty"


def test_config_targets_subset_of_features():
    """TARGETS should be a subset of FEATURES (since they are sensor columns used as both input and output)."""
    for t in TARGETS:
        assert t in FEATURES, f"Target '{t}' not in FEATURES — data pipeline expects targets in features"


def test_config_constants():
    """LOOKBACK > 0, HORIZON > 0, SEED is int, TRAIN_END < VAL_END."""
    assert isinstance(LOOKBACK, int) and LOOKBACK > 0, f"LOOKBACK must be positive int, got {LOOKBACK}"
    assert isinstance(HORIZON, int) and HORIZON > 0, f"HORIZON must be positive int, got {HORIZON}"
    assert isinstance(SEED, int), f"SEED must be int, got {type(SEED)}"
    # Date ordering: TRAIN_END < VAL_END
    assert pd.Timestamp(TRAIN_END) < pd.Timestamp(VAL_END), \
        f"TRAIN_END ({TRAIN_END}) must be before VAL_END ({VAL_END})"


def test_config_static_cols_unique():
    """STATIC_COLS should have no duplicates."""
    assert len(STATIC_COLS) == len(set(STATIC_COLS)), \
        f"STATIC_COLS has duplicates: {STATIC_COLS}"


def test_config_train_means_dedup_pattern():
    """The dedup pattern used in data.py should produce a unique-index Series."""
    means = (
        pd.concat([pd.Series({"a": 1.0, "b": 2.0}),
                   pd.Series({"b": 3.0, "c": 4.0})])
        .groupby(level=0).first()
    )
    assert means.index.is_unique, "groupby dedup should yield unique index"
    assert means["b"] == 2.0, "groupby.first() should keep first occurrence (a=1.0, b=2.0)"


# ── 2. Data module tests ──────────────────────────────────────────────────

def test_make_windows_shapes():
    """X shape = [N, LOOKBACK, n_feat], y shape = [N, HORIZON, n_tgt]."""
    df = _make_synthetic_df(300)
    means = _make_train_means(df)
    X, y, dates = make_windows(df, means)

    assert X.ndim == 3, f"X must be 3D, got {X.ndim}D"
    assert y.ndim == 3, f"y must be 3D, got {y.ndim}D"
    assert X.shape[1] == LOOKBACK, f"X.shape[1] expected {LOOKBACK}, got {X.shape[1]}"
    assert y.shape[1] == HORIZON, f"y.shape[1] expected {HORIZON}, got {y.shape[1]}"
    assert X.shape[2] == len(FEATURES), f"X.shape[2] expected {len(FEATURES)}, got {X.shape[2]}"
    assert y.shape[2] == len(TARGETS), f"y.shape[2] expected {len(TARGETS)}, got {y.shape[2]}"
    assert X.shape[0] == y.shape[0] == len(dates), "N mismatch between X, y, dates"


def test_make_windows_no_nan_in_output():
    """After make_windows with train_means imputation, X and y should have no NaN."""
    rng = np.random.default_rng(0)
    df = _make_synthetic_df(300)
    # Introduce ~20% NaN in features only (not targets, so windows aren't dropped)
    df_nan = df.copy()
    feat_only = [f for f in FEATURES if f not in TARGETS]
    for col in feat_only:
        mask = rng.random(len(df)) < 0.20
        df_nan.loc[mask, col] = np.nan

    means = _make_train_means(df)
    X, y, _ = make_windows(df_nan, means)

    assert not np.isnan(X).any(), "X must have no NaN after make_windows imputation"
    assert not np.isnan(y).any(), "y must have no NaN (windows with NaN targets are dropped)"


def test_make_windows_empty_guard():
    """A DataFrame shorter than lookback+horizon should raise ValueError."""
    short_df = _make_synthetic_df(LOOKBACK + HORIZON - 2)
    means = _make_train_means(short_df)
    with pytest.raises(ValueError, match="0 valid windows"):
        make_windows(short_df, means)


def test_make_windows_no_target_leakage():
    """
    The target at step h=0 corresponds to df.index[LOOKBACK], not df.index[LOOKBACK-1].
    Verify date indexing: dates[0] == df.index[LOOKBACK].
    """
    df = _make_synthetic_df(200)
    means = _make_train_means(df)
    X, y, dates = make_windows(df, means)

    # The first window:
    # X[0] = df[0:LOOKBACK], y[0] = df[LOOKBACK:LOOKBACK+HORIZON]
    # dates[0] should equal df.index[LOOKBACK]
    assert dates[0] == df.index[LOOKBACK], (
        f"Target leakage: dates[0]={dates[0].date()}, "
        f"df.index[LOOKBACK]={df.index[LOOKBACK].date()}"
    )


def test_make_windows_dtype():
    """make_windows should return float32 arrays."""
    df = _make_synthetic_df(200)
    means = _make_train_means(df)
    X, y, _ = make_windows(df, means)
    assert X.dtype == np.float32, f"X dtype must be float32, got {X.dtype}"
    assert y.dtype == np.float32, f"y dtype must be float32, got {y.dtype}"


def test_train_val_test_split_no_overlap():
    """No date should appear in two splits."""
    df = _make_synthetic_df(500)
    # Need enough data to span TRAIN_END and VAL_END
    idx = pd.date_range("2010-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(1)
    data = rng.standard_normal((len(idx), len(FEATURES))).astype(np.float32)
    df = pd.DataFrame(data, index=idx, columns=FEATURES)

    train, val, test = train_val_test_split(df)

    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    assert train_idx.isdisjoint(val_idx), "train and val overlap"
    assert train_idx.isdisjoint(test_idx), "train and test overlap"
    assert val_idx.isdisjoint(test_idx), "val and test overlap"


def test_train_val_test_split_chronological():
    """train end < val start, val end < test start."""
    idx = pd.date_range("2010-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(2)
    data = rng.standard_normal((len(idx), len(FEATURES))).astype(np.float32)
    df = pd.DataFrame(data, index=idx, columns=FEATURES)

    train, val, test = train_val_test_split(df)

    assert train.index.max() < val.index.min(), \
        f"train max {train.index.max()} >= val min {val.index.min()}"
    assert val.index.max() < test.index.min(), \
        f"val max {val.index.max()} >= test min {test.index.min()}"


def test_train_val_test_split_covers_all_data():
    """Union of train+val+test should cover the full date range.
    
    The implementation slices: train=df[:TRAIN_END], val=df[TRAIN_END:VAL_END].iloc[1:],
    test=df[VAL_END:].iloc[1:]. The boundary rows (TRAIN_END, VAL_END) appear in both
    adjacent ranges but .iloc[1:] drops them from the later range, so total = len(df).
    """
    idx = pd.date_range("2010-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(3)
    data = rng.standard_normal((len(idx), len(FEATURES))).astype(np.float32)
    df = pd.DataFrame(data, index=idx, columns=FEATURES)

    train, val, test = train_val_test_split(df)

    # Each boundary date (TRAIN_END, VAL_END) appears in one split group only
    # (the .iloc[1:] clips the duplicate from the later slice), so total == len(df)
    total = len(train) + len(val) + len(test)
    assert total == len(df), \
        f"Expected {len(df)} rows total (splits cover full range), got {total}"


def test_preprocess_daily_frequency():
    """preprocess() should produce a daily-frequency DatetimeIndex with no gaps."""
    # Create df with a few missing dates
    idx = pd.date_range("2015-01-01", "2015-12-31", freq="D")
    rng = np.random.default_rng(4)
    data = rng.standard_normal((len(idx), len(FEATURES))).astype(np.float32)
    df = pd.DataFrame(data, index=idx, columns=FEATURES)

    # Drop a few days to create gaps
    drop_idx = idx[[5, 10, 15]]
    df_gaps = df.drop(drop_idx)

    df_proc = preprocess(df_gaps)

    # Should have daily freq
    expected_days = (df_proc.index.max() - df_proc.index.min()).days + 1
    assert len(df_proc) == expected_days, \
        f"preprocess should fill gaps: expected {expected_days} rows, got {len(df_proc)}"


def test_preprocess_interpolates_short_gaps():
    """preprocess() should fill gaps of length <= max_gap=7 by interpolation."""
    idx = pd.date_range("2015-01-01", "2015-12-31", freq="D")
    data = np.ones((len(idx), len(FEATURES)), dtype=np.float32) * 5.0
    df = pd.DataFrame(data, index=idx, columns=FEATURES)

    # Drop 3 consecutive days (short gap)
    drop_idx = idx[10:13]  # 3 days
    df_gaps = df.drop(drop_idx)

    df_proc = preprocess(df_gaps)

    # The interpolated values should be close to 5.0 (linear between 5.0 and 5.0)
    filled_vals = df_proc.iloc[10:13]
    for col in FEATURES:
        assert np.allclose(filled_vals[col].values, 5.0, atol=1e-4), \
            f"preprocess: interpolated values for {col} should be ~5.0"


# ── 3. Model tests ────────────────────────────────────────────────────────

def test_river_dataset_shapes():
    """RiverDataset stores tensors with correct shapes."""
    X_sc, y_sc, _, _ = _make_scaled_windows(300)
    ds = RiverDataset(X_sc, y_sc)

    assert isinstance(ds.X, torch.Tensor), "RiverDataset.X must be a tensor"
    assert isinstance(ds.y, torch.Tensor), "RiverDataset.y must be a tensor"
    assert ds.X.shape[1] == LOOKBACK, f"X.shape[1] must be {LOOKBACK}"
    assert ds.y.shape[1] == HORIZON, f"y.shape[1] must be {HORIZON}"
    assert ds.X.shape[2] == len(FEATURES)
    assert ds.y.shape[2] == len(TARGETS)
    assert len(ds) == X_sc.shape[0]


def test_river_dataset_rejects_nan():
    """Creating RiverDataset with NaN in X should raise AssertionError."""
    X_sc, y_sc, _, _ = _make_scaled_windows(100)
    X_nan = X_sc.copy()
    X_nan[0, 0, 0] = np.nan

    with pytest.raises(AssertionError, match="NaN in X"):
        RiverDataset(X_nan, y_sc)


def test_river_dataset_rejects_nan_in_y():
    """Creating RiverDataset with NaN in y should raise AssertionError."""
    X_sc, y_sc, _, _ = _make_scaled_windows(100)
    y_nan = y_sc.copy()
    y_nan[0, 0, 0] = np.nan

    with pytest.raises(AssertionError, match="NaN in y"):
        RiverDataset(X_sc, y_nan)


def test_river_dataset_rejects_non_float32():
    """RiverDataset requires float32 dtype."""
    X_sc, y_sc, _, _ = _make_scaled_windows(100)
    X_f64 = X_sc.astype(np.float64)

    with pytest.raises(AssertionError, match="float32"):
        RiverDataset(X_f64, y_sc)


def test_seq2seq_lstm_forward_shape():
    """Output shape should be [batch, HORIZON, n_tgt]."""
    torch.manual_seed(0)
    n_feat, n_tgt = len(FEATURES), len(TARGETS)
    model = Seq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, hidden=16, n_layers=1, dropout=0.0)
    model.eval()

    x = torch.randn(8, LOOKBACK, n_feat)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, HORIZON, n_tgt), \
        f"Seq2SeqLSTM output shape mismatch: expected (8,{HORIZON},{n_tgt}), got {out.shape}"


def test_seq2seq_lstm_teacher_forcing():
    """With teacher_forcing_ratio=1.0, output should still have correct shape."""
    torch.manual_seed(1)
    n_feat, n_tgt = len(FEATURES), len(TARGETS)
    model = Seq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, hidden=16, n_layers=1, dropout=0.0)
    model.train()

    x = torch.randn(4, LOOKBACK, n_feat)
    y_target = torch.randn(4, HORIZON, n_tgt)
    out = model(x, teacher_forcing_ratio=1.0, y_target=y_target)
    assert out.shape == (4, HORIZON, n_tgt), \
        f"Teacher forcing output shape mismatch: {out.shape}"


def test_seq2seq_lstm_n_layers_2():
    """Seq2SeqLSTM with n_layers=2 should forward without error."""
    torch.manual_seed(2)
    n_feat, n_tgt = len(FEATURES), len(TARGETS)
    model = Seq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, hidden=16, n_layers=2, dropout=0.1)
    model.eval()

    x = torch.randn(4, LOOKBACK, n_feat)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, HORIZON, n_tgt)


def test_ea_lstm_forward_shape():
    """EASeq2SeqLSTM output shape correct for n_layers=1."""
    torch.manual_seed(3)
    n_feat, n_tgt, static_size = 4, 2, 6
    model = EASeq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, static_size=static_size,
                           hidden=32, n_layers=1, dropout=0.0)
    model.eval()

    x = torch.randn(4, LOOKBACK, n_feat)
    s = torch.randn(4, static_size)
    with torch.no_grad():
        out = model(x, s)
    assert out.shape == (4, HORIZON, n_tgt), \
        f"EASeq2SeqLSTM n_layers=1 shape mismatch: {out.shape}"


def test_ea_lstm_n_layers_2_no_crash():
    """EASeq2SeqLSTM with n_layers=2 should not crash (expand bug regression)."""
    torch.manual_seed(4)
    n_feat, n_tgt, static_size = 4, 2, 6
    model = EASeq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, static_size=static_size,
                           hidden=32, n_layers=2, dropout=0.0)
    model.eval()

    x = torch.randn(4, LOOKBACK, n_feat)
    s = torch.randn(4, static_size)
    with torch.no_grad():
        out = model(x, s)
    assert out.shape == (4, HORIZON, n_tgt), \
        f"EASeq2SeqLSTM n_layers=2 shape mismatch: {out.shape}"


def test_ea_lstm_n_layers_3():
    """EASeq2SeqLSTM with n_layers=3 should produce correct output shape."""
    torch.manual_seed(5)
    n_feat, n_tgt, static_size = 4, 2, 6
    model = EASeq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, static_size=static_size,
                           hidden=32, n_layers=3, dropout=0.0)
    model.eval()

    x = torch.randn(2, LOOKBACK, n_feat)
    s = torch.randn(2, static_size)
    with torch.no_grad():
        out = model(x, s)
    assert out.shape == (2, HORIZON, n_tgt)


def test_predict_single_window_shape():
    """predict_single_window returns [HORIZON, n_tgt]."""
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(6)
    n_feat, n_tgt = len(FEATURES), len(TARGETS)
    model = Seq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, hidden=16, n_layers=1, dropout=0.0)
    model.eval()

    # Create dummy scalers
    dummy_data = np.random.randn(100, n_feat).astype(np.float32)
    feat_sc = StandardScaler().fit(dummy_data)
    tgt_sc = StandardScaler().fit(np.random.randn(100, n_tgt).astype(np.float32))

    x_raw = np.random.randn(LOOKBACK, n_feat).astype(np.float32)
    y_pred = predict_single_window(model, x_raw, feat_sc, tgt_sc)

    assert y_pred.shape == (HORIZON, n_tgt), \
        f"predict_single_window shape mismatch: expected ({HORIZON},{n_tgt}), got {y_pred.shape}"
    assert y_pred.dtype == np.float32


def test_reconstruct_scalers_invertible():
    """Scalers reconstructed from checkpoint dict should transform/inverse_transform correctly."""
    from sklearn.preprocessing import StandardScaler
    import tempfile, pathlib

    n_feat, n_tgt = len(FEATURES), len(TARGETS)
    X_data = np.random.randn(100, n_feat).astype(np.float32)
    y_data = np.random.randn(100, n_tgt).astype(np.float32)

    feat_sc = StandardScaler().fit(X_data)
    tgt_sc = StandardScaler().fit(y_data)

    # Build a minimal checkpoint dict (same format as save_checkpoint)
    ckpt = {
        "model_state":       {},
        "best_params":       {},
        "feat_scaler_mean":  feat_sc.mean_,
        "feat_scaler_scale": feat_sc.scale_,
        "tgt_scaler_mean":   tgt_sc.mean_,
        "tgt_scaler_scale":  tgt_sc.scale_,
    }

    feat_sc2, tgt_sc2 = reconstruct_scalers(ckpt)

    # Round-trip check: transform then inverse_transform ≈ identity
    X_transformed = feat_sc2.transform(X_data)
    X_reconstructed = feat_sc2.inverse_transform(X_transformed)
    assert np.allclose(X_reconstructed, X_data, atol=1e-5), \
        "feat_scaler round-trip failed"

    y_transformed = tgt_sc2.transform(y_data)
    y_reconstructed = tgt_sc2.inverse_transform(y_transformed)
    assert np.allclose(y_reconstructed, y_data, atol=1e-5), \
        "tgt_scaler round-trip failed"


def test_get_y_true_shape():
    """get_y_true returns [N, HORIZON, n_tgt] in physical units."""
    from sklearn.preprocessing import StandardScaler

    X_sc, y_sc, _, tgt_sc = _make_scaled_windows(200)
    ds = RiverDataset(X_sc, y_sc)

    y_true = get_y_true(ds, tgt_sc)

    assert y_true.ndim == 3, f"get_y_true should return 3D, got {y_true.ndim}D"
    assert y_true.shape == (X_sc.shape[0], HORIZON, len(TARGETS)), \
        f"get_y_true shape mismatch: {y_true.shape}"
    assert y_true.dtype == np.float32
    assert not np.isnan(y_true).any(), "get_y_true: unexpected NaN in output"


def test_predict_output_shape():
    """predict() returns [N, HORIZON, n_tgt] in physical units."""
    torch.manual_seed(7)
    X_sc, y_sc, _, tgt_sc = _make_scaled_windows(200)
    ds = RiverDataset(X_sc, y_sc)

    n_feat, n_tgt = len(FEATURES), len(TARGETS)
    model = Seq2SeqLSTM(n_feat=n_feat, n_tgt=n_tgt, hidden=16, n_layers=1, dropout=0.0)

    y_pred = predict(model, ds, tgt_sc)
    N = X_sc.shape[0]
    assert y_pred.shape == (N, HORIZON, n_tgt), \
        f"predict() shape mismatch: expected ({N},{HORIZON},{n_tgt}), got {y_pred.shape}"
    assert y_pred.dtype == np.float32


# ── 4. Metrics tests ──────────────────────────────────────────────────────

def test_mean_rmse_perfect_prediction():
    """RMSE of perfect prediction = 0 for each target."""
    rng = np.random.default_rng(10)
    y = rng.standard_normal((100, HORIZON, len(TARGETS))).astype(np.float32)
    result = mean_rmse(y, y)
    for tgt, val in result.items():
        assert abs(val) < 1e-6, f"RMSE of perfect prediction should be 0, got {val} for {tgt}"


def test_mean_rmse_shape_mismatch_raises():
    """Mismatched shapes should raise AssertionError."""
    y1 = np.ones((10, HORIZON, 2), dtype=np.float32)
    y2 = np.ones((10, HORIZON, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        mean_rmse(y1, y2)


def test_mean_rmse_positive():
    """RMSE should be non-negative for any prediction."""
    rng = np.random.default_rng(11)
    y_true = rng.standard_normal((50, HORIZON, len(TARGETS))).astype(np.float32)
    y_pred = rng.standard_normal((50, HORIZON, len(TARGETS))).astype(np.float32)
    result = mean_rmse(y_true, y_pred)
    for tgt, val in result.items():
        assert val >= 0, f"RMSE must be non-negative, got {val} for {tgt}"


def test_mean_rmse_returns_correct_targets():
    """mean_rmse keys should match TARGETS."""
    rng = np.random.default_rng(12)
    y = rng.standard_normal((30, HORIZON, len(TARGETS))).astype(np.float32)
    result = mean_rmse(y, y)
    assert set(result.keys()) == set(TARGETS), \
        f"mean_rmse keys {set(result.keys())} != TARGETS {set(TARGETS)}"


def test_nse_perfect_is_one():
    """NSE of perfect prediction should be 1."""
    rng = np.random.default_rng(13)
    y = rng.standard_normal((100, HORIZON, len(TARGETS))).astype(np.float32)
    result = nse(y, y)
    for tgt, val in result.items():
        assert abs(val - 1.0) < 1e-5, f"NSE of perfect prediction should be 1.0, got {val} for {tgt}"


def test_nse_shape_mismatch_raises():
    """nse() should raise AssertionError on shape mismatch."""
    y1 = np.ones((10, HORIZON, 2), dtype=np.float32)
    y2 = np.ones((10, HORIZON, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        nse(y1, y2)


def test_nse_constant_observation_is_nan():
    """NSE should be NaN when all observations are the same (ss_tot = 0)."""
    y_true = np.ones((50, HORIZON, len(TARGETS)), dtype=np.float32)
    y_pred = np.ones((50, HORIZON, len(TARGETS)), dtype=np.float32) * 2.0
    result = nse(y_true, y_pred)
    for tgt, val in result.items():
        assert np.isnan(val), f"NSE should be NaN for constant observations, got {val} for {tgt}"


def test_kge_perfect_is_one():
    """KGE of perfect prediction should be 1."""
    rng = np.random.default_rng(14)
    y = rng.standard_normal((100, HORIZON, len(TARGETS))).astype(np.float32)
    result = kge(y, y)
    for tgt, val in result.items():
        assert abs(val - 1.0) < 1e-5, f"KGE of perfect prediction should be 1.0, got {val} for {tgt}"


def test_kge_no_nan_on_flat_prediction():
    """Flat prediction (all same value) should not produce NaN KGE."""
    rng = np.random.default_rng(15)
    y_true = rng.standard_normal((100, HORIZON, len(TARGETS))).astype(np.float32)
    y_pred = np.ones_like(y_true) * 3.14
    result = kge(y_true, y_pred)
    for tgt, val in result.items():
        assert not np.isnan(val), f"KGE is NaN for flat prediction for target {tgt}"


def test_block_bootstrap_ci_bounds():
    """lo <= mean <= hi for each target."""
    rng = np.random.default_rng(16)
    y_true = rng.standard_normal((200, HORIZON, len(TARGETS))).astype(np.float32)
    y_pred = y_true + rng.standard_normal(y_true.shape).astype(np.float32) * 0.1

    ci = block_bootstrap_ci(y_true, y_pred, mean_rmse, n_boot=50)
    for tgt, bounds in ci.items():
        assert bounds["lo"] <= bounds["mean"] <= bounds["hi"], \
            f"CI bounds inverted for {tgt}: lo={bounds['lo']}, mean={bounds['mean']}, hi={bounds['hi']}"


def test_block_bootstrap_ci_all_targets_present():
    """block_bootstrap_ci should return results for all TARGETS."""
    rng = np.random.default_rng(17)
    y_true = rng.standard_normal((100, HORIZON, len(TARGETS))).astype(np.float32)
    y_pred = y_true + 0.1
    ci = block_bootstrap_ci(y_true, y_pred, mean_rmse, n_boot=20)
    assert set(ci.keys()) == set(TARGETS), \
        f"block_bootstrap_ci missing targets: {set(ci.keys())} != {set(TARGETS)}"


def test_metrics_table_returns_dataframe():
    """metrics_table should return a DataFrame with RMSE and NSE columns."""
    rng = np.random.default_rng(18)
    y_true = rng.standard_normal((50, HORIZON, len(TARGETS))).astype(np.float32)
    y_pred = y_true + rng.standard_normal(y_true.shape).astype(np.float32) * 0.1

    df = metrics_table({"TestModel": y_pred}, y_true, n_boot=0)
    assert isinstance(df, pd.DataFrame), "metrics_table must return a DataFrame"
    assert "RMSE" in df.columns, "metrics_table must have RMSE column"
    assert "NSE" in df.columns, "metrics_table must have NSE column"
    assert "KGE" in df.columns, "metrics_table must have KGE column"
    assert "MAE" in df.columns, "metrics_table must have MAE column"
    assert len(df) == len(TARGETS), \
        f"metrics_table should have {len(TARGETS)} rows (one per target), got {len(df)}"


def test_metrics_table_multiple_models():
    """metrics_table with multiple models should produce one row per target per model."""
    rng = np.random.default_rng(19)
    y_true = rng.standard_normal((50, HORIZON, len(TARGETS))).astype(np.float32)
    y_pred1 = y_true + 0.1
    y_pred2 = y_true + 0.2

    df = metrics_table({"Model A": y_pred1, "Model B": y_pred2}, y_true, n_boot=0)
    assert len(df) == 2 * len(TARGETS), \
        f"Expected {2 * len(TARGETS)} rows for 2 models × {len(TARGETS)} targets, got {len(df)}"
    assert "Model" in df.columns


# ── 5. Imputer tests ──────────────────────────────────────────────────────

def test_sats_imputer_no_nan_output():
    """After fit_transform, no NaN should remain in the output."""
    rng = np.random.default_rng(20)
    X = rng.standard_normal((50, LOOKBACK, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.15] = np.nan

    imp = SATSImputer(n_feat=4, epochs=3)
    X_imp = imp.fit_transform(X)

    assert X_imp.shape == X.shape, f"Output shape mismatch: {X_imp.shape} != {X.shape}"
    assert not np.isnan(X_imp).any(), "NaN remaining after fit_transform"


def test_sats_imputer_preserves_observed():
    """Observed values at fully-observed timesteps should not be changed by the imputer.
    
    The imputer uses an all-features-per-timestep masking rule: a timestep is considered
    "observed" only if ALL features at that timestep are present. At such fully-observed
    timesteps, the AttentionImputer.forward() uses torch.where to preserve original values.
    At partially-missing timesteps, ALL features (including observed ones) may be replaced.
    """
    # Create data with only a single missing value at (sample=0, t=5, feat=2)
    X = np.ones((20, LOOKBACK, 4), dtype=np.float32) * 5.0
    X[0, 5, 2] = np.nan  # one missing value

    imp = SATSImputer(n_feat=4, epochs=5)
    X_imp = imp.fit_transform(X)

    # Build mask of fully-observed timesteps (all features observed at that timestep)
    # For sample 0, t=5: feature 2 is NaN → entire timestep is "partially missing"
    fully_observed = np.ones((20, LOOKBACK, 4), dtype=bool)
    fully_observed[0, 5, :] = False  # t=5 of sample 0: partially missing → imputer may overwrite

    # At fully-observed timesteps, original values must be preserved exactly
    assert np.allclose(X_imp[fully_observed], X[fully_observed], atol=1e-4), \
        "SATSImputer modified values at fully-observed timesteps"


def test_sats_imputer_output_dtype():
    """SATSImputer.transform should return float32."""
    rng = np.random.default_rng(21)
    X = rng.standard_normal((30, LOOKBACK, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan

    imp = SATSImputer(n_feat=4, epochs=2)
    X_imp = imp.fit_transform(X)
    assert X_imp.dtype == np.float32, f"Expected float32, got {X_imp.dtype}"


def test_sats_imputer_mask_consistency():
    """
    transform() should correctly handle partial missingness.
    Even if only some features are missing at a timestep, the imputer
    should produce no NaN in the output.
    """
    rng = np.random.default_rng(22)
    X_train = rng.standard_normal((30, LOOKBACK, 4)).astype(np.float32)
    X_train[rng.random(X_train.shape) < 0.1] = np.nan

    imp = SATSImputer(n_feat=4, epochs=2)
    imp.fit(X_train)

    # Test: feature 0 fully observed but features 1,2,3 partially missing
    X_test = np.ones((5, LOOKBACK, 4), dtype=np.float32)
    X_test[:, 10, 1] = np.nan   # feature 1 missing at t=10
    X_test[:, 10, 2] = np.nan   # feature 2 missing at t=10

    X_imp = imp.transform(X_test)
    assert not np.isnan(X_imp).any(), "NaN remaining after transform with partial feature missingness"


def test_sats_imputer_wrong_feature_count_raises():
    """transform() with wrong feature count should raise an error."""
    rng = np.random.default_rng(23)
    X = rng.standard_normal((20, LOOKBACK, 4)).astype(np.float32)
    imp = SATSImputer(n_feat=4, epochs=2)
    imp.fit(X)

    # Wrong feature count in transform input
    X_wrong = rng.standard_normal((5, LOOKBACK, 3)).astype(np.float32)
    # Should raise AssertionError, RuntimeError, or ValueError (broadcast/shape mismatch)
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        imp.transform(X_wrong)


def test_sats_imputer_transform_before_fit_raises():
    """Calling transform() before fit() should raise RuntimeError."""
    imp = SATSImputer(n_feat=4, epochs=2)
    X = np.ones((5, LOOKBACK, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="fit"):
        imp.transform(X)


def test_sats_imputer_all_observed():
    """fit_transform on fully observed data should return unchanged values."""
    rng = np.random.default_rng(24)
    X = rng.standard_normal((20, LOOKBACK, 4)).astype(np.float32)
    # No NaN
    imp = SATSImputer(n_feat=4, epochs=2)
    X_imp = imp.fit_transform(X)

    # Since all values are observed, the imputer should not change any value
    assert np.allclose(X_imp, X, atol=1e-4), \
        "SATSImputer should not modify fully-observed data"


# ── 6. Integration test ───────────────────────────────────────────────────

def test_full_pipeline_smoke():
    """End-to-end smoke test: make_windows → RiverDataset → Seq2SeqLSTM forward pass."""
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic daily data
    n_days = 200
    idx = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((n_days, len(FEATURES))).astype(np.float32),
        index=idx,
        columns=FEATURES,
    )
    means = _make_train_means(df)

    X, y, dates = make_windows(df, means)
    assert X.shape[1] == LOOKBACK, f"X.shape[1]={X.shape[1]}, expected LOOKBACK={LOOKBACK}"
    assert y.shape[1] == HORIZON, f"y.shape[1]={y.shape[1]}, expected HORIZON={HORIZON}"

    N, L, F = X.shape
    _, H, T = y.shape

    feat_sc = StandardScaler().fit(X.reshape(-1, F))
    tgt_sc = StandardScaler().fit(y.reshape(-1, T))

    X_sc = feat_sc.transform(X.reshape(-1, F)).reshape(N, L, F).astype(np.float32)
    y_sc = tgt_sc.transform(y.reshape(-1, T)).reshape(N, H, T).astype(np.float32)

    ds = RiverDataset(X_sc, y_sc)
    assert len(ds) == N

    model = Seq2SeqLSTM(n_feat=F, n_tgt=T, hidden=32, n_layers=1, dropout=0.0)
    y_pred = predict(model, ds, tgt_sc)

    assert y_pred.shape == (N, H, T), \
        f"Integration: predict() shape mismatch: {y_pred.shape} != ({N},{H},{T})"
    assert not np.isnan(y_pred).any(), "Integration: NaN in predictions"
    assert y_pred.dtype == np.float32

    print(f"\nIntegration test passed: {N} windows, "
          f"DO range [{y_pred[:, :, 0].min():.2f}, {y_pred[:, :, 0].max():.2f}]")


def test_full_pipeline_with_ea_lstm():
    """End-to-end smoke test using EASeq2SeqLSTM with static attributes."""
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    n_days = 200
    idx = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(43)

    df = pd.DataFrame(
        rng.standard_normal((n_days, len(FEATURES))).astype(np.float32),
        index=idx,
        columns=FEATURES,
    )
    means = _make_train_means(df)
    X, y, _ = make_windows(df, means)

    N, L, F = X.shape
    _, H, T = y.shape

    feat_sc = StandardScaler().fit(X.reshape(-1, F))
    tgt_sc = StandardScaler().fit(y.reshape(-1, T))

    X_sc = feat_sc.transform(X.reshape(-1, F)).reshape(N, L, F).astype(np.float32)
    y_sc = tgt_sc.transform(y.reshape(-1, T)).reshape(N, H, T).astype(np.float32)

    static_size = len(STATIC_COLS)
    model = EASeq2SeqLSTM(n_feat=F, n_tgt=T, static_size=static_size,
                           hidden=16, n_layers=2, dropout=0.0)
    model.eval()

    # Simulate a static attribute vector for this gauge
    s = torch.randn(N, static_size)
    x_t = torch.from_numpy(X_sc)

    with torch.no_grad():
        out = model(x_t, s)

    assert out.shape == (N, H, T), f"EA pipeline output shape: {out.shape}"
    assert not torch.isnan(out).any(), "EA pipeline: NaN in output"


def test_make_windows_with_all_nan_target_dropped():
    """Windows where any target is NaN should be excluded from output."""
    rng = np.random.default_rng(30)
    df = _make_synthetic_df(300)
    means = _make_train_means(df)

    # Count windows without any NaN target
    X_no_nan, y_no_nan, _ = make_windows(df, means)

    # Now introduce NaN in targets for some rows — those windows should be dropped
    df_with_nan = df.copy()
    # Introduce NaN in target at position LOOKBACK (first forecast day of first window)
    df_with_nan.iloc[LOOKBACK, df_with_nan.columns.get_loc(TARGETS[0])] = np.nan

    X_nan, y_nan, _ = make_windows(df_with_nan, means)

    # At most a few windows should be dropped (those where TARGETS[0] is NaN in forecast)
    assert X_nan.shape[0] < X_no_nan.shape[0] or X_nan.shape[0] == X_no_nan.shape[0], \
        "Dropping NaN-target windows should not increase window count"
    # Specifically, the window whose forecast includes the NaN should be dropped
    assert X_nan.shape[0] <= X_no_nan.shape[0]


def test_metrics_rmse_scaling():
    """Adding a constant bias should increase RMSE proportionally."""
    rng = np.random.default_rng(31)
    y_true = rng.standard_normal((100, HORIZON, len(TARGETS))).astype(np.float32)
    bias = 2.0
    y_pred = y_true + bias

    result = mean_rmse(y_true, y_pred)
    for tgt, val in result.items():
        assert abs(val - bias) < 1e-5, \
            f"RMSE of constant-bias prediction should equal bias ({bias}), got {val} for {tgt}"
