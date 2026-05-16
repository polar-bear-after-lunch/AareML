"""
Tests for new AareML notebooks (nb11-nb16).
Checks that result CSVs exist and contain valid data.
"""
import pytest
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

def test_ablation_results_exist():
    f = RESULTS / "ablation_results.csv"
    assert f.exists(), "ablation_results.csv missing"
    df = pd.read_csv(f)
    assert len(df) >= 8, "Expected at least 8 ablation rows"
    assert "RMSE_DO" in df.columns

def test_ar_baseline_results():
    f = RESULTS / "ar_baseline_results.csv"
    assert f.exists(), "ar_baseline_results.csv missing"
    df = pd.read_csv(f)
    assert "RMSE" in df.columns
    ar_row = df[df["Model"].str.contains("AR", case=False)]
    assert len(ar_row) > 0, "No AR row found"
    assert ar_row["RMSE"].values[0] < 0.6, "AR RMSE unexpectedly high"

def test_cv_transfer_results():
    f = RESULTS / "cv_transfer_results.csv"
    assert f.exists(), "cv_transfer_results.csv missing"
    df = pd.read_csv(f)
    assert "source_gauge" in df.columns
    assert "rmse_do" in df.columns
    assert len(df) >= 50, f"Expected >=50 pairs, got {len(df)}"
    mean_rmse = df["rmse_do"].mean()
    assert 0.3 < mean_rmse < 1.0, f"Mean RMSE {mean_rmse:.3f} outside expected range"

def test_ridge_transfer_results():
    f = RESULTS / "ridge_transfer_results.csv"
    assert f.exists(), "ridge_transfer_results.csv missing"
    df = pd.read_csv(f)
    assert "rmse_do" in df.columns
    mean = df["rmse_do"].mean()
    assert mean > 0.4, "Ridge zero-shot RMSE suspiciously low"
    assert mean < 1.0, "Ridge zero-shot RMSE suspiciously high"

def test_ea_lstm_results_updated():
    f = RESULTS / "ea_lstm_results.csv"
    assert f.exists(), "ea_lstm_results.csv missing"
    df = pd.read_csv(f)
    assert "rmse_do" in df.columns
    mean = df["rmse_do"].mean()
    assert 0.3 < mean < 0.6, f"EA-LSTM mean RMSE {mean:.3f} outside expected range"

def test_temp_multisite_results():
    f = RESULTS / "temp_multisite_combined.csv"
    assert f.exists(), "temp_multisite_combined.csv missing"
    df = pd.read_csv(f)
    ea = df[df["strategy"] == "ea_lstm_temp"] if "strategy" in df.columns else df
    assert len(ea) > 0
