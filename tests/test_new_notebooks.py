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
    # Column may be named 'RMSE' or 'DO RMSE'
    rmse_col = "RMSE" if "RMSE" in df.columns else "DO RMSE"
    assert rmse_col in df.columns, f"No RMSE column found; columns: {list(df.columns)}"
    ar_row = df[df["Model"].str.contains("AR", case=False)]
    assert len(ar_row) > 0, "No AR row found"
    assert ar_row[rmse_col].values[0] < 0.5, "AR RMSE unexpectedly high"

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


def test_nb17_nh_results():
    """nb17 NeuralHydrology EA-LSTM: test metrics CSV exists and has valid results."""
    import glob
    nh_runs = sorted(glob.glob(str(RESULTS / "nh_run/aareml_ealstm_*/test/model_epoch030/test_metrics.csv")))
    assert len(nh_runs) > 0, "No NeuralHydrology test_metrics.csv found"
    df = pd.read_csv(nh_runs[-1])
    assert "NSE" in df.columns and "RMSE" in df.columns
    valid = df.dropna(subset=["NSE", "RMSE"])
    assert len(valid) >= 8, f"Expected ≥8 valid gauges, got {len(valid)}"
    assert valid["RMSE"].mean() < 0.8, f"Mean RMSE too high: {valid['RMSE'].mean():.3f}"
    assert valid["NSE"].mean() > 0.5, f"Mean NSE too low: {valid['NSE'].mean():.3f}"


def test_nb18_cascaded_results():
    """nb18 cascaded DO: results CSV exists with correct structure."""
    f = RESULTS / "cascaded_do_results.csv"
    assert f.exists(), "cascaded_do_results.csv missing"
    df = pd.read_csv(f)
    assert "setup" in df.columns or "Setup" in df.columns or "model" in df.columns
    # Linear baseline should be worse than standard LSTM
    rmse_vals = df["RMSE"].values if "RMSE" in df.columns else df.iloc[:,1].values
    assert min(rmse_vals) > 0.25, "Cascaded RMSE suspiciously low"
    assert max(rmse_vals) < 1.5, "Cascaded RMSE suspiciously high"


def test_significance_tests_exist():
    """Wilcoxon test results from nb04 should be saved."""
    f = RESULTS / "significance_tests.json"
    assert f.exists(), "significance_tests.json missing"
    import json
    data = json.loads(f.read_text())
    assert "lstm_zeroshot_vs_ridge" in data
    assert data["lstm_zeroshot_vs_ridge"]["significant"] == True
    assert data["lstm_zeroshot_vs_ridge"]["p"] < 0.05
