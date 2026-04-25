#!/bin/bash
# =============================================================================
# AareML — UBELIX Master Script
#
# Submits all jobs with proper dependencies (sequential, 1 GPU at a time):
#   03 (LSTM) → 04 (multi-site DO) → 04b (temp multi-site) → 05 (SHAP)
#
# Usage:
#   bash ubelix/run_all.sh
#
# After running, monitor with:
#   squeue --me
# =============================================================================

set -e

# Always run from the AareML root
cd /storage/homefs/tn20y076/AareML

echo "============================================="
echo "  AareML — Submitting UBELIX Jobs"
echo "  Working directory: $(pwd)"
echo "============================================="

mkdir -p results logs

# ── Job 03: LSTM + Optuna (runs first) ───────────────────────────────────────
echo ""
echo "Submitting job 03 (LSTM single-site + Optuna)..."
JOB_03=$(sbatch --parsable ubelix/job_03_lstm.sh)
echo "  → Job ID: $JOB_03"

# ── Job 04: Multi-site (runs after 03 succeeds) ───────────────────────────────
echo ""
echo "Submitting job 04 (multi-site evaluation, depends on job $JOB_03)..."
JOB_04=$(sbatch --parsable --dependency=afterok:$JOB_03 ubelix/job_04_multisite.sh)
echo "  → Job ID: $JOB_04"

# ── Job 04b: Temp multi-site (runs after 04 succeeds) ────────────────────────
echo ""
echo "Submitting job 04b (temperature multi-site, depends on job $JOB_04)..."
JOB_04B=$(sbatch --parsable --dependency=afterok:$JOB_04 ubelix/job_04b_temp.sh)
echo "  → Job ID: $JOB_04B"

# ── Job 05: SHAP (runs after 04b succeeds) ────────────────────────────────────
echo ""
echo "Submitting job 05 (SHAP attribution, depends on job $JOB_04B)..."
JOB_05=$(sbatch --parsable --dependency=afterok:$JOB_04B ubelix/job_05_shap.sh)
echo "  → Job ID: $JOB_05"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  All jobs submitted"
echo "============================================="
echo ""
echo "  Job 03  (LSTM):            $JOB_03"
echo "  Job 04  (Multi-site DO):   $JOB_04   [waits for $JOB_03]"
echo "  Job 04b (Temp multi-site): $JOB_04B  [waits for $JOB_04]"
echo "  Job 05  (SHAP):            $JOB_05   [waits for $JOB_04B]"
echo ""
echo "  Monitor:   squeue --me"
echo "  Cancel:    scancel $JOB_03 $JOB_04 $JOB_04B $JOB_05"
echo "  Logs:      tail -f logs/job_03_lstm_${JOB_03}.out"
echo ""
echo "You will receive an email when each job finishes."
echo "Total expected time: ~20 min (03) + ~30 min (04) + ~2h (04b) + ~20 min (05)"
