#!/bin/bash
# =============================================================================
# AareML — UBELIX Master Script
#
# Submits all three jobs with proper dependencies:
#   Job 03 (LSTM) runs first
#   Jobs 04 (multi-site) and 05 (SHAP) run in parallel after 03 finishes
#
# Usage:
#   bash ubelix/run_all.sh
#
# After running, monitor with:
#   squeue --me
# =============================================================================

set -e

echo "============================================="
echo "  AareML — Submitting UBELIX Jobs"
echo "============================================="

mkdir -p logs

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

# ── Job 05: SHAP (runs after 03 succeeds, parallel with 04) ───────────────────
echo ""
echo "Submitting job 05 (SHAP attribution, depends on job $JOB_03)..."
JOB_05=$(sbatch --parsable --dependency=afterok:$JOB_03 ubelix/job_05_shap.sh)
echo "  → Job ID: $JOB_05"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  All jobs submitted"
echo "============================================="
echo ""
echo "  Job 03 (LSTM):       $JOB_03"
echo "  Job 04 (Multi-site): $JOB_04  [waits for $JOB_03]"
echo "  Job 05 (SHAP):       $JOB_05  [waits for $JOB_03]"
echo ""
echo "  Monitor:   squeue --me"
echo "  Cancel:    scancel $JOB_03 $JOB_04 $JOB_05"
echo "  Logs:      tail -f logs/job_03_lstm_${JOB_03}.out"
echo ""
echo "You will receive an email when each job finishes."
echo "Total expected time: ~20 min (job 03) + ~30 min (jobs 04+05 in parallel)"
