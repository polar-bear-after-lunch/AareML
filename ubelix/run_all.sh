#!/bin/bash
# =============================================================================
# AareML — UBELIX Master Submit Script
#
# Submits ALL jobs with proper dependencies:
#
#   01 (EDA) → 02 (Baselines) → 03 (LSTM) → 04 (Multi-site DO)
#                                          → 04b (Temp multi-site)
#                                          → 05 (SHAP)
#                                          → 06 (Lake Mendota)
#                                          → 07 (Lake EDA)
#                                          → 08 (USGS transfer)
#                                          → 09 (Canton Zurich)
#                                          → 10 (Swiss Lakes)
#
# GPU jobs (03, 04, 04b, 05, 08, 10): gpu-invest partition, RTX 4090
# CPU jobs (01, 02, 06, 07, 09):      epyc2 partition
#
# Usage:
#   bash ubelix/run_all.sh
#
# Monitor: squeue --me
# =============================================================================

set -e

cd /storage/homefs/tn20y076/AareML
mkdir -p results logs

echo "============================================="
echo "  AareML — Submitting ALL UBELIX Jobs"
echo "  Working directory: $(pwd)"
echo "  $(date)"
echo "============================================="

# ── Job 01: EDA (CPU, no dependencies) ───────────────────────────────────────
echo ""
echo "Submitting job 01 (EDA)..."
JOB_01=$(sbatch --parsable ubelix/job_01_eda.sh)
echo "  → Job ID: $JOB_01"

# ── Job 02: Baselines (CPU, after 01) ────────────────────────────────────────
echo ""
echo "Submitting job 02 (Baselines, depends on $JOB_01)..."
JOB_02=$(sbatch --parsable --dependency=afterok:$JOB_01 ubelix/job_02_baselines.sh)
echo "  → Job ID: $JOB_02"

# ── Job 03: LSTM + Optuna (GPU, after 02) ────────────────────────────────────
echo ""
echo "Submitting job 03 (LSTM + Optuna, depends on $JOB_02)..."
JOB_03=$(sbatch --parsable --dependency=afterok:$JOB_02 ubelix/job_03_lstm.sh)
echo "  → Job ID: $JOB_03"

# ── Job 04: Multi-site DO (GPU, after 03) ────────────────────────────────────
echo ""
echo "Submitting job 04 (Multi-site DO, depends on $JOB_03)..."
JOB_04=$(sbatch --parsable --dependency=afterok:$JOB_03 ubelix/job_04_multisite.sh)
echo "  → Job ID: $JOB_04"

# ── Job 04b: Temp multi-site (GPU, after 04) ─────────────────────────────────
echo ""
echo "Submitting job 04b (Temp multi-site, depends on $JOB_04)..."
JOB_04B=$(sbatch --parsable --dependency=afterok:$JOB_04 ubelix/job_04b_temp.sh)
echo "  → Job ID: $JOB_04B"

# ── Job 05: SHAP (GPU, after 04b) ────────────────────────────────────────────
echo ""
echo "Submitting job 05 (SHAP, depends on $JOB_04B)..."
JOB_05=$(sbatch --parsable --dependency=afterok:$JOB_04B ubelix/job_05_shap.sh)
echo "  → Job ID: $JOB_05"

# ── Job 06: Lake Mendota (CPU, after 05) ─────────────────────────────────────
echo ""
echo "Submitting job 06 (Lake Mendota, depends on $JOB_05)..."
JOB_06=$(sbatch --parsable --dependency=afterok:$JOB_05 ubelix/job_06_lake_mendota.sh)
echo "  → Job ID: $JOB_06"

# ── Job 07: Lake EDA (CPU, after 06) ─────────────────────────────────────────
echo ""
echo "Submitting job 07 (Lake EDA, depends on $JOB_06)..."
JOB_07=$(sbatch --parsable --dependency=afterok:$JOB_06 ubelix/job_07_lake_eda.sh)
echo "  → Job ID: $JOB_07"

# ── Job 08: USGS transfer (GPU, after 07) ────────────────────────────────────
echo ""
echo "Submitting job 08 (USGS transfer, depends on $JOB_07)..."
JOB_08=$(sbatch --parsable --dependency=afterok:$JOB_07 ubelix/job_08_usgs.sh)
echo "  → Job ID: $JOB_08"

# ── Job 09: Canton Zurich (CPU, after 08) ────────────────────────────────────
echo ""
echo "Submitting job 09 (Canton Zurich, depends on $JOB_08)..."
JOB_09=$(sbatch --parsable --dependency=afterok:$JOB_08 ubelix/job_09_canton_zurich.sh)
echo "  → Job ID: $JOB_09"

# ── Job 10: Swiss Lakes (GPU, after 09) ──────────────────────────────────────
echo ""
echo "Submitting job 10 (Swiss Lakes, depends on $JOB_09)..."
JOB_10=$(sbatch --parsable --dependency=afterok:$JOB_09 ubelix/job_10_lakes.sh)
echo "  → Job ID: $JOB_10"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  All 11 jobs submitted"
echo "============================================="
echo ""
echo "  Job 01  (EDA):             $JOB_01   [CPU]"
echo "  Job 02  (Baselines):       $JOB_02   [CPU, after $JOB_01]"
echo "  Job 03  (LSTM+Optuna):     $JOB_03   [GPU, after $JOB_02]"
echo "  Job 04  (Multi-site DO):   $JOB_04   [GPU, after $JOB_03]"
echo "  Job 04b (Temp multi-site): $JOB_04B  [GPU, after $JOB_04]"
echo "  Job 05  (SHAP):            $JOB_05   [GPU, after $JOB_04B]"
echo "  Job 06  (Lake Mendota):    $JOB_06   [CPU, after $JOB_05]"
echo "  Job 07  (Lake EDA):        $JOB_07   [CPU, after $JOB_06]"
echo "  Job 08  (USGS transfer):   $JOB_08   [GPU, after $JOB_07]"
echo "  Job 09  (Canton Zurich):   $JOB_09   [CPU, after $JOB_08]"
echo "  Job 10  (Swiss Lakes):     $JOB_10   [GPU, after $JOB_09]"
echo ""
echo "  Monitor:  squeue --me"
echo "  Cancel:   scancel $JOB_01 $JOB_02 $JOB_03 $JOB_04 $JOB_04B $JOB_05 $JOB_06 $JOB_07 $JOB_08 $JOB_09 $JOB_10"
echo "  Logs:     tail -f logs/job_03_lstm_${JOB_03}.out"
echo ""
echo "  Expected total runtime: ~8-10 hours"
echo "  (nb03 Optuna is the bottleneck: ~3-4 hours)"
