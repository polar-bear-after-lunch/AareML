#!/bin/bash
# ── AareML: Submit all notebooks sequentially to UBELIX ──────────────────────
# Usage: bash ubelix/run_all_notebooks.sh
# Each job auto-commits results to GitHub when done (auto-push in each job script).
# Jobs are submitted sequentially with --dependency=afterok so they run in order.
# This avoids concurrent writes to shared files (scalers, results, etc).

set -euo pipefail
cd /storage/homefs/tn20y076/AareML

echo "============================================================"
echo "  AareML — Full notebook rerun"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

JOBS=(
    "job_01_eda.sh"
    "job_02_baselines.sh"
    "job_03_lstm.sh"
    "job_04_multisite.sh"
    "job_04b_temp.sh"
    "job_04c_temp_precip.sh"
    "job_05_shap.sh"
    "job_06_lake_mendota.sh"
    "job_07_lake_eda.sh"
    "job_08_usgs.sh"
    "job_09_canton_zurich.sh"
    "job_10_lakes.sh"
    "job_11_ablation.sh"
    "job_12_error_analysis.sh"
    "job_13_seasonal.sh"
    "job_14_ar_baseline.sh"
    "job_15_rigor.sh"
    "job_16_cross_val.sh"
    "job_17_neuralhydrology.sh"
    "job_18_cascaded.sh"
)

PREV_JOB_ID=""
SUBMITTED=0

for JOB in "${JOBS[@]}"; do
    SCRIPT="ubelix/$JOB"
    if [ ! -f "$SCRIPT" ]; then
        echo "  [SKIP] $JOB — script not found"
        continue
    fi

    if [ -z "$PREV_JOB_ID" ]; then
        # Submit first job immediately
        JOB_ID=$(sbatch --parsable "$SCRIPT")
    else
        # Each subsequent job waits for the previous to complete successfully
        JOB_ID=$(sbatch --parsable --dependency=afterok:${PREV_JOB_ID} "$SCRIPT")
    fi

    TIMESTAMP=$(date '+%H:%M:%S')
    echo "  [$TIMESTAMP] Submitted $JOB → job ID $JOB_ID"
    PREV_JOB_ID=$JOB_ID
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================================"
echo "  $SUBMITTED jobs submitted."
echo "  Monitor: squeue -u tn20y076"
echo "  Results will auto-push to GitHub as each job completes."
echo "  Final job ID: $PREV_JOB_ID"
echo "============================================================"
