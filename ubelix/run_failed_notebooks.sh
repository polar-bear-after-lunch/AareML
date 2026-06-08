#!/bin/bash
# ── AareML: Rerun only the failed notebooks ──────────────────────────────────
# Usage: bash ubelix/run_failed_notebooks.sh
# Submits nb06, nb07, nb14, nb15, nb16, nb17, nb18 sequentially.

set -euo pipefail
cd /storage/homefs/tn20y076/AareML

# Guard: refuse if AareML jobs already queued
if squeue --me --noheader 2>/dev/null | grep -q "aareml"; then
    echo "ERROR: AareML jobs already in queue. Run: scancel --me first."
    squeue --me
    exit 1
fi

echo "============================================================"
echo "  AareML — Partial rerun (failed notebooks)"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

JOBS=(
    "job_06_lake_mendota.sh"
    "job_07_lake_eda.sh"
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
        JOB_ID=$(sbatch --parsable "$SCRIPT")
    else
        JOB_ID=$(sbatch --parsable --dependency=afterok:${PREV_JOB_ID} "$SCRIPT")
    fi
    echo "  [$(date '+%H:%M:%S')] Submitted $JOB → job ID $JOB_ID"
    PREV_JOB_ID=$JOB_ID
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================================"
echo "  $SUBMITTED jobs submitted. Monitor: squeue -u tn20y076"
echo "  Then run: bash ubelix/run_all_notebooks.sh for full rerun"
echo "============================================================"
