#!/bin/bash
# =============================================================================
# AareML — Run All Notebooks in Order
# Clears all outputs first, then executes each notebook sequentially.
# Generates a summary log at results/notebook_run_log.txt
#
# Usage:
#   bash run_all_notebooks.sh           # run all notebooks
#   bash run_all_notebooks.sh --dry-run # just clear outputs, don't execute
# =============================================================================

set -e

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

NOTEBOOKS=(
    "notebooks/01_data_exploration.ipynb"
    "notebooks/02_baselines.ipynb"
    "notebooks/03_lstm_single_site.ipynb"
    "notebooks/04_multisite_analysis.ipynb"
    "notebooks/04b_multisite_temperature.ipynb"
    "notebooks/05_shap_interpretation.ipynb"
    "notebooks/06_cross_ecosystem_lake.ipynb"
    "notebooks/07_lake_eda.ipynb"
)

LOG="results/notebook_run_log.txt"
mkdir -p results

echo "=============================================" | tee "$LOG"
echo "  AareML — Notebook Run Log" | tee -a "$LOG"
echo "  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "  Dry run: $DRY_RUN" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

TOTAL=0
PASSED=0
FAILED=0
FAILED_LIST=()

for NB in "${NOTEBOOKS[@]}"; do

    NAME=$(basename "$NB")
    echo "─────────────────────────────────────────────" | tee -a "$LOG"
    echo "  $NAME" | tee -a "$LOG"

    # Step 1: Clear outputs
    echo -n "  [1/2] Clearing outputs... " | tee -a "$LOG"
    jupyter nbconvert --clear-output --inplace "$NB" 2>/dev/null
    echo "done" | tee -a "$LOG"

    if $DRY_RUN; then
        echo "  [2/2] Skipped (dry run)" | tee -a "$LOG"
        echo "" | tee -a "$LOG"
        continue
    fi

    # Step 2: Execute
    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    echo -n "  [2/2] Executing (started $START_TS)... " | tee -a "$LOG"
    START=$(date +%s)

    if jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=7200 \
        --ExecutePreprocessor.kernel_name=aareml \
        "$NB" 2>"/tmp/${NAME%.ipynb}_err.txt"; then

        END=$(date +%s)
        END_TS=$(date '+%H:%M:%S')
        ELAPSED=$((END - START))
        MINS=$((ELAPSED / 60))
        SECS=$((ELAPSED % 60))
        echo "✓ ${MINS}m ${SECS}s (finished $END_TS)" | tee -a "$LOG"
        PASSED=$((PASSED + 1))
    else
        END=$(date +%s)
        END_TS=$(date '+%H:%M:%S')
        ELAPSED=$((END - START))
        echo "✗ FAILED after ${ELAPSED}s (finished $END_TS)" | tee -a "$LOG"
        echo "  Error log: /tmp/${NAME%.ipynb}_err.txt" | tee -a "$LOG"
        # Print last 10 lines of error
        echo "  Last error lines:" | tee -a "$LOG"
        tail -10 "/tmp/${NAME%.ipynb}_err.txt" | sed 's/^/    /' | tee -a "$LOG"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$NAME")
    fi

    TOTAL=$((TOTAL + 1))
    echo "" | tee -a "$LOG"

done

# Summary
echo "=============================================" | tee -a "$LOG"
echo "  SUMMARY" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
echo "  Total notebooks : $TOTAL" | tee -a "$LOG"
echo "  Passed          : $PASSED" | tee -a "$LOG"
echo "  Failed          : $FAILED" | tee -a "$LOG"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "  Failed notebooks:" | tee -a "$LOG"
    for nb in "${FAILED_LIST[@]}"; do
        echo "    - $nb" | tee -a "$LOG"
    done
fi
echo "" | tee -a "$LOG"
echo "  Log saved to: $LOG" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
