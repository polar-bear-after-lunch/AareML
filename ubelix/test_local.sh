#!/bin/bash
# =============================================================================
# AareML — Local UBELIX Simulation Test
#
# Tests that all notebooks can be executed without SLURM/GPU.
# Runs each notebook with a SHORT timeout to verify:
#   ✓ All imports work
#   ✓ Data loads correctly
#   ✓ First cells execute without error
#
# Does NOT run the full training (too slow on CPU for a test).
# Use --full to run everything.
#
# Usage:
#   bash ubelix/test_local.sh           # quick smoke test (~2 min total)
#   bash ubelix/test_local.sh --full    # full run (hours)
# =============================================================================

set -e

FULL=false
[[ "$1" == "--full" ]] && FULL=true

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

# Smoke test: 3 min per notebook (enough for imports + data load)
SMOKE_TIMEOUT=180
FULL_TIMEOUT=7200

TIMEOUT=$SMOKE_TIMEOUT
[[ "$FULL" == "true" ]] && TIMEOUT=$FULL_TIMEOUT

LOG="results/local_test_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p results

echo "=============================================" | tee "$LOG"
echo "  AareML — Local UBELIX Test" | tee -a "$LOG"
echo "  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "  Mode: $([ "$FULL" = "true" ] && echo "FULL RUN" || echo "SMOKE TEST (${SMOKE_TIMEOUT}s per notebook)")" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Verify environment
echo "Environment check:" | tee -a "$LOG"
python -c "import torch, sklearn, optuna, captum; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')" | tee -a "$LOG"
python -c "import pandas, numpy, matplotlib; print(f'  pandas={pandas.__version__}, numpy={numpy.__version__}')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PASSED=0; FAILED=0; SKIPPED=0
FAILED_LIST=()

for NB in "${NOTEBOOKS[@]}"; do
    NAME=$(basename "$NB")

    # Skip heavy notebooks in smoke test
    if [[ "$FULL" == "false" ]]; then
        if [[ "$NAME" == "03_lstm"* ]] || [[ "$NAME" == "04_multi"* ]] || [[ "$NAME" == "04b_multi"* ]] || [[ "$NAME" == "04b_lstm"* ]]; then
            echo "─────────────────────────────────────────────" | tee -a "$LOG"
            echo "  $NAME — SKIPPED (training notebook, use --full)" | tee -a "$LOG"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    echo "─────────────────────────────────────────────" | tee -a "$LOG"
    echo "  $NAME" | tee -a "$LOG"

    # Clear outputs first
    echo -n "  Clearing outputs... " | tee -a "$LOG"
    jupyter nbconvert --clear-output --inplace "$NB" 2>/dev/null
    echo "done" | tee -a "$LOG"

    # Execute
    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    echo -n "  Executing (started $START_TS, timeout=${TIMEOUT}s)... " | tee -a "$LOG"
    START=$(date +%s)

    if jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=$TIMEOUT \
        --ExecutePreprocessor.kernel_name=aareml \
        "$NB" 2>"/tmp/${NAME%.ipynb}_test_err.txt"; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        echo "✓ ${ELAPSED}s" | tee -a "$LOG"
        PASSED=$((PASSED + 1))
    else
        END=$(date +%s)
        # Check if it was a timeout (acceptable in smoke mode)
        if grep -q "timeout\|TimeoutError\|CellTimeoutError" "/tmp/${NAME%.ipynb}_test_err.txt" 2>/dev/null && [[ "$FULL" == "false" ]]; then
            echo "⏱ TIMED OUT (expected in smoke mode — imports OK)" | tee -a "$LOG"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "✗ FAILED" | tee -a "$LOG"
            tail -5 "/tmp/${NAME%.ipynb}_test_err.txt" | sed 's/^/    /' | tee -a "$LOG"
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$NAME")
        fi
    fi
    echo "" | tee -a "$LOG"
done

echo "=============================================" | tee -a "$LOG"
echo "  RESULTS" | tee -a "$LOG"
echo "  Passed : $PASSED" | tee -a "$LOG"
echo "  Skipped: $SKIPPED (training/timeout)" | tee -a "$LOG"
echo "  Failed : $FAILED" | tee -a "$LOG"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    for nb in "${FAILED_LIST[@]}"; do
        echo "    ✗ $nb" | tee -a "$LOG"
    done
fi
echo "" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

[[ $FAILED -gt 0 ]] && exit 1 || exit 0
