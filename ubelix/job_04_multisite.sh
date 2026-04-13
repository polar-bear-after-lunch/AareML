#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 04 (Multi-Site Evaluation)
#
# IMPORTANT: Run AFTER job_03_lstm.sh completes — needs the saved checkpoint.
#
# Estimated runtime: ~30 min on RTX 3090 (16 gauges × zero-shot + retrain)
# GPU memory needed: ~6 GB
#
# Submit with: sbatch ubelix/job_04_multisite.sh
# Or chain after job 03: sbatch --dependency=afterok:<JOB_03_ID> ubelix/job_04_multisite.sh
# =============================================================================

#SBATCH --job-name="aareml_04_multisite"
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G            # more RAM for loading 16 gauges
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch   # ← replace with your email
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/job_04_multisite_%j.out
#SBATCH --error=logs/job_04_multisite_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd $SLURM_SUBMIT_DIR

# Check that notebook 03 checkpoint exists
if [ ! -f "results/best_model.pt" ]; then
    echo "ERROR: results/best_model.pt not found."
    echo "Please run job_03_lstm.sh first."
    exit 1
fi

# ── Run notebook 04 ──────────────────────────────────────────────────────────
echo ""
echo "Running notebook 04: Multi-Site Evaluation..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/04_multisite_analysis.ipynb

echo ""
echo "Notebook 04 complete."
echo "Job finished: $(date)"
