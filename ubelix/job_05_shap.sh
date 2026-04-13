#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 05 (SHAP Attribution)
#
# IMPORTANT: Run AFTER job_03_lstm.sh completes — needs the saved checkpoint.
# Can run in parallel with job_04_multisite.sh.
#
# Estimated runtime: ~20 min on RTX 3090
# GPU memory needed: ~4 GB
#
# Submit with: sbatch ubelix/job_05_shap.sh
# Or chain after job 03: sbatch --dependency=afterok:<JOB_03_ID> ubelix/job_05_shap.sh
# =============================================================================

#SBATCH --job-name="aareml_05_shap"
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --qos=job_gpu
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch   # ← replace with your email
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/job_05_shap_%j.out
#SBATCH --error=logs/job_05_shap_%j.err

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

# Check checkpoint
if [ ! -f "results/best_model.pt" ]; then
    echo "ERROR: results/best_model.pt not found."
    echo "Please run job_03_lstm.sh first."
    exit 1
fi

# ── Run notebook 05 ──────────────────────────────────────────────────────────
echo ""
echo "Running notebook 05: SHAP Attribution..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=5400 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/05_shap_interpretation.ipynb

echo ""
echo "Notebook 05 complete."
echo "Job finished: $(date)"
