#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 16 (Cross-Validation: Zero-Shot Transfer)
#
# Leave-one-out cross-validation across all 16 DO gauges.
# Trains LSTM on each gauge in turn, zero-shot transfers to all others.
# 16 source gauges × 50 epochs each ≈ 7–8 h on RTX 4090.
#
# Runs independently — no dependency on other jobs.
#
# Submit with: sbatch ubelix/job_16_cross_val.sh
# Monitor with: squeue --me
# View output: tail -f logs/job_16_cross_val_<JOBID>.out
# =============================================================================

#SBATCH --job-name="aareml_16_cross_val"
#SBATCH --time=04:00:00              # 4h — reduced epochs for speed
#SBATCH --mem-per-cpu=16G            # 16 GB RAM (more gauges loaded at once)
#SBATCH --cpus-per-task=4            # 4 CPU cores for data loading
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue            # do not auto-restart if preempted
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch   # ← replace with your email
#SBATCH --mail-type=END,FAIL        # notify on completion or failure
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_16_cross_val_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_16_cross_val_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Load modules and activate environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml

# Move to project root
cd /storage/homefs/tn20y076/AareML

# ── Run notebook 16 ──────────────────────────────────────────────────────────
echo ""
echo "Running notebook 16: Cross-Validation (LOO zero-shot transfer)..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/16_cross_validation.ipynb

echo ""
echo "Notebook 16 complete."
echo "Results saved to: results/cv_transfer_results.csv"
echo "Figures saved to: figures/nb16_cv_heatmap.png"
echo "             and: figures/nb16_source_comparison.png"
echo "Job finished: $(date)"
