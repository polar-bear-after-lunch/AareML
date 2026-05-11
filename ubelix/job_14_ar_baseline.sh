#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 14 (Autoregressive Baseline Comparison)
#
# CPU-only job — statsmodels AR/VAR, no GPU needed.
# Estimated runtime: ~15 min
# =============================================================================

#SBATCH --job-name="aareml_14_ar"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_14_ar_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_14_ar_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/14_autoregressive_baseline.ipynb

echo "Notebook 14 complete. Job finished: $(date)"
