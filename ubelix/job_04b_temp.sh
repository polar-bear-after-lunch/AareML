#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 04b (Temperature Multi-Site Analysis)
#
# IMPORTANT: Run AFTER job_04_multisite.sh completes.
#
# Estimated runtime: ~2h on RTX 4090 (80+ gauges × transfer)
# GPU memory needed: ~6 GB
#
# Submit with: sbatch ubelix/job_04b_temp.sh
# Or chain after job 04: sbatch --dependency=afterok:<JOB_04_ID> ubelix/job_04b_temp.sh
# =============================================================================

#SBATCH --job-name="aareml_04b_temp"
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_04b_temp_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_04b_temp_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

echo ""
echo "Running notebook 04b: Temperature Multi-Site Analysis..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/04b_multisite_temperature.ipynb

echo ""
echo "Notebook 04b complete."
echo "Job finished: $(date)"
