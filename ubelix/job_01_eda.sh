#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 01 (Data Exploration / EDA)
#
# CPU-only job — no GPU needed.
# Estimated runtime: ~10 min
#
# Submit with: sbatch ubelix/job_01_eda.sh
# =============================================================================

#SBATCH --job-name="aareml_01_eda"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_01_eda_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_01_eda_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

echo ""
echo "Running notebook 01: Data Exploration..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/01_data_exploration.ipynb

echo ""
echo "Notebook 01 complete."
echo "Job finished: $(date)"
