#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 07 (Lake EDA)
#
# CPU-only job — no GPU needed.
# Estimated runtime: ~10 min
#
# Submit with: sbatch ubelix/job_07_lake_eda.sh
# =============================================================================

#SBATCH --job-name="aareml_07_lake_eda"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_07_lake_eda_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_07_lake_eda_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

echo ""
echo "Running notebook 07: Lake EDA..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/07_lake_eda.ipynb

echo ""
echo "Notebook 07 complete."
echo "Job finished: $(date)"
