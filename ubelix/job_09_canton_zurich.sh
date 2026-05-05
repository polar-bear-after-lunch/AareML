#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 09 (Canton Zurich Analysis)
#
# CPU-only job — no GPU needed.
# Estimated runtime: ~10 min
#
# Submit with: sbatch ubelix/job_09_canton_zurich.sh
# =============================================================================

#SBATCH --job-name="aareml_09_canton"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_09_canton_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_09_canton_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

echo ""
echo "Running notebook 09: Canton Zurich Analysis..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/09_canton_zurich_analysis.ipynb

echo ""
echo "Notebook 09 complete."
echo "Job finished: $(date)"
