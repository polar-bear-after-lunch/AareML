#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 06 (Cross-Ecosystem Lake Mendota)
#
# CPU-only job — no GPU needed.
# Runs the Lake Mendota cross-ecosystem experiment using the saved river LSTM.
# Estimated runtime: ~15 min
#
# IMPORTANT: Run AFTER job_03_lstm.sh — needs results/lstm_single_site_best.pt
#
# Submit with: sbatch ubelix/job_06_lake_mendota.sh
# =============================================================================

#SBATCH --job-name="aareml_06_lake"
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_06_lake_mendota_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_06_lake_mendota_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

if [ ! -f "results/lstm_single_site_best.pt" ]; then
    echo "ERROR: results/lstm_single_site_best.pt not found. Run job_03_lstm.sh first."
    exit 1
fi

echo ""
echo "Running notebook 06: Cross-Ecosystem Lake Mendota..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=2700 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/06_cross_ecosystem_lake.ipynb

echo ""
echo "Notebook 06 complete."
echo "Job finished: $(date)"
