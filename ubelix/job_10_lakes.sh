#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 10 (Swiss Lakes EDA + LSTM)
#
# IMPORTANT: Run AFTER job_05_shap.sh — needs lstm_single_site_best.pt
#
# Estimated runtime: ~1h on RTX 4090 (EDA + zero-shot + 100 epochs retrain)
# GPU memory needed: ~4 GB
# =============================================================================

#SBATCH --job-name="aareml_10_lakes"
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_10_lakes_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_10_lakes_%j.err

set -e
mkdir -p /storage/homefs/tn20y076/AareML/logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

# Check checkpoint exists
if [ ! -f "results/lstm_single_site_best.pt" ]; then
    echo "ERROR: results/lstm_single_site_best.pt not found."
    echo "Please run job_03_lstm.sh first."
    exit 1
fi

# Download Swiss lakes data if not present
if [ ! -d "data/swiss-lakes" ] || [ -z "$(ls -A data/swiss-lakes 2>/dev/null)" ]; then
    echo "Downloading Swiss lakes dataset..."
    python download_data.py --swiss-lakes
fi

echo ""
echo "Running notebook 10: Swiss Lakes EDA + LSTM..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/10_swiss_lakes_lstm.ipynb || NB_EXIT=$?

echo ""
echo "Notebook 10 complete."
echo "Job finished: $(date)"

# Auto-push results to GitHub
cd /storage/homefs/tn20y076/AareML
git config user.email "aareml@project.ch"
git config user.name "AareML"
git add -A
git commit -m "ubelix run $(basename $0 .sh) $(date '+%Y-%m-%d %H:%M')" || true
for attempt in 1 2 3 4 5; do
    git add -A
    git commit -m "ubelix run $(SCRIPT=$(basename $0); echo ${SCRIPT%.sh}) $(date '+%Y-%m-%d %H:%M') NB_EXIT=$NB_EXIT" || true
    git reset --hard HEAD
    git pull origin main && git push origin main && echo "Results pushed (attempt $attempt)." && break
    echo "Push attempt $attempt failed, retrying in 15s..."
    sleep 15
done
