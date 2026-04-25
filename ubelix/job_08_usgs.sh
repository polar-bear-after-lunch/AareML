#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 08 (USGS Cross-Continental Transfer)
#
# IMPORTANT: Run AFTER job_05_shap.sh completes (needs checkpoint).
#
# Estimated runtime: ~30 min on RTX 4090 (5 USGS sites, zero-shot)
# GPU memory needed: ~4 GB
#
# Submit with: sbatch ubelix/job_08_usgs.sh
# =============================================================================

#SBATCH --job-name="aareml_08_usgs"
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_08_usgs_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_08_usgs_%j.err

set -e
mkdir -p /storage/homefs/tn20y076/AareML/logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

# Install dataretrieval if not present
python -c "import dataretrieval" 2>/dev/null || pip install dataretrieval -q

# Check checkpoint exists
if [ ! -f "results/lstm_single_site_best.pt" ]; then
    echo "ERROR: results/lstm_single_site_best.pt not found."
    echo "Please run job_03_lstm.sh first."
    exit 1
fi

echo ""
echo "Running notebook 08: USGS Cross-Continental Transfer..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/08_usgs_transfer.ipynb

echo ""
echo "Notebook 08 complete."
echo "Job finished: $(date)"
