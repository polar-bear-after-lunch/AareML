#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 04 (Multi-Site Evaluation)
#
# IMPORTANT: Run AFTER job_03_lstm.sh completes — needs the saved checkpoint.
#
# Estimated runtime: ~60 min on RTX 4090 (16 gauges × zero-shot + retrain + EA-LSTM)
# GPU memory needed: ~8 GB
# NOTE (nb04 update): EA-LSTM now uses CAMELS-CH base attributes (Höge et al. 2023):
# elev_mean, aridity, p_mean, frac_snow + landcover from CAMELS-CH-Chem
# (forest_frac, crop_frac, urban_frac). Source: camels-ch-base/camels_ch_attributes.csv.
#
# Submit with: sbatch ubelix/job_04_multisite.sh
# Or chain after job 03: sbatch --dependency=afterok:<JOB_03_ID> ubelix/job_04_multisite.sh
# =============================================================================

#SBATCH --job-name="aareml_04_multisite"
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G            # more RAM for loading 16 gauges
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue            # do not auto-restart if preempted
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch   # ← replace with your email
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_04_multisite_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_04_multisite_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

# Check that notebook 03 checkpoint exists
if [ ! -f "results/lstm_single_site_best.pt" ]; then
    echo "ERROR: results/lstm_single_site_best.pt not found."
    echo "Please run job_03_lstm.sh first."
    exit 1
fi

# ── Run notebook 04 ──────────────────────────────────────────────────────────
echo ""
echo "Running notebook 04: Multi-Site Evaluation..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/04_multisite_analysis.ipynb || NB_EXIT=$?

echo ""
echo "Notebook 04 complete."
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
