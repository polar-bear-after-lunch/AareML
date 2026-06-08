#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 04b (Temperature Multi-Site Analysis)
#
# IMPORTANT: Run AFTER job_04_multisite.sh completes.
#
# Estimated runtime: ~6h on RTX 4090 (86 gauges × transfer + EA-LSTM training)
# NOTE (nb04b 86-gauge expansion): gauge discovery now scans all 86 CAMELS-CH-Chem
# temperature gauges directly from raw CSVs (≥365 days threshold), replacing the
# previous DO-filtered list of ~15 gauges. Runtime doubled to 6h accordingly.
# GPU memory needed: ~8 GB
# NOTE (nb04b update): EA-LSTM now uses CAMELS-CH base attributes (Höge et al. 2023):
# elev_mean, aridity, p_mean, frac_snow + landcover from CAMELS-CH-Chem
# (forest_frac, crop_frac, urban_frac). Matching nb04 static feature set.
# Results saved to temp_ea_lstm_results.csv and temp_multisite_combined.csv.
#
# Submit with: sbatch ubelix/job_04b_temp.sh
# Or chain after job 04: sbatch --dependency=afterok:<JOB_04_ID> ubelix/job_04b_temp.sh
# =============================================================================

#SBATCH --job-name="aareml_04b_temp"
#SBATCH --time=06:00:00
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
NB_EXIT=0
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/04b_multisite_temperature.ipynb || NB_EXIT=$?

echo ""
echo "Notebook 04b complete."
echo "Job finished: $(date)"

# Auto-push results to GitHub
cd /storage/homefs/tn20y076/AareML
git config user.email "aareml@project.ch"
git config user.name "AareML"
git add -A
git commit -m "ubelix run $(basename $0 .sh) $(date '+%Y-%m-%d %H:%M')" || true
for attempt in 1 2 3 4 5; do
    git stash || true
    git add -A
    git commit -m "ubelix run $(basename $0 .sh) $(date '+%Y-%m-%d %H:%M') NB_EXIT=$NB_EXIT" || true
    git stash pop || true
    git pull --rebase origin main && git push origin main && echo "Results pushed (attempt $attempt)." && break
    echo "Push attempt $attempt failed, retrying in 15s..."
    sleep 15
done
