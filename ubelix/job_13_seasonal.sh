#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 13 (Seasonal Analysis) — CPU only
# Estimated runtime: ~15 min
# =============================================================================

#SBATCH --job-name="aareml_13_seasonal"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_13_seasonal_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_13_seasonal_%j.err

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
    notebooks/13_seasonal_analysis.ipynb

echo "Notebook 13 complete. Job finished: $(date)"

# Auto-push results to GitHub
cd /storage/homefs/tn20y076/AareML
git config user.email "aareml@project.ch"
git config user.name "AareML"
git add -A
git commit -m "ubelix run $(basename $0 .sh) $(date '+%Y-%m-%d %H:%M')" || true
for attempt in 1 2 3 4 5; do
    SCRIPT_NAME=$(basename $0 .sh)
    git add -A
    git commit -m "ubelix run $SCRIPT_NAME $(date '+%Y-%m-%d %H:%M') NB_EXIT=$NB_EXIT" || true
    git pull --no-rebase -X ours origin main
    git push origin main && echo "Results pushed (attempt $attempt)." && break
    echo "Push attempt $attempt failed, retrying in 15s..."
    sleep 15
done
