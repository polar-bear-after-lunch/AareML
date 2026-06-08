#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 15 (Scientific Rigor) — GPU node
# Estimated runtime: ~20 min
# =============================================================================

#SBATCH --job-name="aareml_15_rigor"
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_15_rigor_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_15_rigor_%j.err

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
    notebooks/15_scientific_rigor.ipynb

echo "Notebook 15 complete. Job finished: $(date)"

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
    git fetch origin main
    git rebase origin/main && git push origin main && echo "Results pushed (attempt $attempt)." && break
    git rebase --abort 2>/dev/null || true
    echo "Push attempt $attempt failed, retrying in 15s..."
    sleep 15
done
