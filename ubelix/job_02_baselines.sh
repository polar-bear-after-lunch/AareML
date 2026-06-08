#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 02 (Baselines)
#
# CPU-only job — no GPU needed.
# Estimated runtime: ~15 min
#
# Submit with: sbatch ubelix/job_02_baselines.sh
# Or chain after job 01: sbatch --dependency=afterok:<JOB_01_ID> ubelix/job_02_baselines.sh
# =============================================================================

#SBATCH --job-name="aareml_02_baselines"
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_02_baselines_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_02_baselines_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

echo ""
echo "Running notebook 02: Baselines..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=2700 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/02_baselines.ipynb || NB_EXIT=$?

echo ""
echo "Notebook 02 complete."
echo "Results saved to: results/baseline_results.csv"
echo "Results saved to: results/baseline_per_gauge.csv"
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
