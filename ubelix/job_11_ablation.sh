#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 11 (Ablation Study)
#
# GPU job — ablation trains 4 × 3 seeds × 250 epochs = heavy.
# Estimated runtime: ~90 min on RTX 4090
#
# Submit after job_03_lstm.sh (needs checkpoint + scalers)
# =============================================================================

#SBATCH --job-name="aareml_11_ablation"
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --account=gratis
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_11_ablation_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_11_ablation_%j.err

set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs -I{} echo "GPU: {}"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
cd /storage/homefs/tn20y076/AareML

echo ""
echo "Running notebook 11: Ablation Study..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=9000 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/11_ablation_study.ipynb || NB_EXIT=$?

echo ""
echo "Notebook 11 complete."
echo "Results: results/ablation_results.csv"
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
