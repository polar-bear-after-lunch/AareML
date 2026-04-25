#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 03 (LSTM Single-Site + Optuna)
#
# Estimated runtime: ~20 min on RTX 3090 (vs ~90 min on Mac CPU)
# GPU memory needed: ~4 GB
#
# Submit with: sbatch ubelix/job_03_lstm.sh
# Monitor with: squeue --me
# View output: tail -f logs/job_03_lstm.out
# =============================================================================

#SBATCH --job-name="aareml_03_lstm"
#SBATCH --time=02:00:00              # 2 hours max (plenty of buffer)
#SBATCH --mem-per-cpu=8G             # 8 GB RAM
#SBATCH --cpus-per-task=4            # 4 CPU cores for data loading
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue            # do not auto-restart if preempted
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch   # ← replace with your email
#SBATCH --mail-type=END,FAIL        # notify on completion or failure
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_03_lstm_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_03_lstm_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -e
mkdir -p logs results figures

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Load modules and activate environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml

# Move to project root
cd $SLURM_SUBMIT_DIR

# ── Run notebook 03 ──────────────────────────────────────────────────────────
echo ""
echo "Running notebook 03: LSTM Single-Site + Optuna..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/03_lstm_single_site.ipynb

echo ""
echo "Notebook 03 complete."
echo "Results saved to: results/"
echo "Figures saved to: figures/"
echo "Job finished: $(date)"
