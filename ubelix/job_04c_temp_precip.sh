#!/bin/bash
# =============================================================================
# AareML — SLURM Job: Notebook 04c (Temperature Forecast with EC/Precip Input)
#
# IMPORTANT: Run AFTER job_04b_temp.sh completes (no hard SLURM dependency set,
# but nb04c loads zero-shot reference results written by nb04b).
#
# Background: Thiago Nascimento (Eawag) suggested using precipitation as an
# input feature for temperature forecasting. CAMELS-CH meteorological forcing
# is not available locally, so this notebook uses electrical conductivity (EC)
# as a precipitation proxy (EC drops during rain events due to dilution).
#
# Estimated runtime: ~2h on RTX 4090 (86 gauges × 2 models = 172 training runs)
# GPU memory needed: ~6 GB
#
# Submit with: sbatch ubelix/job_04c_temp_precip.sh
# Or chain after job 04b: sbatch --dependency=afterok:<JOB_04B_ID> ubelix/job_04c_temp_precip.sh
# =============================================================================

#SBATCH --job-name="aareml_04c_temp_precip"
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --no-requeue
#SBATCH --mail-user=YOUR_EMAIL@unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_04c_temp_precip_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_04c_temp_precip_%j.err

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
echo "Running notebook 04c: Temperature Forecast with EC/Precip Input..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=10800 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/04c_temp_precip_forecast.ipynb

echo ""
echo "Notebook 04c complete."
echo "Results written to:"
echo "  results/temp_ec_results.csv"
echo "  results/temp_forecast_comparison.csv"
echo "  figures/nb04c_ec_temp_correlation.png"
echo "Job finished: $(date)"
