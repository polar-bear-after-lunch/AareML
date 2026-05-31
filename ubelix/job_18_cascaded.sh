#!/bin/bash
#SBATCH --job-name="aareml_18_cascaded"
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_18_cascaded_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_18_cascaded_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /storage/homefs/tn20y076/AareML

mkdir -p logs

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=10800 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/18_cascaded_do_model.ipynb

echo ""
echo "Notebook 18 complete."
echo "Job finished: $(date)"
