#!/bin/bash
#SBATCH --job-name="aareml_17_neuralhydrology"
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --output=/storage/homefs/tn20y076/AareML/logs/job_17_neuralhydrology_%j.out
#SBATCH --error=/storage/homefs/tn20y076/AareML/logs/job_17_neuralhydrology_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /storage/homefs/tn20y076/AareML

# Install neuralhydrology if needed
pip install neuralhydrology -q

mkdir -p logs

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=aareml \
    --ExecutePreprocessor.cwd=/storage/homefs/tn20y076/AareML \
    notebooks/17_neuralhydrology.ipynb

echo ""
echo "Notebook 17 complete."
echo "Job finished: $(date)"
