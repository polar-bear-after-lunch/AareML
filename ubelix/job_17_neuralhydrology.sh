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
pip install neuralhydrology xarray -q

# Patch NeuralHydrology logging_utils.py to avoid CalledProcessError
# when running outside a git repo (git describe --always fails on installed packages)
NH_LOG=$(python3 -c "import neuralhydrology.utils.logging_utils as m; import inspect; print(inspect.getfile(m))" 2>/dev/null)
if [ -n "$NH_LOG" ]; then
    sed -i 's/return subprocess.check_output(\["git"/return None  # patched: was subprocess.check_output(["git"/' "$NH_LOG" 2>/dev/null || true
    echo "Patched NeuralHydrology logging_utils.py at $NH_LOG"
fi

mkdir -p logs

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=aareml \
    notebooks/17_neuralhydrology.ipynb

echo ""
echo "Notebook 17 complete."
echo "Job finished: $(date)"

# Auto-push results to GitHub
cd /storage/homefs/tn20y076/AareML
git config user.email "aareml@project.ch"
git config user.name "AareML"
git add -A
git commit -m "ubelix run nb17 $(date '+%Y-%m-%d %H:%M')" && git push origin main
echo "Results pushed to GitHub."
