#!/bin/bash
# =============================================================================
# AareML — UBELIX Environment Setup
# Run this ONCE after logging in to UBELIX for the first time.
# Usage: bash ubelix/setup_env.sh
# =============================================================================

set -e  # exit on any error

echo "============================================="
echo "  AareML — UBELIX Environment Setup"
echo "============================================="

# ── 1. Load Anaconda module ──────────────────────────────────────────────────
echo ""
echo "[1/6] Loading Anaconda3 module..."
module load Anaconda3
eval "$(conda shell.bash hook)"

# ── 2. Create conda environment ──────────────────────────────────────────────
echo ""
echo "[2/6] Creating conda environment 'aareml' (Python 3.11)..."
if conda env list | grep -q "^aareml "; then
    echo "  Environment 'aareml' already exists — skipping creation."
else
    conda create -n aareml python=3.11 -y
fi
conda activate aareml

# ── 3. Install llvmlite + numba via conda (avoids cmake build) ───────────────
echo ""
echo "[3/6] Installing llvmlite + numba via conda-forge..."
conda install -c conda-forge llvmlite numba -y

# ── 4. Install PyTorch with CUDA 12.1 support ────────────────────────────────
echo ""
echo "[4/6] Installing PyTorch with CUDA 12.1 (for RTX 3090 / Tesla P100)..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── 5. Install remaining requirements ────────────────────────────────────────
echo ""
echo "[5/6] Installing project requirements..."
pip install -r requirements.txt --quiet

# ── 6. Register kernel for Jupyter ──────────────────────────────────────────
echo ""
echo "[6/6] Registering Jupyter kernel..."
pip install ipykernel --quiet
python -m ipykernel install --user --name aareml --display-name "Python (aareml)"

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Verification"
echo "============================================="
python -c "
import torch
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:        {torch.cuda.get_device_name(0)}')
    print(f'VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import sklearn; print(f'sklearn:    {sklearn.__version__}')
import optuna;  print(f'Optuna:     {optuna.__version__}')
"

echo ""
echo "Setup complete! Environment 'aareml' is ready."
echo "To activate manually: module load Anaconda3 && conda activate aareml"
