# AareML

**Predicting River Water Quality in Swiss Catchments with LSTMs**

CAS in Advanced Machine Learning · University of Bern · June 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)

## Overview

AareML applies a sequence-to-sequence LSTM, modelled on the [LakeBeD-US benchmark](https://essd.copernicus.org/articles/17/3141/2025/) (McAfee et al., 2025), to predict dissolved oxygen (DO) and water temperature at 14-day horizons across 86 Swiss river gauges from the [CAMELS-CH-Chem dataset](https://zenodo.org/records/14980027) (Nascimento et al., 2025).

## Key Results

| Model | DO RMSE | Temp RMSE | NSE |
|-------|---------|-----------|-----|
| Persistence | 0.339 mg/L | 1.365 °C | 0.860 |
| Ridge Regression | 0.303 mg/L | 1.261 °C | 0.888 |
| **LSTM (default)** | **0.299 mg/L** | **1.253 °C** | **0.892** |
| LSTM (Optuna best) | 0.301 mg/L | 1.280 °C | 0.889 |
| LakeBeD-US LSTM (ref.) | 1.400 mg/L | — | — |

**Multi-site transfer** (12 gauges): mean DO RMSE = 0.425 mg/L (zero-shot), 0.386 mg/L (per-gauge retraining) — both 3.3–3.6× better than the LakeBeD-US lake reference.

**SHAP findings**: temperature[t−1] is the dominant driver (mean |SHAP|=0.644), ahead of DO itself. Effective LSTM memory: 3–4 days despite 21-day lookback.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/polar-bear-after-lunch/AareML.git
cd AareML
```

### 2. Create environment
```bash
conda create -n aareml python=3.11 -y
conda activate aareml
conda install -c conda-forge llvmlite numba -y
pip install -r requirements.txt
```

### 3. Download data (~360 MB)
```bash
python download_data.py
```
This downloads and prepares both datasets automatically:
- **CAMELS-CH-Chem** (~165 MB) from [Zenodo](https://zenodo.org/records/14980027)
- **LakeBeD-US Lake Mendota** (~194 MB) from [Hugging Face](https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE)

### 4. Run notebooks in order
```
01_data_exploration.ipynb      — EDA and data availability
02_baselines.ipynb             — Persistence, Climatology, Ridge
03_lstm_single_site.ipynb      — Seq2Seq LSTM + Optuna tuning
04_multisite_analysis.ipynb    — Zero-shot transfer + per-gauge retraining
05_shap_interpretation.ipynb   — GradientSHAP attribution
06_cross_ecosystem_lake.ipynb  — River vs. Lake Mendota comparison
```

## Repository Structure

```
AareML/
├── notebooks/          — Jupyter notebooks (01–06)
├── src/
│   ├── config.py       — Shared configuration
│   ├── data.py         — Data loading and preprocessing
│   ├── metrics.py      — RMSE, MAE, NSE, KGE, bootstrap CI
│   ├── model.py        — Seq2SeqLSTM, EA-LSTM, training utilities
│   └── impute.py       — Self-attention imputer (SAITS-inspired)
├── results/            — CSV results tables and checkpoints
├── figures/            — Generated figures
├── ubelix/             — SLURM job scripts for UBELIX HPC
├── data/               — Data directory (excluded from git)
├── download_data.py    — Data download script
├── requirements.txt
└── CHANGELOG.md
```

## Data

Data is excluded from this repository due to size. Use `python download_data.py` to fetch all required datasets. See [CHANGELOG.md](CHANGELOG.md) for version history.

## Citation

If you use this code, please cite the underlying datasets:

- Nascimento et al. (2025). CAMELS-CH-Chem. Zenodo. https://doi.org/10.5281/zenodo.14980027
- McAfee et al. (2025). LakeBeD-US. ESSD. https://doi.org/10.5194/essd-17-3141-2025

## License

MIT License — see [LICENSE](LICENSE) for details.

*AI assistance (Perplexity Computer) was used for code scaffolding, report drafting, and data exploration. All scientific interpretations are the author's own.*
