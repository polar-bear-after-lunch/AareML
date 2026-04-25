# AareML

**Predicting River Water Quality in Swiss Catchments with LSTMs**

CAS in Advanced Machine Learning · University of Bern · June 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)

## Overview

AareML applies a sequence-to-sequence LSTM, modelled on the [LakeBeD-US benchmark](https://essd.copernicus.org/articles/17/3141/2025/) (McAfee et al., 2025), to predict dissolved oxygen (DO) and water temperature at 14-day horizons across 86 Swiss river gauges from the [CAMELS-CH-Chem dataset](https://zenodo.org/records/14980027) (Nascimento et al., 2025).

## Key Results

| Model | DO RMSE | Temp RMSE | KGE |
|-------|---------|-----------|-----|
| Persistence | 0.339 mg/L | 1.365 °C | 0.930 |
| Climatology | 0.334 mg/L | 1.444 °C | 0.853 |
| Ridge Regression | 0.303 mg/L | 1.261 °C | 0.908 |
| **LSTM (default)** | **0.307 mg/L** | **1.270 °C** | **0.854** |
| **LSTM (best, Optuna)** | **0.302 mg/L** | **1.247 °C** | **0.945** |
| LakeBeD-US LSTM (ref.) | 1.400 mg/L | — | — |

**Multi-site DO transfer** (12 gauges): mean RMSE = 0.425 mg/L (zero-shot), 0.386 mg/L (per-gauge) — 3.3–3.6× better than the LakeBeD-US lake reference.

**Temperature multi-site** (80+ gauges): results from UBELIX run — see `results/04b_temp_transfer_v1.20.csv`.

**SHAP findings**: `temp_sensor[t−1]` is the dominant driver (mean |SHAP| = 0.644), ahead of DO itself. Effective LSTM memory: 3–4 days despite 21-day lookback.

**Cross-ecosystem**: Lake Mendota Ridge DO RMSE = 1.030 mg/L (3.4× higher than river Ridge), confirming river water quality is substantially more predictable than lake DO.

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
Downloads and prepares:
- **CAMELS-CH-Chem** (~165 MB) from [Zenodo](https://zenodo.org/records/14980027)
- **LakeBeD-US Lake Mendota** (~194 MB) from [Hugging Face](https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE)

> **Note:** USGS data for notebook 08 is downloaded automatically at runtime via the `dataretrieval` package — no manual step needed.

### 4. Run notebooks in order
```
01_data_exploration.ipynb       — EDA and data availability
02_baselines.ipynb              — Persistence, Climatology, Ridge
03_lstm_single_site.ipynb       — Seq2Seq LSTM + Optuna tuning + 3-seed ensemble
04_multisite_analysis.ipynb     — Zero-shot transfer + per-gauge retraining (DO)
04b_multisite_temperature.ipynb — Temperature multi-site (80+ gauges)
05_shap_interpretation.ipynb    — GradientSHAP attribution
06_cross_ecosystem_lake.ipynb   — River vs. Lake Mendota comparison
07_lake_eda.ipynb               — Lake Mendota EDA
```

### 5. Running on UBELIX HPC
```bash
# Sync code to UBELIX
bash sync_to_ubelix.sh

# On UBELIX — download data and set up environment (first time only)
python download_data.py
bash ubelix/setup_env.sh

# Submit full job chain (03 → 04 → 04b → 05)
cd ubelix
sbatch run_all.sh

# Fetch results back to Mac
bash fetch_from_ubelix.sh
```

## Repository Structure

```
AareML/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_lstm_single_site.ipynb
│   ├── 04_multisite_analysis.ipynb
│   ├── 04b_multisite_temperature.ipynb
│   ├── 05_shap_interpretation.ipynb
│   ├── 06_cross_ecosystem_lake.ipynb
│   └── 07_lake_eda.ipynb
├── src/
│   ├── config.py       — Shared configuration (LOOKBACK=21, HORIZON=14)
│   ├── data.py         — Data loading, preprocessing, windowing
│   ├── metrics.py      — RMSE, MAE, NSE, KGE, bootstrap CI
│   ├── model.py        — Seq2SeqLSTM, EA-LSTM, ReduceLROnPlateau, checkpoints
│   └── impute.py       — Self-attention imputer (SAITS-inspired)
├── ubelix/
│   ├── run_all.sh          — Submit full job chain
│   ├── job_03_lstm.sh      — SLURM: notebook 03
│   ├── job_04_multisite.sh — SLURM: notebook 04
│   ├── job_04b_temp.sh     — SLURM: notebook 04b
│   ├── job_05_shap.sh      — SLURM: notebook 05
│   ├── setup_env.sh        — Conda environment setup
│   └── test_local.sh       — Smoke test (CPU, LOCAL_TEST mode)
├── tests/
│   └── test_src.py         — 53 pytest tests (all passing)
├── results/            — CSV results, checkpoints, Optuna study
├── figures/            — Generated figures (39 PNGs)
├── data/               — Data directory (git-ignored)
├── download_data.py    — Downloads CAMELS-CH-Chem + LakeBeD-US
├── sync_to_ubelix.sh   — rsync Mac → UBELIX
├── fetch_from_ubelix.sh — rsync UBELIX → Mac (results + notebooks)
├── run_all_notebooks.sh — Run all notebooks locally with timestamps
├── requirements.txt
└── CHANGELOG.md
```

## Testing

```bash
cd AareML
python -m pytest tests/test_src.py -v
# 53 tests, all passing
```

## Data

Data is excluded from this repository due to size. Use `python download_data.py` to fetch all datasets (~360 MB total). The script downloads, extracts, and preprocesses everything automatically.

USGS continuous monitoring data (notebook 08) is fetched at runtime via the `dataretrieval` Python package — no manual download required. An internet connection is needed when running notebook 08.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for the full version history (v1.0–v1.20).
Current version: **v1.22** (April 2026)

## Citation

If you use this code, please cite the underlying datasets:

- Nascimento et al. (2025). CAMELS-CH-Chem. *Zenodo*. https://doi.org/10.5281/zenodo.14980027
- McAfee et al. (2025). LakeBeD-US. *ESSD*, 17, 3141–3170. https://doi.org/10.5194/essd-17-3141-2025

## License

MIT License — see [LICENSE](LICENSE) for details.

*AI assistance (Perplexity Computer) was used for code scaffolding, report drafting, and data exploration. All scientific interpretations are the author's own.*
