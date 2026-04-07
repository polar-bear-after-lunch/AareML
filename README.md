# AareML

**Predicting River Water Quality in Swiss Catchments using Machine Learning**

CAS Advanced Machine Learning — Final Project | University of Bern | June 2026

---

## Overview

AareML applies a sequence-to-sequence LSTM benchmark — originally defined on the [LakeBeD-US](https://essd.copernicus.org/articles/17/3141/2025/) lake water quality dataset — to Swiss river catchment data from [CAMELS-CH-Chem](https://pmc.ncbi.nlm.nih.gov/articles/PMC12287458/), the first European CAMELS extension incorporating water quality.

**Core ML task:** Predict dissolved oxygen concentration and water temperature at a 14-day horizon from a 21-day lookback window using daily sensor measurements across 115 Swiss catchments.

**Central research question:**
> Can the LakeBeD-US LSTM benchmark transfer from lakes to rivers, and what catchment characteristics drive variation in forecast skill across Swiss sites?

---

## Data

| Dataset | Source | Description |
|---|---|---|
| CAMELS-CH-Chem | [Zenodo](https://zenodo.org/records/14980027) | 40 water quality parameters, 115 Swiss catchments, 1981–2020 |
| CAMELS-CH | [Zenodo](https://zenodo.org/records/7784633) | Hydro-meteorological time series, catchment attributes |
| LakeBeD-US (CSE) | [Hugging Face](https://huggingface.co/datasets/lakebed-us) | Benchmark reference |

> Data files are not committed to this repository. See `data/README.md` for download instructions.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | EDA, data quality, spatial & temporal coverage |
| `02_baselines.ipynb` | Persistence, linear regression, Random Forest |
| `03_lstm_single_site.ipynb` | Seq2seq LSTM benchmark replication |
| `04_multisite_analysis.ipynb` | Cross-catchment evaluation (115 sites) |
| `05_shap_interpretation.ipynb` | SHAP feature importance analysis |

---

## Repository Structure

```
AareML/
├── notebooks/          # Jupyter notebooks (numbered pipeline)
├── src/                # Reusable Python modules (data loading, models, utils)
├── data/               # Data directory (not tracked by git — see data/README.md)
├── results/            # Model outputs, metrics CSVs
├── figures/            # Generated plots and maps
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Methods

- **Model:** Sequence-to-sequence LSTM-RNN (encoder-decoder), replicating LakeBeD-US benchmark
- **Imputation:** SAITS (self-attention-based imputation for time series)
- **Hyperparameter tuning:** Optuna (tree-structured Parzen estimator)
- **Baselines:** Persistence, OLS regression, Random Forest
- **Interpretation:** SHAP values for feature importance across catchment types

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
optuna
shap
geopandas
pyarrow     # for Parquet files
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

- McAfee et al. (2025). LakeBeD-US. *Earth System Science Data.* https://doi.org/10.5194/essd-17-3141-2025
- Nascimento et al. (2025). CAMELS-CH-Chem. *Scientific Data.* https://doi.org/10.1038/s41597-025-05625-1
- Höge et al. (2023). CAMELS-CH. *Earth System Science Data.* https://doi.org/10.5194/essd-15-5755-2023
