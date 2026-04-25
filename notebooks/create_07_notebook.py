#!/usr/bin/env python3
"""Create notebook 07: Lake Mendota EDA"""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# Cell 0 — Markdown: Colab badge + title
cell0 = new_markdown_cell(source="""\
# Notebook 07 — Lake Mendota EDA
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/polar-bear-after-lunch/AareML/blob/main/notebooks/07_lake_eda.ipynb)

Exploratory data analysis of Lake Mendota surface data from LakeBeD-US (McAfee et al., 2025).
Mirrors notebook 01 structure for direct river-vs-lake comparison.

**Key questions:**
- How does Lake Mendota's DO range, seasonality, and coverage compare to gauge 2473?
- Are the temporal patterns similar enough to justify the same modelling framework?
- What are the key differences that might explain the 3.4× RMSE gap?""")

# Cell 1 — CPU optimisation
cell1 = new_code_cell(source="""\
import os, multiprocessing
N_CPU = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(N_CPU)
os.environ['MKL_NUM_THREADS'] = str(N_CPU)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CPU)
print(f'CPU cores: {N_CPU}')""")

# Cell 2 — Imports
cell2 = new_code_cell(source="""\
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 130})

FIGURES_DIR = Path('../figures')
DATA_DIR    = Path('../data/lakebed-us')""")

# Cell 3 — Markdown: Load Data
cell3 = new_markdown_cell(source="## 1. Load Data")

# Cell 4 — Load and basic info
cell4 = new_code_cell(source="""\
# Load daily surface data
surface_csv = DATA_DIR / 'ME_daily_surface.csv'
if surface_csv.exists():
    lake = pd.read_csv(surface_csv, parse_dates=['date'], index_col='date')
    print(f'Loaded from CSV: {len(lake)} daily rows')
else:
    print('ME_daily_surface.csv not found. Run: python download_data.py --lake')
    print('Or run notebook 06 first which generates this file.')
    raise FileNotFoundError(str(surface_csv))

# Reindex to continuous daily frequency (fill gaps with NaN)
idx = pd.date_range(lake.index.min(), lake.index.max(), freq='D')
lake = lake.reindex(idx)

print(f'\\nLake Mendota surface data (depth ≈ 1m)')
print(f'Date range : {lake.index.min().date()} → {lake.index.max().date()}')
print(f'Total days : {len(lake)}')
print(f'\\nVariable coverage:')
for col in lake.columns:
    cov = lake[col].notna().mean()
    valid = lake[col].dropna()
    print(f'  {col:12s}: {cov:5.1%}  |  '
          f'range [{valid.min():.2f}, {valid.max():.2f}]  |  '
          f'mean {valid.mean():.2f}  |  std {valid.std():.2f}')""")

# Cell 5 — Markdown: Full Time Series
cell5 = new_markdown_cell(source="## 2. Full Time Series (mirroring Fig 2 from notebook 01)")

# Cell 6 — Full time series plot
cell6 = new_code_cell(source="""\
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
colors = ['#A84B2F', '#01696F', '#006494', '#7A39BB']
labels = ['Temperature (°C)', 'DO (mg/L)', 'Chlorophyll-a (RFU)', 'Phycocyanin']
cols   = ['temp', 'do', 'chla_rfu', 'phyco']

for ax, col, color, label in zip(axes, cols, colors, labels):
    ax.plot(lake.index, lake[col], color=color, linewidth=0.7, alpha=0.85)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Year')
plt.suptitle('Lake Mendota — Surface Sensor Time Series (depth ≈ 1m, daily median)',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_mendota_full_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: 07_mendota_full_timeseries.png')""")

# Cell 7 — Markdown: Seasonal DO Cycle
cell7 = new_markdown_cell(source="""\
## 3. Seasonal DO Cycle (mirroring Fig 3 from notebook 01)
Compare with gauge 2473: DO ranges 8–14 mg/L with summer minimum.""")

# Cell 8 — Seasonal DO cycle
cell8 = new_code_cell(source="""\
# Monthly boxplot of DO — same as notebook 01 Fig 3
lake_use = lake[lake.index >= '2006-01-01'].copy()
lake_use['month'] = lake_use.index.month
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DO seasonal cycle
ax = axes[0]
do_by_month = [lake_use[lake_use['month']==m]['do'].dropna().values
               for m in range(1, 13)]
bp = ax.boxplot(do_by_month, labels=month_labels, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor('#01696F')
    patch.set_alpha(0.7)
ax.set_ylabel('Daily Median DO (mg/L)')
ax.set_title('Lake Mendota — Seasonal DO Cycle')
ax.grid(True, alpha=0.3)

# Temperature seasonal cycle
ax = axes[1]
temp_by_month = [lake_use[lake_use['month']==m]['temp'].dropna().values
                 for m in range(1, 13)]
bp2 = ax.boxplot(temp_by_month, labels=month_labels, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
for patch in bp2['boxes']:
    patch.set_facecolor('#A84B2F')
    patch.set_alpha(0.7)
ax.set_ylabel('Daily Median Temperature (°C)')
ax.set_title('Lake Mendota — Seasonal Temperature Cycle')
ax.grid(True, alpha=0.3)

plt.suptitle('Lake Mendota — Seasonal Cycles (2006–2023)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_mendota_seasonal_cycles.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: 07_mendota_seasonal_cycles.png')

# Print seasonal stats
print('\\nMonthly DO medians (mg/L):')
print(lake_use.groupby('month')['do'].median().round(2).to_string())""")

# Cell 9 — Markdown: DO Distribution Comparison
cell9 = new_markdown_cell(source="## 4. DO Distribution Comparison: River vs Lake")

# Cell 10 — River vs lake DO distribution
cell10 = new_code_cell(source="""\
# Load river gauge 2473 data for comparison
import sys; sys.path.insert(0, '..')
from src.data import load_gauge, preprocess

raw_river = load_gauge('2473')
river = preprocess(raw_river)
river_test = river[river.index >= '2017-01-01']  # test period

# Lake test period (same 2017 onwards)
lake_test = lake_use[lake_use.index >= '2017-01-01']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DO distributions
ax = axes[0]
river_do = river_test['O2C_sensor'].dropna()
lake_do  = lake_test['do'].dropna()
ax.hist(river_do, bins=50, alpha=0.6, color='#01696F', label=f'River gauge 2473 (n={len(river_do):,})')
ax.hist(lake_do,  bins=50, alpha=0.6, color='#A84B2F', label=f'Lake Mendota (n={len(lake_do):,})')
ax.set_xlabel('DO (mg/L)')
ax.set_ylabel('Count')
ax.set_title('DO Distribution — Test Period (2017+)')
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature distributions
ax = axes[1]
river_temp = river_test['temp_sensor'].dropna()
lake_temp  = lake_test['temp'].dropna()
ax.hist(river_temp, bins=50, alpha=0.6, color='#01696F', label=f'River gauge 2473 (n={len(river_temp):,})')
ax.hist(lake_temp,  bins=50, alpha=0.6, color='#A84B2F', label=f'Lake Mendota (n={len(lake_temp):,})')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Count')
ax.set_title('Temperature Distribution — Test Period (2017+)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('River vs Lake — Variable Distributions', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_river_vs_lake_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print('DO statistics comparison:')
print(f"{'':20s} {'River 2473':>14s}  {'Lake Mendota':>14s}")
print(f"{'Mean (mg/L)':20s} {river_do.mean():>14.3f}  {lake_do.mean():>14.3f}")
print(f"{'Std (mg/L)':20s} {river_do.std():>14.3f}  {lake_do.std():>14.3f}")
print(f"{'Min (mg/L)':20s} {river_do.min():>14.3f}  {lake_do.min():>14.3f}")
print(f"{'Max (mg/L)':20s} {river_do.max():>14.3f}  {lake_do.max():>14.3f}")
print(f"{'Coverage':20s} {river_test['O2C_sensor'].notna().mean():>14.1%}  {lake_test['do'].notna().mean():>14.1%}")""")

# Cell 11 — Markdown: Autocorrelation Analysis
cell11 = new_markdown_cell(source="""\
## 5. Autocorrelation Analysis
The SHAP analysis found the LSTM has an effective memory of 3–4 days.
Does the autocorrelation structure explain why rivers are more predictable?""")

# Cell 12 — Autocorrelation comparison
cell12 = new_code_cell(source="""\
from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# River DO autocorrelation
river_do_clean = river['O2C_sensor'].dropna()
lake_do_clean  = lake_use['do'].dropna()

for row, (series, name, color) in enumerate([
    (river_do_clean, 'River Gauge 2473', '#01696F'),
    (lake_do_clean,  'Lake Mendota',     '#A84B2F'),
]):
    # ACF plot (manual, up to 30 lags)
    ax = axes[row, 0]
    lags = range(1, 31)
    acf_vals = [series.autocorr(lag=lag) for lag in lags]
    ax.bar(lags, acf_vals, color=color, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(1.96/len(series)**0.5,  color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(-1.96/len(series)**0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{name} — DO Autocorrelation (lags 1–30)')
    ax.grid(True, alpha=0.3)

    # Rolling std (variability)
    ax = axes[row, 1]
    roll_std = series.rolling(30).std()
    ax.plot(series.index, roll_std, color=color, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('30-day Rolling Std (mg/L)')
    ax.set_title(f'{name} — DO Variability Over Time')
    ax.grid(True, alpha=0.3)

plt.suptitle('River vs Lake — DO Autocorrelation and Variability', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_river_vs_lake_autocorrelation.png', dpi=150, bbox_inches='tight')
plt.show()

print('DO autocorrelation at key lags:')
print(f"{'Lag':>5s}  {'River 2473':>12s}  {'Lake Mendota':>12s}")
for lag in [1, 2, 3, 7, 14, 21]:
    r_acf = river_do_clean.autocorr(lag=lag)
    l_acf = lake_do_clean.autocorr(lag=lag)
    print(f"{lag:>5d}  {r_acf:>12.3f}  {l_acf:>12.3f}")""")

# Cell 13 — Markdown: Data Coverage Comparison
cell13 = new_markdown_cell(source="## 6. Data Coverage Comparison")

# Cell 14 — Coverage comparison
cell14 = new_code_cell(source="""\
# Year-by-year coverage
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (series, name, color) in zip(axes, [
    (river['O2C_sensor'], 'River Gauge 2473', '#01696F'),
    (lake_use['do'],      'Lake Mendota',     '#A84B2F'),
]):
    yearly_cov = series.groupby(series.index.year).apply(lambda x: x.notna().mean())
    ax.bar(yearly_cov.index, yearly_cov.values, color=color, alpha=0.75)
    ax.axhline(0.1, color='red', linestyle='--', linewidth=1, label='10% threshold')
    ax.set_xlabel('Year')
    ax.set_ylabel('DO Coverage')
    ax.set_title(f'{name} — Annual DO Coverage')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('River vs Lake — Annual DO Data Coverage', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_coverage_comparison.png', dpi=150, bbox_inches='tight')
plt.show()""")

# Cell 15 — Markdown: Summary
cell15 = new_markdown_cell(source="## 7. Summary: Does it make sense to compare?")

# Cell 16 — Summary comparison table
cell16 = new_code_cell(source="""\
# Print full comparison table
print('='*70)
print('RIVER vs LAKE COMPARISON SUMMARY')
print('='*70)

comparisons = {
    'Dataset':          ('CAMELS-CH-Chem',      'LakeBeD-US (McAfee 2025)'),
    'Location':         ('Switzerland (Alpine)', 'Wisconsin, USA (temperate)'),
    'Water body':       ('River (lotic)',        'Lake (lentic)'),
    'Depth':            ('Well-mixed, N/A',      '~1 m surface'),
    'Period':           ('1981–2020',            '2006–2023'),
    'DO coverage':      ('~97% (gauge 2473)',    '~51% (Mendota)'),
    'DO mean':          (f'{river_do.mean():.2f} mg/L', f'{lake_do.mean():.2f} mg/L'),
    'DO std':           (f'{river_do.std():.2f} mg/L',  f'{lake_do.std():.2f} mg/L'),
    'DO range':         (f'{river_do.min():.1f}–{river_do.max():.1f} mg/L',
                         f'{lake_do.min():.1f}–{lake_do.max():.1f} mg/L'),
    'Temp range':       (f'{river_temp.min():.1f}–{river_temp.max():.1f} °C',
                         f'{lake_temp.min():.1f}–{lake_temp.max():.1f} °C'),
    'DO autocorr(1d)':  (f'{river_do_clean.autocorr(1):.3f}',
                         f'{lake_do_clean.autocorr(1):.3f}'),
    'DO autocorr(7d)':  (f'{river_do_clean.autocorr(7):.3f}',
                         f'{lake_do_clean.autocorr(7):.3f}'),
    'Ridge DO RMSE':    ('0.303 mg/L',           '1.030 mg/L'),
    'LSTM DO RMSE':     ('0.299 mg/L',           '— (pending)'),
}

print(f"{'Property':25s}  {'River Gauge 2473':25s}  {'Lake Mendota':25s}")
print('-'*78)
for k, (v1, v2) in comparisons.items():
    print(f'{k:25s}  {v1:25s}  {v2:25s}')

print()
print('Conclusion: The comparison is scientifically meaningful but not controlled.')
print('Both systems show clear seasonal DO cycles and respond to temperature,')
print('justifying the same LSTM architecture. Key differences:')
print('  1. River DO is more autocorrelated (higher day-1 ACF) → more predictable')
print('  2. Lake DO has wider range → harder to constrain forecasts')
print('  3. Lake data is sparser (51% vs 97%) → more imputation noise')
print('  4. Different input features (lake: chla/phyco vs river: pH/EC)')""")

# Assemble all cells
nb.cells = [
    cell0, cell1, cell2, cell3, cell4,
    cell5, cell6, cell7, cell8, cell9,
    cell10, cell11, cell12, cell13, cell14,
    cell15, cell16,
]

# Set kernel metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "aareml",
        "language": "python",
        "name": "aareml"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

# Write notebook
output_path = '/home/user/workspace/AareML/notebooks/07_lake_eda.ipynb'
with open(output_path, 'w') as f:
    nbformat.write(nb, f)

print(f'Notebook written to: {output_path}')

# Validate: ensure it is valid JSON
with open(output_path, 'r') as f:
    parsed = json.load(f)
print(f'Valid JSON: True')
print(f'Number of cells: {len(parsed["cells"])}')
print('Notebook created successfully')
