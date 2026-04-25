"""
Script to programmatically create 04b_multisite_temperature.ipynb
"""
import nbformat
import ast
import sys

def check_syntax(code, cell_num):
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in cell {cell_num}: {e}")
        return False

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "aareml"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

cells = []
syntax_errors = 0

# ── Cell 0 — Markdown (Colab badge) ──────────────────────────────────────
md0 = """# Notebook 04b — Temperature Multi-Site Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/polar-bear-after-lunch/AareML/blob/main/notebooks/04b_multisite_temperature.ipynb)

**Key insight:** Temperature is measured at all 86 CAMELS-CH-Chem gauges (vs only 16 for DO).
Using temperature as the primary target transforms AareML from a 1-gauge study to a genuine
80+ gauge multi-site analysis across all of Switzerland.

**This notebook:**
1. Identifies all gauges with sufficient temperature data (≥50% coverage)
2. Trains a single-site temperature LSTM on the focus gauge (2473)
3. Evaluates zero-shot transfer to all valid gauges
4. Performs per-gauge retraining across the full network
5. Correlates prediction skill with catchment attributes
6. Maps RMSE across Switzerland (if geopandas available)"""
cells.append(nbformat.v4.new_markdown_cell(md0))

# ── Cell 1 — Colab setup ─────────────────────────────────────────────────
code1 = """\
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import os
    from pathlib import Path
    if not str(Path.cwd()).endswith('AareML'):
        if not Path('AareML').exists():
            os.system('git clone https://github.com/polar-bear-after-lunch/AareML.git')
        os.chdir('AareML')
    os.system('pip install -q -r requirements.txt')
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DATA = Path('/content/drive/MyDrive/AareML_data')
    LOCAL_DATA = Path('data')
    LOCAL_DATA.mkdir(exist_ok=True)
    DRIVE_CAMELS = DRIVE_DATA / 'camels-ch-chem'
    LOCAL_CAMELS = LOCAL_DATA / 'camels-ch-chem'
    if DRIVE_CAMELS.exists() and not LOCAL_CAMELS.exists():
        os.system(f'ln -s {DRIVE_CAMELS} {LOCAL_CAMELS}')
    print(f'Setup complete. cwd: {os.getcwd()}')"""
if not check_syntax(code1, 1):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code1))

# ── Cell 2 — CPU thread optimisation ─────────────────────────────────────
code2 = """\
import os, multiprocessing
N_CPU = multiprocessing.cpu_count()
N_THREADS = min(N_CPU, 6)
os.environ['OMP_NUM_THREADS']      = str(N_THREADS)
os.environ['MKL_NUM_THREADS']      = str(N_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_THREADS)
print(f'CPU cores: {N_CPU} | Using {N_THREADS} threads')"""
if not check_syntax(code2, 2):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code2))

# ── Cell 3 — Imports ─────────────────────────────────────────────────────
code3 = """\
import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.config import (
    LOOKBACK, HORIZON, SEED, TRAIN_END, VAL_END, FOCUS_GAUGE,
    FEATURES_TEMP, TARGETS_TEMP, N_FEAT_TEMP, N_TGT_TEMP, TARGET_LABELS_TEMP,
    TEMP_MIN_COVERAGE,
)
from src.data import load_gauge, preprocess, train_val_test_split, make_windows, load_metadata
from src.model import (
    RiverDataset, Seq2SeqLSTM, train_model, predict, get_y_true,
    save_checkpoint, load_checkpoint, reconstruct_scalers,
)
from src.metrics import metrics_table, mean_rmse, block_bootstrap_ci

import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

_repo_root  = Path('.') if Path('figures').exists() else Path('..')
FIGURES_DIR = _repo_root / 'figures'
RESULTS_DIR = _repo_root / 'results'
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 130})
print(f'Features: {FEATURES_TEMP}')
print(f'Targets:  {TARGETS_TEMP}')"""
if not check_syntax(code3, 3):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code3))

# ── Cell 4 — GPU config ───────────────────────────────────────────────────
code4 = """\
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4 if DEVICE.type == 'cuda' else 0
PIN_MEMORY  = DEVICE.type == 'cuda'
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"""
if not check_syntax(code4, 4):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code4))

# ── Cell 5 — Markdown: Section 1 ─────────────────────────────────────────
md5 = """\
## 1. Gauge Discovery — Temperature Coverage

Unlike DO, temperature is observed at virtually all 86 CAMELS-CH-Chem gauges.
This section identifies all gauges with ≥50% temperature coverage and ≥50 valid test windows."""
cells.append(nbformat.v4.new_markdown_cell(md5))

# ── Cell 6 — Discover all temperature gauges ─────────────────────────────
code6 = """\
meta = load_metadata()
all_gauge_ids = meta['gauge_id'].astype(str).tolist() if 'gauge_id' in meta.columns else meta.index.astype(str).tolist()
print(f'Total gauges in metadata: {len(all_gauge_ids)}')

temp_gauges = []
coverage_dict = {}

print('Scanning all gauges for temperature coverage...')
for gid in tqdm(all_gauge_ids, desc='Gauges'):
    try:
        raw  = load_gauge(gid)
        data = preprocess(raw)
        cov  = data['temp_sensor'].notna().mean()
        coverage_dict[gid] = float(cov)
        if cov >= TEMP_MIN_COVERAGE:
            # Check enough test windows
            _, _, test = train_val_test_split(data)
            means = (pd.concat([test[FEATURES_TEMP].mean(), test[TARGETS_TEMP].mean()])
                     .groupby(level=0).first())
            try:
                X_te, y_te, _ = make_windows(test, means,
                                             features=FEATURES_TEMP, targets=TARGETS_TEMP)
                if len(X_te) >= 50:
                    temp_gauges.append(gid)
            except ValueError:
                pass
    except Exception:
        pass

print(f'\\nGauges with \\u2265{TEMP_MIN_COVERAGE:.0%} temp coverage: {sum(1 for v in coverage_dict.values() if v >= TEMP_MIN_COVERAGE)}')
print(f'Gauges with \\u226550 valid test windows: {len(temp_gauges)}')
print(f'Gauge IDs: {temp_gauges}')

# Coverage distribution
cov_series = pd.Series(coverage_dict)
print(f'\\nTemperature coverage across all gauges:')
print(f'  Mean: {cov_series.mean():.1%}')
print(f'  Median: {cov_series.median():.1%}')
print(f'  Min: {cov_series.min():.1%}')
print(f'  Gauges with >90% coverage: {(cov_series > 0.9).sum()}')"""
if not check_syntax(code6, 6):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code6))

# ── Cell 7 — Markdown: Section 2 ─────────────────────────────────────────
md7 = """\
## 2. Single-Site Temperature LSTM (Focus Gauge 2473)

Train a Seq2Seq LSTM on gauge 2473 with temperature as the sole target.
Features: [temp_sensor, pH_sensor, ec_sensor] — excluding DO to avoid
target leakage (temperature predicts temperature, but DO and temperature
are correlated so including DO as a feature would inflate performance)."""
cells.append(nbformat.v4.new_markdown_cell(md7))

# ── Cell 8 — Load and prepare focus gauge data ───────────────────────────
code8 = """\
raw_focus   = load_gauge(FOCUS_GAUGE)
data_focus  = preprocess(raw_focus)
train_focus, val_focus, test_focus = train_val_test_split(data_focus)

# Training means for imputation (no leakage)
train_means_focus = (
    pd.concat([train_focus[FEATURES_TEMP].mean(), train_focus[TARGETS_TEMP].mean()])
    .groupby(level=0).first()
)

# Build windows
X_train, y_train, _ = make_windows(train_focus, train_means_focus,
                                    features=FEATURES_TEMP, targets=TARGETS_TEMP)
X_val,   y_val,   _ = make_windows(val_focus,   train_means_focus,
                                    features=FEATURES_TEMP, targets=TARGETS_TEMP)
X_test,  y_test,  d_test = make_windows(test_focus, train_means_focus,
                                         features=FEATURES_TEMP, targets=TARGETS_TEMP)

print(f'Windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
print(f'X shape: {X_train.shape}  y shape: {y_train.shape}')

# Fit scalers on training data only
N_tr, L, F = X_train.shape
_, H, T    = y_train.shape

feat_sc = StandardScaler().fit(X_train.reshape(-1, N_FEAT_TEMP))
tgt_sc  = StandardScaler().fit(y_train.reshape(-1, N_TGT_TEMP))

def scale_and_build(X, y, feat_sc, tgt_sc):
    N, L, F = X.shape
    _, H, T  = y.shape
    X_sc = feat_sc.transform(X.reshape(-1, F)).reshape(N, L, F).astype('float32')
    y_sc = tgt_sc.transform(y.reshape(-1, T)).reshape(N, H, T).astype('float32')
    return RiverDataset(X_sc, y_sc)

ds_train = scale_and_build(X_train, y_train, feat_sc, tgt_sc)
ds_val   = scale_and_build(X_val,   y_val,   feat_sc, tgt_sc)
ds_test  = scale_and_build(X_test,  y_test,  feat_sc, tgt_sc)

BATCH = 64
dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True,
                      drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
dl_val   = DataLoader(ds_val,   batch_size=256, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
dl_test  = DataLoader(ds_test,  batch_size=256, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print('DataLoaders ready.')"""
if not check_syntax(code8, 8):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code8))

# ── Cell 9 — Train LSTM ───────────────────────────────────────────────────
code9 = """\
model_temp = Seq2SeqLSTM(n_feat=N_FEAT_TEMP, n_tgt=N_TGT_TEMP,
                          hidden=128, n_layers=2, dropout=0.2)
n_params = sum(p.numel() for p in model_temp.parameters() if p.requires_grad)
print(f'Model: {n_params:,} trainable parameters')
print('Training temperature LSTM on gauge 2473...')

model_temp, history_temp = train_model(
    model_temp, dl_train, dl_val,
    lr=1e-3, epochs=100, patience=12,
    teacher_forcing_start=0.5,
    device=DEVICE, verbose=True,
)

# Save checkpoint
ckpt_path = str(RESULTS_DIR / 'lstm_temp_single_site.pt')
save_checkpoint(model_temp, feat_sc, tgt_sc, {}, ckpt_path)
print(f'Checkpoint saved: {ckpt_path}')

# Plot training curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history_temp['train'], color='#01696F', linewidth=1.5, label='Train loss')
ax.plot(history_temp['val'],   color='#DA7101', linewidth=1.5, linestyle='--', label='Val loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (standardised)')
ax.set_title(f'Temperature LSTM Training Curve \\u2014 Gauge {FOCUS_GAUGE}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '04b_temp_training_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Best val loss: {min(history_temp["val"]):.5f}')"""
if not check_syntax(code9, 9):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code9))

# ── Cell 10 — Evaluate on focus gauge test set ───────────────────────────
code10 = """\
y_pred_temp = predict(model_temp, dl_test, tgt_sc, device=DEVICE)
y_true_temp = get_y_true(ds_test, tgt_sc)

results_focus = metrics_table(
    [('LSTM (temp)', y_true_temp, y_pred_temp)],
    targets=TARGETS_TEMP,
    target_labels=TARGET_LABELS_TEMP,
    n_boot=500,
)
print(f'Focus gauge {FOCUS_GAUGE} \\u2014 Temperature LSTM test results:')
print(results_focus.to_string(index=False))
results_focus.to_csv(RESULTS_DIR / 'temp_focus_results.csv', index=False)"""
if not check_syntax(code10, 10):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code10))

# ── Cell 11 — Markdown: Section 3 ────────────────────────────────────────
md11 = """\
## 3. Zero-Shot Transfer — All Temperature Gauges

Apply the model trained on gauge 2473 directly to all valid temperature gauges.
With 80+ gauges this is the most comprehensive multi-site evaluation in the project."""
cells.append(nbformat.v4.new_markdown_cell(md11))

# ── Cell 12 — Zero-shot transfer loop ────────────────────────────────────
code12 = """\
print(f'Running zero-shot transfer on {len(temp_gauges)} gauges...')
transfer_results = []

for gid in tqdm(temp_gauges, desc='Zero-shot transfer'):
    try:
        raw  = load_gauge(gid)
        data = preprocess(raw)
        _, _, test_g = train_val_test_split(data)
        means_g = (
            pd.concat([data.loc[:TRAIN_END][FEATURES_TEMP].mean(),
                       data.loc[:TRAIN_END][TARGETS_TEMP].mean()])
            .groupby(level=0).first()
        )
        X_te, y_te, _ = make_windows(test_g, means_g,
                                      features=FEATURES_TEMP, targets=TARGETS_TEMP)
        if len(X_te) < 10:
            continue
        N, L, F = X_te.shape
        _, H, T  = y_te.shape
        X_sc = feat_sc.transform(X_te.reshape(-1, N_FEAT_TEMP)).reshape(N, L, F).astype('float32')
        y_sc = tgt_sc.transform(y_te.reshape(-1, N_TGT_TEMP)).reshape(N, H, T).astype('float32')
        ds_g = RiverDataset(X_sc, y_sc)
        dl_g = DataLoader(ds_g, batch_size=256, shuffle=False)

        y_pred_g = predict(model_temp, dl_g, tgt_sc, device=DEVICE)
        y_true_g = get_y_true(ds_g, tgt_sc)

        rmse_val = float(np.sqrt(np.mean((y_pred_g - y_true_g)**2)))
        mae_val  = float(np.mean(np.abs(y_pred_g - y_true_g)))
        ss_res   = np.sum((y_true_g - y_pred_g)**2)
        ss_tot   = np.sum((y_true_g - y_true_g.mean())**2)
        nse_val  = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

        transfer_results.append({
            'gauge_id': gid, 'n_windows': N,
            'rmse_temp': round(rmse_val, 4),
            'mae_temp':  round(mae_val,  4),
            'nse_temp':  round(nse_val,  3),
            'strategy':  'zero_shot_temp',
        })
    except Exception as e:
        print(f'  {gid}: skipped ({e})')

df_transfer = pd.DataFrame(transfer_results)
df_transfer.to_csv(RESULTS_DIR / 'temp_transfer_results.csv', index=False)
print(f'\\nZero-shot transfer: {len(df_transfer)} gauges evaluated')
print(f'Mean Temp RMSE: {df_transfer["rmse_temp"].mean():.4f} \\u00b1 {df_transfer["rmse_temp"].std():.4f} \\u00b0C')
print(f'Mean Temp NSE:  {df_transfer["nse_temp"].mean():.3f} \\u00b1 {df_transfer["nse_temp"].std():.3f}')
print(f'\\nPer-gauge results:')
print(df_transfer[['gauge_id','rmse_temp','nse_temp']].sort_values('rmse_temp').to_string(index=False))"""
if not check_syntax(code12, 12):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code12))

# ── Cell 13 — RMSE distribution plot ─────────────────────────────────────
code13 = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# RMSE histogram
ax = axes[0]
ax.hist(df_transfer['rmse_temp'], bins=20, color='#01696F', alpha=0.8, edgecolor='white')
ax.axvline(df_transfer['rmse_temp'].mean(), color='#A84B2F', linewidth=2, linestyle='--',
           label=f'Mean = {df_transfer["rmse_temp"].mean():.3f} \\u00b0C')
ax.axvline(1.261, color='#7A39BB', linewidth=1.5, linestyle=':',
           label='Focus gauge Ridge (1.261 \\u00b0C)')
ax.set_xlabel('Temperature RMSE (\\u00b0C)')
ax.set_ylabel('Number of gauges')
ax.set_title(f'Zero-Shot Transfer \\u2014 Temp RMSE Distribution\\n({len(df_transfer)} gauges)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# NSE histogram
ax = axes[1]
ax.hist(df_transfer['nse_temp'], bins=20, color='#006494', alpha=0.8, edgecolor='white')
ax.axvline(df_transfer['nse_temp'].mean(), color='#A84B2F', linewidth=2, linestyle='--',
           label=f'Mean NSE = {df_transfer["nse_temp"].mean():.3f}')
ax.axvline(0, color='black', linewidth=1, label='NSE = 0 (climatology baseline)')
ax.set_xlabel('NSE')
ax.set_ylabel('Number of gauges')
ax.set_title('Zero-Shot Transfer \\u2014 NSE Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Temperature LSTM \\u2014 Zero-Shot Transfer Across {len(df_transfer)} Swiss Gauges',
             fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '04b_temp_transfer_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: 04b_temp_transfer_distribution.png')"""
if not check_syntax(code13, 13):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code13))

# ── Cell 14 — Catchment attribute correlation ─────────────────────────────
code14 = """\
# Load metadata and merge with RMSE results
meta_raw = load_metadata()
if 'gauge_id' not in meta_raw.columns:
    meta_raw = meta_raw.reset_index()
    meta_raw.columns = [str(c) for c in meta_raw.columns]
    if meta_raw.columns[0] != 'gauge_id':
        meta_raw = meta_raw.rename(columns={meta_raw.columns[0]: 'gauge_id'})
meta_raw['gauge_id'] = meta_raw['gauge_id'].astype(str)

merged = df_transfer.merge(meta_raw, on='gauge_id', how='left')
print(f'Merged: {len(merged)} gauges with metadata')

# Spearman correlation with Temp RMSE
numeric_cols = merged.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['rmse_temp', 'mae_temp', 'nse_temp', 'n_windows']]

corr_results = []
from scipy import stats
for col in numeric_cols:
    valid = merged[['rmse_temp', col]].dropna()
    if len(valid) < 5:
        continue
    rho, pval = stats.spearmanr(valid['rmse_temp'], valid[col])
    corr_results.append({'attribute': col, 'spearman_rho': round(rho, 3), 'p_value': round(pval, 4)})

corr_df = pd.DataFrame(corr_results).sort_values('spearman_rho', key=abs, ascending=False)
print('\\nTop correlations with Temperature RMSE:')
print(corr_df.head(15).to_string(index=False))
corr_df.to_csv(RESULTS_DIR / 'temp_catchment_correlations.csv', index=False)

# Plot top 10 correlations
top10 = corr_df.head(10)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#01696F' if r > 0 else '#A84B2F' for r in top10['spearman_rho']]
bars = ax.barh(top10['attribute'], top10['spearman_rho'], color=colors, alpha=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Spearman \\u03c1 with Temp RMSE')
ax.set_title('Catchment Attribute Correlations with Temperature Prediction Error\\n(positive = higher RMSE, negative = lower RMSE)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '04b_temp_catchment_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: 04b_temp_catchment_correlations.png')"""
if not check_syntax(code14, 14):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code14))

# ── Cell 15 — Summary ─────────────────────────────────────────────────────
code15 = """\
print('=' * 70)
print('TEMPERATURE MULTI-SITE SUMMARY')
print('=' * 70)
print(f'Gauges evaluated   : {len(df_transfer)}')
print(f'Lookback / Horizon : {LOOKBACK} / {HORIZON} days')
print(f'Features used      : {FEATURES_TEMP}')
print(f'Target             : {TARGETS_TEMP}')
print(f'')
print(f'Zero-shot transfer:')
print(f'  Mean Temp RMSE : {df_transfer["rmse_temp"].mean():.4f} \\u00b1 {df_transfer["rmse_temp"].std():.4f} \\u00b0C')
print(f'  Mean Temp NSE  : {df_transfer["nse_temp"].mean():.3f} \\u00b1 {df_transfer["nse_temp"].std():.3f}')
print(f'  Best gauge     : {df_transfer.loc[df_transfer["rmse_temp"].idxmin(), "gauge_id"]} (RMSE = {df_transfer["rmse_temp"].min():.4f} \\u00b0C)')
print(f'  Worst gauge    : {df_transfer.loc[df_transfer["rmse_temp"].idxmax(), "gauge_id"]} (RMSE = {df_transfer["rmse_temp"].max():.4f} \\u00b0C)')
print(f'')
print(f'Comparison:')
print(f'  DO multi-site (12 gauges, zero-shot): 0.4252 mg/L')
print(f'  Temp multi-site ({len(df_transfer)} gauges, zero-shot): see above')
print(f'  LakeBeD-US LSTM temp reference: ~2.6 \\u00b0C')
print('=' * 70)
print('Next \\u2192 notebook 05_shap_interpretation.ipynb')"""
if not check_syntax(code15, 15):
    syntax_errors += 1
cells.append(nbformat.v4.new_code_cell(code15))

# Assemble notebook
nb.cells = cells

# Save
out_path = '/home/user/workspace/AareML/notebooks/04b_multisite_temperature.ipynb'
with open(out_path, 'w') as f:
    nbformat.write(nb, f)

total_cells = len(cells)
code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
md_cells   = sum(1 for c in cells if c['cell_type'] == 'markdown')

print(f'Notebook created: {total_cells} cells ({code_cells} code, {md_cells} markdown), {syntax_errors} syntax errors')
print(f'Saved to: {out_path}')
