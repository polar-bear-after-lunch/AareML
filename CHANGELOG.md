# AareML — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v1.22] — 2026-04-24

### New Notebook
- **`notebooks/08_usgs_transfer.ipynb`** — Cross-continental zero-shot transfer: Swiss LSTM checkpoint applied to 5 US rivers (Potomac, Willamette, Mississippi, Fox, Missouri) via USGS NWIS continuous monitoring data. Features mapped from USGS parameter codes to AareML naming convention; missing pH/EC channels zero-padded (conservative lower bound). Saves `results/usgs_transfer_results.csv` and figures `08_usgs_transfer.png`, `08_usgs_horizon_rmse.png`.
- **`src/config.py`** — Added `USGS_SITES`, `USGS_COL_MAP`, `USGS_FEATURES`, `USGS_TARGET`
- **`run_all_notebooks.sh`** — Added notebook 08 to run sequence

---

## [v1.21] — 2026-04-24

### Model Improvement
- **`src/model.py`** — Replaced `nn.MSELoss` with combined NSE+MSE loss (`alpha=0.5`) in `train_model`. The NSE component (`mse / var(target)`) penalises distributional mismatch, expected to improve KGE and reduce the RMSE gap vs Ridge regression.

### Report v1.9
- Updated single-site results from v1.20 run (LSTM best DO RMSE=0.319, KGE=0.942)
- Updated multi-site DO means (transfer=0.427, per-gauge=0.388, EA-LSTM=0.417 mg/L)
- Corrected gauge 2068 temperature RMSE (3.16 → 1.36°C)
- Added `04_multisite_map.png` and `04_multisite_rmse_comparison.png` to Section 5.3
- Added KGE trade-off explanation sentence in Section 5.2

---

## [v1.20] — 2026-04-23

### Hyperparameter Improvements (3-model consensus: Sonnet, GPT-5.4, Opus)
- **NB03** — Default model: 100→120 epochs, patience 12→15
- **NB03** — Optuna: 50→75 trials, 30→40 epochs per trial, patience 5→8
- **NB03** — Best model: 150→250 epochs, patience 15→25
- **NB03** — Added `teacher_forcing_start` [0.3–0.7] to Optuna search space
- **NB03** — Added 3-seed ensemble (seeds 0, 42, 123) for best model
- **NB04** — Per-gauge retrain: 80→150 epochs, patience 10→20
- **NB04b** — Reduced 100→60 epochs, patience 12→10 (transfer learning converges faster; prevents catastrophic forgetting)
- **NB04b** — Results saved with version tag (`v1.20_60ep`) for comparison with original 100-epoch run

### Quality Improvements
- **`src/model.py`** — Added `ReduceLROnPlateau` (factor=0.5, patience=5) to `train_model` and `train_ea_model` — expected ~5–10% further RMSE reduction
- **NB02** — Added per-gauge Ridge RMSE save to `results/baseline_per_gauge.csv` for use in NB04 significance test
- **NB04** — Fixed `KeyError: 'gauge_id'` in Wilcoxon test cell — `baseline_results.csv` is model-level, now loads from `baseline_per_gauge.csv` with hardcoded fallback

### Infrastructure
- **NB01** — Fixed 4 figure filenames missing `f` prefix (`{FOCUS_GAUGE}` was literal in saved PNG filenames)
- **`fetch_from_ubelix.sh`** — Now also fetches `notebooks/` folder so executed notebook outputs are pulled back locally
- **`ubelix/job_04b_temp.sh`** — New SLURM job script for notebook 04b temperature multi-site
- **`ubelix/run_all.sh`** — Full chain now 03 → 04 → 04b → 05 (was missing 04b)
- **`ubelix/`** — All job scripts now use absolute path `/storage/homefs/tn20y076/AareML` as working directory

---

## [v1.19] — 2026-04-22

### Bug Fixes
- SLURM log paths changed to absolute (`/storage/homefs/tn20y076/AareML/logs/`) — relative paths caused logs to not be created
- `job_04_multisite.sh`, `job_05_shap.sh` — Fixed checkpoint filename `best_model.pt` → `lstm_single_site_best.pt`
- `fetch_from_ubelix.sh` — Added `notebooks/` to fetch list
- NB01 — Fixed 4 figure filenames with literal `{FOCUS_GAUGE}` (missing `f` prefix)
- `job_04b_temp.sh` — New SLURM script for temperature multi-site notebook
- `run_all.sh` — Added 04b to job chain

---

## [v1.18] — 2026-04-22

### Bug Fixes (Round 3 Audit — 13 fixes)
- **C1** NB04b — `predict()` called with DataLoader instead of Dataset in Cells 10 and 12 — crashes immediately; fixed to pass `ds_test`/`ds_g`
- **C2** NB04b — `metrics_table()` called with wrong API (list-of-tuples, missing `y_true`, invalid `target_labels` kwarg); fixed to dict API
- **C3** NB04b — Test-split means used for gauge screening imputation (data leakage); fixed to use training-split means
- **C4** `src/model.py` — `train_model` and `train_ea_model` crash with `TypeError` if `best_state` is None (val loss never improved); added guard with clear RuntimeError
- **M1** NB05 — Strategy filter `transfer` → `transfer_normed` to match NB04 output
- **M2** NB04b — FOCUS_GAUGE included in zero-shot transfer (in-sample contamination); now excluded
- **M3** `src/metrics.py` — Bootstrap off-by-one: `N - block_size` → `N - block_size + 1`
- **M4** `src/metrics.py` — Bootstrap degenerates when N ≤ block_size; added adaptive block_size reduction
- **M5** `src/metrics.py` — KGE bias ratio blows up near 0°C; epsilon guard (`abs(obs.mean()) > 1e-3`)
- **M6** `src/model.py` — `drop_last=True` can silently produce empty training loader; added explicit check
- **M7** NB04 — Wilcoxon test used hardcoded RMSE values; now computed from actual result DataFrames
- **M8** NB04b — Silent exception swallowing in transfer loop; improved error logging + empty-result guard

### Infrastructure
- UBELIX job scripts: switched GPU from `rtx3090` (blocked on gratis QoS) to `rtx4090` (allowed)
- Jobs 03 → 04 → 05 now run sequentially (one GPU at a time) to stay within gratis per-user GPU limit

---

## [v1.17] — 2026-04-21

### Bug Fixes
- **Notebook 04b** — Fixed `save_checkpoint` call with wrong argument order. Was `save_checkpoint(model_temp, feat_sc, tgt_sc, {}, ckpt_path)` (path last); corrected to `save_checkpoint(ckpt_path, model_temp, {}, feat_sc, tgt_sc)` (path first, matching signature). This caused `AttributeError: 'StandardScaler' object has no attribute 'state_dict'` at runtime on UBELIX.

### Other
- `download_data.py` — Final summary message now lists correct notebook run order: `01 → 02 → 03 → 04 → 04b → 05 → 06 → 07` (was missing 04b and 07).
- `sync_to_ubelix.sh` — New convenience script to rsync code to UBELIX, excluding data, results, figures, PDFs, and zips.

---

## [v1.16] — 2026-04-20

### Added
- `notebooks/04b_multisite_temperature.ipynb` — Temperature multi-site analysis across all 86 Swiss gauges (vs 12 for DO). Uses [temp, pH, EC] as features (no DO to avoid target leakage). Covers: gauge discovery, single-site LSTM training, zero-shot transfer to 80+ gauges, catchment attribute correlation, RMSE distribution.
- `src/config.py` — Added `FEATURES_TEMP`, `TARGETS_TEMP`, `N_FEAT_TEMP`, `N_TGT_TEMP`, `TARGET_LABELS_TEMP`, `TEMP_MIN_COVERAGE` for temperature-only analysis.
- Report Section 5.3b — Temperature Multi-Site Analysis placeholder (pending UBELIX execution).
- Report Section 1 — Temperature critical thresholds added (18°C/21°C/25°C).
- Report Section 6.1 — Three-mechanism explanation of river-lake predictability gap (autocorrelation structure, physical range, feature-space mismatch).
- Report Appendix E — Glossary (25 terms: hydrology, ML, dataset/evaluation).

---

## [v1.15] — 2026-04-20

### Fixed (remaining major + minor bugs)
- **G-U1** `src/config.py` — `STATIC_COLS` now lists both naming conventions (CAMELS-CH and legacy) with runtime deduplication — avoids `KeyError` when metadata columns differ.
- **G-U2/U3** NB01, NB06, NB07 — `../figures` and `../data` paths now auto-detect repo root — works both locally and in Colab after `os.chdir`.
- **S-U1** NB05 Cell 14 — SHAP heatmap x-axis label was backwards (showed oldest lag as most recent) — flipped array and corrected labels to "t-1 = yesterday".
- **S-U2** NB04 Cell 24 — `groupby` used land-cover value columns instead of `gauge_id` — fixed to `groupby('gauge_id')`.
- **O-U1** NB04 Cell 30 — EA-LSTM trained but never evaluated — added full per-gauge evaluation loop, saves `results/ea_lstm_results.csv`.
- **O-U2** NB04 — "transfer" strategy renamed `transfer_normed` with clarifying comment (per-gauge scalers, not true zero-shot).
- **O-U3** NB05 — SHAP worst-window index now bounded to `[:n_explain]` to avoid showing wrong window.
- **O-U4** NB03 — Graceful fallback when `baseline_rmse_by_horizon.csv` not found.
- **O-U5** `src/data.py` — `train_val_test_split` uses strict-inequality slicing — no longer drops boundary day.
- **O-U7** NB03 — All hardcoded `"Gauge 2473"` strings replaced with `f"Gauge {FOCUS_GAUGE}"`.
- **S-U3** `src/model.py` — `reconstruct_scalers` sets `n_samples_seen_ = 10000` instead of `1`.

### Added
- `tests/test_src.py` — 53 pytest tests covering all 5 src/ modules + 2 integration tests. All 53 tests pass (2 expected RuntimeWarnings for flat-prediction KGE edge case).

---

## [v1.14] — 2026-04-19

### Fixed (3-model consensus bugs — Round 2 audit)
- **C1** `src/impute.py` — `SATSImputer.transform()` used feature-0-only mask; now uses `.all(axis=-1)` matching `fit()` behaviour.
- **C2** `src/model.py` — Removed dead `static_proj` module from `EASeq2SeqLSTM` (was defined but never called, polluting optimizer state).
- **C3** `notebooks/03` — Optuna `MedianPruner` now receives per-epoch intermediate values (was reported only once after full training, making pruner a no-op).
- **C4** `notebooks/03` — Added clarifying comment to `scale_split` variables (incorrectly flagged as dead code — they feed DataLoaders below).
- **C5** `src/data.py` — Removed `bfill()` from `merge_nawaf_features` — backward fill leaks future chemistry values; replaced with `ffill()` only.
- **C6** `notebooks/04` — Added comment explaining EA-LSTM uses focus-gauge scalers intentionally for zero-shot transfer consistency.

### Quality
- 3-model independent bug audits (Sonnet 4.6, GPT-5.4, Opus 4.7) before and after fixes.
- Round 1: 11/33/30 bugs found → Round 2: 11/10/14 bugs (70%/53% reduction for GPT/Opus).
- All 126 notebook cells + 5 src/ files pass syntax check.

---

## [v1.13] — 2026-04-19

### Fixed (11 bugs)
- Bug #1 (×7): `os.chdir('AareML')` now conditional in all Colab setup cells.
- Bug #2+3+10: Duplicated-index `train_means` in notebook 04 cells 9, 12, 14 — fixed with `pd.concat().groupby(level=0).first()`.
- Bug #4: Climatology DOY off-by-one in notebook 06 (`days=h+1` → `days=h`).
- Bug #5: Notebook 05 scalers refitted instead of reconstructed from checkpoint — now uses `reconstruct_scalers(ckpt)`.
- Bug #6: Dead `tgt_idx` variable removed from `make_lake_windows` (notebook 06).
- Bug #7: `feat_scaler` fit on windowed data in notebook 06 — now fit on raw daily training data.
- Bug #8: Int vs str index mismatch in EA-LSTM cell (notebook 04) — `meta.index.astype(str)` added.
- Bug #9: `'var' in dir()` → `'var' in globals()` in notebook 03.
- Bug #11: Duplicate `DEVICE` definition merged in notebook 03.
- CPU thread count clamped to 6 (empirically optimal for macOS) — was using all logical cores causing slowdown.
- Model tracking added to effort log — "Which model are you?" at session start.

---

## [v1.12] — 2026-04-16

### Added
- Google Drive caching in all Colab setup cells — data persists across sessions.
- Adaptive Optuna trials: 50 on GPU, 20 on CPU (notebook 03).
- Adaptive SHAP windows: 2000 on GPU, 500 on CPU (notebook 05).
- GitHub repo link added to report Appendix B and C.
- `data/lakebed-us/` and `data/*.parquet` added to `.gitignore`.

### Fixed
- `os.chdir('AareML')` unconditional in Colab cells → guarded with path check.

---

## [v1.11] — 2026-04-15

### Added
- `notebooks/07_lake_eda.ipynb` — Lake Mendota EDA mirroring notebook 01 structure: full time series, seasonal cycles, distribution comparison, autocorrelation analysis confirming river DO is more autocorrelated at all lags 1–21 days, coverage comparison (river 97% vs lake 51%), summary comparison table (12 properties).
- Report Appendix D — Report Version History (6-entry scientific changelog).
- Colab badge added to all 7 notebooks.
- tqdm progress bars added to long loops in notebooks 02, 04, 05, 06.
- CPU thread optimisation cell added to all notebooks.

---

## [v1.10] — 2026-04-15

### Results
- Real SHAP results (notebook 05, GradientSHAP, 500 windows):
  - `temp_sensor[t−1]`: dominant driver (mean |SHAP| = 0.644)
  - `O2C_sensor[t−1]`: second driver (mean |SHAP| = 0.527)
  - Effective LSTM memory = 3–4 days despite 21-day lookback.
  - pH and EC contribute negligibly.
- Added report Section 5.4 (SHAP Attribution Results).
- Added `download_data.py` — one script downloads all datasets (~360 MB).
- Updated README with full setup instructions and results table.
- Added `data/lakebed-us/` to `.gitignore`.

---

## [v1.9] — 2026-04-14

### Results
- Real multi-site results (notebook 04, 12 gauges):
  - Zero-shot transfer: DO RMSE = 0.425 ± 0.083 mg/L (3.3× better than LakeBeD-US).
  - Per-gauge retrain: DO RMSE = 0.386 ± 0.092 mg/L (3.6× better).
  - Gauge latitude/northing: strongest catchment predictor (Spearman ρ = 0.78, p = 0.005).
- Fixed `feat_scaler` undefined error in EA-LSTM cell (notebook 04).
- Updated report Section 5.3 with full 24-row results table.

---

## [v1.8] — 2026-04-13

### Quality
- Applied 30 peer-review fixes including:
  - Corrected abstract RMSE misattribution (default vs Optuna best).
  - Fixed table numbering (Table 4 = full model comparison, Table 6 = Lake Mendota).
  - Added CI overlap discussion — top 3 models statistically indistinguishable.
  - Expanded acronyms: LSTM, HBV, TPE, NAWA on first use.
  - Corrected Zhi et al. (2021): predicts DO not nitrate.
  - Added teacher forcing formula p(e) = max(0, 0.5 − e/N).
  - Added statistical significance caveat.
  - Added confounders paragraph for cross-ecosystem comparison.
  - Two new limitation bullets: single seed, statistical significance.

---

## [v1.7] — 2026-04-12

### Results
- Real LSTM single-site results (notebook 03, 20 Optuna trials, ~10h CPU):
  - LSTM (default): DO RMSE = 0.299 mg/L, KGE = 0.918.
  - LSTM (best, hidden=256): DO RMSE = 0.301 mg/L, KGE = 0.927.
  - All models 4.7× better than LakeBeD-US LSTM reference (1.40 mg/L).
- Fixed notebook 04 `gauge_id` KeyError and `DO_GAUGES` undefined error.
- Updated report Section 5.2 and abstract with real numbers.

---

## [v1.6] — 2026-04-11

### New Content
- **Notebook 06** `notebooks/06_cross_ecosystem_lake.ipynb` — Full reproducible cross-ecosystem experiment notebook (14 cells, executed cleanly). Loads pre-processed Lake Mendota data, runs all three baselines with 80/10/10 chronological split, generates comparison figures.
- Report Section 5.4 — "Cross-Ecosystem Experiment: Lake Mendota" added with Table 4 and Figure 8.
- Report Section 6.1 updated — Discussion now cites 3.4× and 4.6× quantitative gaps with reference to notebook 06.
- `figures/06_mendota_timeseries.png` — Lake Mendota DO and temperature full time series.
- Du et al. (2023) reference added to report (SAITS imputation paper).
- Effort log updated — 14 sessions, 41.5 hours logged (34.6% of 120-hour budget).
- Russian report updated with all new content translated.

---

## [v1.5] — 2026-04-11

### New Features
- **Cross-ecosystem experiment** — Downloaded LakeBeD-US Lake Mendota high-frequency data (101M rows) from Hugging Face, processed to daily surface observations (3,511 days, 2006–2023), and ran all three AareML baselines (persistence, climatology, Ridge) on the lake test set.
- `results/lake_mendota_results.csv` — Full results table with RMSE, MAE, NSE.
- `figures/06_lake_mendota_rmse_by_horizon.png` — Per-horizon RMSE for all three baselines on Lake Mendota.
- `figures/06_river_vs_lake_comparison.png` — Direct River vs Lake RMSE comparison with LakeBeD-US LSTM reference line.
- **Key finding:** Lake Ridge DO RMSE = 1.030 mg/L vs River Ridge = 0.303 mg/L (3.4× gap). AareML Ridge on lake data already beats the published LakeBeD-US LSTM (1.40 mg/L).

---

## [v1.4] — 2026-04-08

### Developer Experience
- **Debug asserts** added throughout all `src/` modules — every critical shape, dtype, range, and temporal invariant is now explicitly checked with an informative error message.
- **Debug print checkpoints** (`if __debug__:` guards) added at every major pipeline stage — loading, preprocessing, splitting, windowing, training, predicting, imputing. Run with `python -O` to silence all debug output in production.
- `src/data.py` — asserts in `load_gauge`, `preprocess`, `train_val_test_split`, `make_windows`.
- `src/model.py` — asserts in `RiverDataset.__init__`, `train_model`, `predict`, `save_checkpoint`, `load_checkpoint`.
- `src/metrics.py` — asserts in `mean_rmse`, `mean_mae`, `metrics_table`.
- `src/impute.py` — asserts in `SATSImputer.fit` and `SATSImputer.transform`.

---

## [v1.3] — 2026-04-07

### New Features
- **I3** `src/impute.py` — New SAITS-inspired self-attention imputer (`SATSImputer`). Trains a single-layer multi-head attention model on observed timesteps using 15% random masking (BERT-style). `fit()` / `transform()` / `fit_transform()` API. Reference: Du et al. (2023).
- **I5** `notebooks/03_lstm_single_site.ipynb` — Added per-horizon RMSE curve cell (days 1–14).
- **I6** `src/model.py` — Added `EALSTMCell`, `EASeq2SeqLSTM`, `EARiverDataset`, and `train_ea_model`. Entity-Aware LSTM (Kratzert et al. 2019) incorporates static catchment attributes.
- **I7** `src/data.py` + `src/config.py` — Added `load_nawaf()` and `merge_nawaf_features()` for NAWA FRACHT monthly chemistry features. Monthly data is forward-filled to daily resolution without leaking future values.
- **I8** `src/config.py` — Added extended target support: `USE_EXTENDED_TARGETS` flag (default False), `TARGETS_EXTENDED` (DO + temp + pH + EC).

---

## [v1.2] — 2026-04-07

### Bug Fixes
- **B1** `src/model.py` — `RiverDataset` now stores one stacked tensor instead of a Python list. Faster `__getitem__`, lower memory fragmentation.
- **B2** `src/model.py` — `get_y_true` reads stored tensor directly instead of looping — ~100× faster on large datasets.
- **B3** `src/model.py` — `load_checkpoint` now passes `weights_only=False` to suppress the PyTorch 2.x security warning.
- **B4** `src/data.py` — `make_windows` raises a clear `ValueError` when it produces 0 valid windows.
- **B5** `notebooks/03` — Removed redundant second `make_windows` call and unused `y_tr_win` variable.
- **B6** `src/metrics.py` — `kge()` now guards all three components against degenerate predictions.

### New Features
- **F1** `src/model.py` — Added `predict_single_window(model, x_raw, feat_scaler, tgt_scaler)`.
- **F2** `src/metrics.py` — `metrics_table` now computes `MAE_lo` / `MAE_hi` bootstrap confidence intervals.
- **F3** `notebooks/03` — Optuna study persisted to `results/optuna_study.db` with `load_if_exists=True`.
- **F4** `src/data.py` — Added `score_gauge()` helper for the multi-site loop.
- **F5** `src/model.py` — Added `reconstruct_scalers(ckpt)` to rebuild scalers from a checkpoint.
- **F6** `requirements.txt` — All package versions pinned to confirmed-working set (Python 3.11, April 2026).

---

## [v1.1] — 2026-04-07

### Added
- `AareML-report.pdf` — 13-page project report (cover, abstract, intro, related work, data, methods, results, discussion, conclusion, references, appendix).
- `AareML-effort-log.pdf` — Project effort tracking log (30.5 hours logged, 120-hour budget).

---

## [v1.0] — 2026-04-06

### Added
- Initial project scaffold with full notebook suite (01–05) and shared `src/` module.
- `notebooks/01_data_exploration.ipynb` — EDA with 14 figures.
- `notebooks/02_baselines.ipynb` — Persistence, climatology, Ridge regression with block-bootstrap CIs. Real results: Ridge DO RMSE = 0.303 mg/L, NSE = 0.888, KGE = 0.908.
- `notebooks/03_lstm_single_site.ipynb` — Seq2Seq LSTM with Optuna tuning (20 trials), teacher forcing, early stopping.
- `notebooks/04_multisite_analysis.ipynb` — Zero-shot transfer + per-gauge retraining across 16 DO gauges.
- `notebooks/05_shap_interpretation.ipynb` — GradientSHAP + catchment-level GBM surrogate.
- `src/config.py` — Shared configuration (LOOKBACK=21, HORIZON=14, FOCUS_GAUGE='2473', splits).
- `src/data.py` — Data loading, preprocessing, windowing utilities.
- `src/metrics.py` — RMSE, MAE, NSE, KGE, block bootstrap CI, metrics_table.
- `src/model.py` — RiverDataset, Seq2SeqLSTM, train_model, predict, checkpoint helpers.
- `results/baseline_results.csv` — Real baseline results from notebook 02.
- `figures/` — 17 PNG figures from notebooks 01–03.
