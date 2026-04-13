# AareML — Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v1.5] — 2026-04-11

### New Features
- **Cross-ecosystem experiment** — Downloaded LakeBeD-US Lake Mendota high-frequency data (101M rows) from Hugging Face, processed to daily surface observations (3,511 days, 2006–2023), and ran all three AareML baselines (persistence, climatology, Ridge) on the lake test set.
- `results/lake_mendota_results.csv` — Full results table with RMSE, MAE, NSE.
- `figures/06_lake_mendota_rmse_by_horizon.png` — Per-horizon RMSE for all three baselines on Lake Mendota.
- `figures/06_river_vs_lake_comparison.png` — Direct River vs Lake RMSE comparison with LakeBeD-US LSTM reference line.
- **Key finding:** Lake Ridge DO RMSE = 1.030 mg/L vs River Ridge = 0.303 mg/L (3.4× gap). The AareML Ridge baseline on lake data already beats the published LakeBeD-US LSTM (1.40 mg/L), suggesting Ridge regression is a surprisingly strong baseline when sufficient sensor history is available.

---

## [v1.4] — 2026-04-08

### Developer Experience
- **Debug asserts** added throughout all `src/` modules — every critical shape, dtype, range, and temporal invariant is now explicitly checked with an informative error message.
- **Debug print checkpoints** (`if __debug__:` guards) added at every major pipeline stage — loading, preprocessing, splitting, windowing, training, predicting, imputing. Run with `python -O` to silence all debug output in production.
- `src/data.py` — asserts in `load_gauge`, `preprocess`, `train_val_test_split`, `make_windows`
- `src/model.py` — asserts in `RiverDataset.__init__`, `train_model` (start + per-epoch loss finiteness), `predict`, `save_checkpoint`, `load_checkpoint`
- `src/metrics.py` — asserts in `mean_rmse`, `mean_mae`, `metrics_table`
- `src/impute.py` — asserts in `SATSImputer.fit` and `SATSImputer.transform`

---

## [v1.3] — 2026-04-07

### New Features
- **I3** `src/impute.py` — New SAITS-inspired self-attention imputer (`SATSImputer`). Trains a single-layer multi-head attention model on observed timesteps using 15% random masking (BERT-style). `fit()` / `transform()` / `fit_transform()` API. Handles patchy DO coverage at non-focus gauges far better than mean imputation. Reference: Du et al. (2023), SAITS, Expert Systems with Applications.
- **I5** `notebooks/03_lstm_single_site.ipynb` — Added per-horizon RMSE curve cell (days 1–14) after the Optuna results section, so LSTM and baseline degradation can be compared directly across the full horizon.
- **I6** `src/model.py` — Added `EALSTMCell`, `EASeq2SeqLSTM`, `EARiverDataset`, and `train_ea_model`. Entity-Aware LSTM (Kratzert et al. 2019) incorporates static catchment attributes into the input and cell LSTM gates, enabling a single model to adapt to multiple gauges simultaneously. `notebooks/04_multisite_analysis.ipynb` updated with EA-LSTM multi-site training loop.
- **I7** `src/data.py` + `src/config.py` — Added `load_nawaf()` and `merge_nawaf_features()` for NAWA FRACHT monthly chemistry features (NO3_N, NH4_N, TP, TN, DOC, Q_m3s). `NAWAF_FEATURES` and `FEATURES_WITH_NAWAF` added to config. Monthly data is forward/backward-filled to daily resolution without leaking future values.
- **I8** `src/config.py` — Added extended target support: `USE_EXTENDED_TARGETS` flag (default False), `TARGETS_EXTENDED` (DO + temp + pH + EC), `ACTIVE_TARGETS` / `ACTIVE_TARGET_LABELS` computed from flag. Flip `USE_EXTENDED_TARGETS = True` to switch the whole pipeline to 4-target mode.

---

## [v1.2] — 2026-04-07

### Bug Fixes
- **B1** `src/model.py` — `RiverDataset` now stores one stacked tensor instead of a Python list of per-sample tensors. Faster `__getitem__`, lower memory fragmentation.
- **B2** `src/model.py` — `get_y_true` reads the stored tensor directly instead of looping sample-by-sample with `torch.stack`. ~100x faster on large datasets.
- **B3** `src/model.py` — `load_checkpoint` now passes `weights_only=False` to suppress the PyTorch 2.x security warning.
- **B4** `src/data.py` — `make_windows` raises a clear `ValueError` when it produces 0 valid windows, instead of returning an empty array that causes a confusing crash later.
- **B5** `notebooks/03_lstm_single_site.ipynb` — Removed redundant second `make_windows` call and the unused `y_tr_win` variable that doubled windowing time.
- **B6** `src/metrics.py` — `kge()` now guards all three components (r, β, γ) against degenerate predictions. Flat predictions (e.g. persistence at long horizons) no longer produce silent `nan` values in the results table.

### New Features
- **F1** `src/model.py` — Added `predict_single_window(model, x_raw, feat_scaler, tgt_scaler)` — forecast from a single raw `[lookback, n_feat]` numpy array in one call, without building a full dataset.
- **F2** `src/metrics.py` — `metrics_table` now computes `MAE_lo` / `MAE_hi` bootstrap confidence intervals alongside the existing RMSE CIs.
- **F3** `notebooks/03_lstm_single_site.ipynb` — Optuna study is now persisted to `results/optuna_study.db`. If the kernel is interrupted mid-tuning, the study resumes from where it left off (`load_if_exists=True`).
- **F4** `src/data.py` — Added `score_gauge(gauge_id, feat_scaler, tgt_scaler, predict_fn, ...)` helper that runs the full load → preprocess → split → window → scale → predict pipeline for one gauge. Used in notebook 04 multi-site loop.
- **F5** `src/model.py` — Added `reconstruct_scalers(ckpt)` that rebuilds `StandardScaler` objects from a saved checkpoint dict, so scalers don't need to be manually refitted after loading a model.
- **F6** `requirements.txt` — All package versions are now pinned to the confirmed-working set (Python 3.11, April 2026). Added note about conda install for llvmlite/numba.

### Other
- `.gitignore` — Added `results/optuna_study.db` (auto-regenerated, can grow large).

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

---

## [v1.6] — 2026-04-11

### New Content
- **Notebook 06** `notebooks/06_cross_ecosystem_lake.ipynb` — Full reproducible cross-ecosystem experiment notebook (14 cells, executed cleanly). Loads pre-processed Lake Mendota data, runs all three baselines with 80/10/10 chronological split, generates comparison figures.
- **Report Section 5.4** — "Cross-Ecosystem Experiment: Lake Mendota" added to results section with two descriptive paragraphs, Table 4 (Lake Mendota baseline results), and Figure 8 (river vs. lake RMSE comparison).
- **Report Section 6.1 updated** — Discussion now cites 3.4× and 4.6× quantitative gaps with reference to notebook 06.
- **Figure** `figures/06_mendota_timeseries.png` — Lake Mendota DO and temperature full time series.
- **Du et al. (2023) reference** added to report (SAITS imputation paper).
- **Effort log updated** — 14 sessions, 41.5 hours logged (34.6% of 120-hour budget).
- **Russian report updated** with all new content translated.
