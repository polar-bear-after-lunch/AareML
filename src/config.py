"""
AareML — shared configuration.
All notebooks import from here so settings stay in sync.
"""
from pathlib import Path

# ── Repo layout ────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent
REPO_ROOT    = SRC_DIR.parent
DATA_ROOT    = REPO_ROOT / "data" / "camels-ch-chem"

DAILY_DIR    = DATA_ROOT / "stream_water_chemistry/timeseries/daily"
HOURLY_DIR   = DATA_ROOT / "stream_water_chemistry/timeseries/hourly"
NAWAF_DIR    = DATA_ROOT / "stream_water_chemistry/interval_samples/nawa_fracht"
NAWAT_DIR    = DATA_ROOT / "stream_water_chemistry/interval_samples/nawa_trend"
LANDCOVER_DIR = DATA_ROOT / "catchment_aggregated_data/landcover_data"
METADATA_FILE = DATA_ROOT / "gauges_metadata/camels_ch_chem_gauges_metadata.csv"

FIGURES_DIR  = REPO_ROOT / "figures"
RESULTS_DIR  = REPO_ROOT / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Task hyper-parameters (LakeBeD-US benchmark setup) ────────────────────
LOOKBACK   = 21     # days of input history
HORIZON    = 14     # days to forecast
FEATURES   = ["temp_sensor", "pH_sensor", "ec_sensor", "O2C_sensor"]
TARGETS    = ["O2C_sensor", "temp_sensor"]
TARGET_LABELS = {"O2C_sensor": "DO (mg/L)", "temp_sensor": "Temp (°C)"}
N_FEAT     = len(FEATURES)
N_TGT      = len(TARGETS)

# ── Extended targets (I8: add pH and EC) ──────────────────────────────────
# Set USE_EXTENDED_TARGETS = True to predict DO, temp, pH, and EC
USE_EXTENDED_TARGETS = False   # False = default (DO + temp only, LakeBeD-US compatible)

TARGETS_EXTENDED   = ["O2C_sensor", "temp_sensor", "pH_sensor", "ec_sensor"]
TARGET_LABELS_EXTENDED = {
    "O2C_sensor":  "DO (mg/L)",
    "temp_sensor": "Temp (°C)",
    "pH_sensor":   "pH",
    "ec_sensor":   "EC (µS/cm)",
}

# Active targets — switch by setting USE_EXTENDED_TARGETS above
ACTIVE_TARGETS       = TARGETS_EXTENDED if USE_EXTENDED_TARGETS else TARGETS
ACTIVE_TARGET_LABELS = TARGET_LABELS_EXTENDED if USE_EXTENDED_TARGETS else TARGET_LABELS
N_TGT_EXTENDED       = len(TARGETS_EXTENDED)

# ── NAWA FRACHT chemistry features (I7) ───────────────────────────────────
# NAWA FRACHT column names — verify against actual CSV headers in
# data/camels-ch-chem/nawa_fracht/
# Run: pd.read_csv('data/camels-ch-chem/nawa_fracht/<any_file>.csv').columns.tolist()
# Both audit reports (GPT-5.4 Bug #1 and Opus 4.7 Bug #3) confirmed the actual
# CSV files use lowercase names and q_mean_sensor (not Q_m3s), and NH4_N does
# not exist in the dataset.
NAWAF_FEATURES = [
    "NO3_N",          # nitrate-N — verify exact column name in CSV
    "tp",             # total phosphorus — actual column name in CSV
    "tn",             # total nitrogen — actual column name in CSV
    "doc",            # dissolved organic carbon — actual column name in CSV
    "q_mean_sensor",  # discharge — actual column name in CSV (not "Q_m3s")
]
# To verify: run load_nawaf('2473') and print the columns

# Features with NAWA FRACHT appended (used when NAWA data is available)
FEATURES_WITH_NAWAF = FEATURES + NAWAF_FEATURES

# ── Static catchment attribute columns for EA-LSTM (I6) ───────────────────
# These are CANDIDATE names — actual availability checked at runtime in notebook 04.
# G-U1 fix: the original list contained names that don't exist in the metadata CSV.
# Use both common naming conventions; notebook 04 filters to whichever exist.
# To verify, run:
# pd.read_csv('data/camels-ch-chem/gauges_metadata/camels_ch_chem_gauges_metadata.csv').columns.tolist()
# Common CAMELS-CH column names (try both conventions):
STATIC_COLS = [
    "area",           "area_km2",          # catchment area
    "ele_mt_smn",     "mean_elev",         # mean elevation
    "slp_dg_sav",     "slope_mean",        # mean slope
    "for_pc_sse",     "forest_frac",       # forest %
    "crp_pc_sse",     "agriculture_frac",  # cropland %
]
# Deduplicate while preserving order
STATIC_COLS = list(dict.fromkeys(STATIC_COLS))

# ── Train / val / test split boundaries ───────────────────────────────────
TRAIN_END  = "2014-12-31"
VAL_END    = "2016-12-31"
# Test: 2017-01-01 onward

# ── Focus gauge (best DO data availability) ───────────────────────────────
FOCUS_GAUGE = "2473"

# ── Random seed ───────────────────────────────────────────────────────────
SEED = 42

# ── Temperature-only configuration (notebook 04b) ─────────────────────────
# All 86 gauges have temperature data — enables true multi-site analysis
# across the full CAMELS-CH-Chem network.
FEATURES_TEMP = ["temp_sensor", "pH_sensor", "ec_sensor"]   # inputs (no DO as feature)
TARGETS_TEMP  = ["temp_sensor"]                               # predict temperature only
N_FEAT_TEMP   = len(FEATURES_TEMP)
N_TGT_TEMP    = len(TARGETS_TEMP)

TARGET_LABELS_TEMP = {"temp_sensor": "Temp (°C)"}

# Minimum coverage threshold for temperature gauges (virtually all pass)
TEMP_MIN_COVERAGE = 0.50   # 50% — generous since temp is well-observed
