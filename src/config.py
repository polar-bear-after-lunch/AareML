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
# Most commonly available monthly chemistry variables across CAMELS-CH-Chem gauges
NAWAF_FEATURES = [
    "NO3_N",    # nitrate nitrogen (mg/L)
    "NH4_N",    # ammonium nitrogen (mg/L)
    "TP",       # total phosphorus (mg/L)
    "TN",       # total nitrogen (mg/L)
    "DOC",      # dissolved organic carbon (mg/L)
    "Q_m3s",    # discharge at sample time (m³/s)
]

# Features with NAWA FRACHT appended (used when NAWA data is available)
FEATURES_WITH_NAWAF = FEATURES + NAWAF_FEATURES

# ── Train / val / test split boundaries ───────────────────────────────────
TRAIN_END  = "2014-12-31"
VAL_END    = "2016-12-31"
# Test: 2017-01-01 onward

# ── Focus gauge (best DO data availability) ───────────────────────────────
FOCUS_GAUGE = "2473"

# ── Random seed ───────────────────────────────────────────────────────────
SEED = 42
