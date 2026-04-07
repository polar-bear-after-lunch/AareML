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

# ── Train / val / test split boundaries ───────────────────────────────────
TRAIN_END  = "2014-12-31"
VAL_END    = "2016-12-31"
# Test: 2017-01-01 onward

# ── Focus gauge (best DO data availability) ───────────────────────────────
FOCUS_GAUGE = "2473"

# ── Random seed ───────────────────────────────────────────────────────────
SEED = 42
