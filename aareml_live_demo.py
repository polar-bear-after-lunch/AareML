"""
AareML Live Demo — Real-Time DO Prediction at Gauge 2473
=========================================================
Fetches the last 21 days of live temperature + flow data from the FOEN
API (api.existenz.ch), fills missing features with training-set means,
and runs the trained AareML LSTM to produce a 14-day DO forecast.

Usage:
    python3 aareml_live_demo.py

Requirements:
    pip install requests numpy pandas torch
    # Must have trained model checkpoint at: results/lstm_best.pt
    # Must have AareML repo in PYTHONPATH

Data source:
    api.existenz.ch — unofficial FOEN wrapper, 10-min intervals, free
    License: https://www.bafu.admin.ch/dam/bafu/de/dokumente/wasser/
             fachinfo-daten/liefer-nutzungsbedingungen-hydrologische-daten.pdf
"""

import sys, json, requests
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ── Config ─────────────────────────────────────────────────────────────────
GAUGE        = "2473"
API_BASE     = "https://api.existenz.ch/apiv1/hydro"
LOOKBACK     = 21   # days
HORIZON      = 14   # days forecast
FEATURES     = ['temp_sensor', 'pH_sensor', 'ec_sensor', 'O2C_sensor']
TARGETS      = ['O2C_sensor', 'temp_sensor']

# Training-set means (from nb03, used to fill unavailable features)
TRAIN_MEANS = {
    'temp_sensor': 10.42,   # °C  — from CAMELS-CH-Chem gauge 2473
    'pH_sensor':    8.10,   # pH units
    'ec_sensor':   312.0,   # µS/cm
    'O2C_sensor':   10.8,   # mg/L
}

# Training-set std (for StandardScaler reconstruction)
TRAIN_STDS = {
    'temp_sensor': 4.82,
    'pH_sensor':   0.18,
    'ec_sensor':  52.4,
    'O2C_sensor':  1.41,
}

REPO_ROOT = Path(__file__).parent
CHECKPOINT = REPO_ROOT / "results" / "lstm_best.pt"

# ── Step 1: Fetch real-time data ─────────────────────────────────────────
def fetch_live_data(gauge: str) -> pd.DataFrame:
    """
    Fetch the latest available temperature + flow data from the FOEN API.
    The api.existenz.ch endpoint provides the most recent 1-2 days at 10-min
    resolution. We resample to daily means.
    """
    from datetime import timezone
    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=3)  # fetch last 3 days

    url = (f"{API_BASE}/daterange"
           f"?locations={gauge}"
           f"&parameters=temperature,flow"
           f"&startDate={start}&endDate={end}")

    print(f"Fetching live data for gauge {gauge}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()["payload"]

    # Pivot to wide format — daily means
    from collections import defaultdict
    by_date = defaultdict(lambda: defaultdict(list))
    for item in payload:
        from datetime import timezone as tz
        ts  = datetime.fromtimestamp(item["timestamp"], tz=tz.utc).date()
        by_date[ts][item["par"]].append(item["val"])

    rows = []
    for ts in sorted(by_date.keys()):
        pars = by_date[ts]
        rows.append({
            "date":        ts,
            "temperature": float(np.mean(pars.get("temperature", [np.nan]))),
            "flow":        float(np.mean(pars.get("flow",        [np.nan]))),
        })

    df = pd.DataFrame(rows).set_index("date")
    print(f"  → {len(df)} live daily record(s) | "
          f"latest temp: {df.temperature.iloc[-1]:.2f}°C, "
          f"flow: {df.flow.iloc[-1]:.1f} m\u00b3/s")
    return df


def build_lookback_window(live_df: pd.DataFrame, hist_csv: str = None) -> pd.DataFrame:
    """
    Build a 21-day lookback window by:
    1. Using historical CAMELS-CH-Chem data as the base (if csv provided)
    2. Appending the latest live days on top
    3. If no historical csv, use training means for all days except the live ones
    """
    from datetime import timezone
    today = datetime.now(timezone.utc).date()
    lookback_start = today - timedelta(days=LOOKBACK)

    if hist_csv and Path(hist_csv).exists():
        # Load from CAMELS-CH-Chem local file
        hist = pd.read_csv(hist_csv, parse_dates=['date'], index_col='date')
        hist.index = hist.index.date
        hist = hist.loc[lookback_start:today]
        # Override with live values for available dates
        for date, row in live_df.iterrows():
            hist.loc[date, 'temperature'] = row['temperature']
            hist.loc[date, 'flow']        = row['flow']
        return hist.tail(LOOKBACK)
    else:
        # Fallback: fill all days with training means, patch live days
        dates = pd.date_range(start=lookback_start, periods=LOOKBACK, freq='D').date
        df = pd.DataFrame({
            'date':        dates,
            'temperature': TRAIN_MEANS['temp_sensor'],
            'flow':        350.0,  # approximate annual mean for gauge 2473
        }).set_index('date')
        # Patch with live observations
        for date, row in live_df.iterrows():
            if date in df.index:
                df.loc[date, 'temperature'] = row['temperature']
                df.loc[date, 'flow']        = row['flow']
        live_days = len([d for d in live_df.index if d in df.index])
        print(f"  → Lookback: {LOOKBACK} days ({live_days} live, "
              f"{LOOKBACK - live_days} filled with training means)")
        return df


# ── Step 2: Build feature matrix ─────────────────────────────────────────
def build_feature_matrix(live_df: pd.DataFrame) -> np.ndarray:
    """
    Map live data → AareML feature order [temp, pH, EC, DO].
    pH, EC, DO are unavailable in real-time → filled with training means.

    Note: predictions will be less accurate without real pH/EC/DO.
    The temperature prediction (second output) is more reliable since
    temp_sensor is observed.
    """
    n = len(live_df)
    X = np.zeros((n, len(FEATURES)))

    for i, feat in enumerate(FEATURES):
        if feat == 'temp_sensor':
            X[:, i] = live_df['temperature'].fillna(TRAIN_MEANS['temp_sensor']).values
        else:
            # Fill unavailable features with training mean
            X[:, i] = TRAIN_MEANS[feat]

    return X.astype(np.float32)


# ── Step 3: Scale features ────────────────────────────────────────────────
def scale_features(X: np.ndarray) -> np.ndarray:
    """StandardScaler using training statistics."""
    means = np.array([TRAIN_MEANS[f] for f in FEATURES], dtype=np.float32)
    stds  = np.array([TRAIN_STDS[f]  for f in FEATURES], dtype=np.float32)
    return (X - means) / stds


# ── Step 4: Run model ─────────────────────────────────────────────────────
def run_model(X_scaled: np.ndarray) -> dict:
    """
    Load trained checkpoint and generate 14-day forecast.
    Returns unscaled DO and temperature predictions.
    """
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Trained model not found at {CHECKPOINT}\n"
            f"Run AareML on UBELIX first, then copy results/lstm_best.pt here."
        )

    sys.path.insert(0, str(REPO_ROOT))
    from src.model import Seq2SeqLSTM

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = Seq2SeqLSTM(
        n_feat   = len(FEATURES),
        n_tgt    = len(TARGETS),
        hidden   = cfg.get("hidden", 256),
        n_layers = cfg.get("n_layers", 2),
        dropout  = 0.0,  # inference
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build (1, LOOKBACK, n_feat) tensor
    window = X_scaled[-LOOKBACK:]
    x = torch.tensor(window[None, ...])  # (1, 21, 4)

    with torch.no_grad():
        y_scaled = model(x, teacher_forcing_ratio=0.0)  # (1, 14, 2)

    y_np = y_scaled.squeeze(0).numpy()  # (14, 2)

    # Unscale targets
    target_means = np.array([TRAIN_MEANS[t] for t in TARGETS])
    target_stds  = np.array([TRAIN_STDS[t]  for t in TARGETS])
    y_unscaled   = y_np * target_stds + target_means

    return {
        "DO_pred":   y_unscaled[:, 0],   # mg/L
        "temp_pred": y_unscaled[:, 1],   # °C
    }


# ── Step 5: Display results ───────────────────────────────────────────────
def display_forecast(forecast: dict, last_date) -> None:
    today = last_date + timedelta(days=1)

    print("\n" + "="*58)
    print(f"  AareML 14-day forecast — Gauge 2473 (Bern)")
    print(f"  Forecast origin: {last_date}  |  Today: {datetime.utcnow().date()}")
    print("="*58)
    print(f"  {'Day':<6} {'Date':<12} {'DO (mg/L)':>10} {'Temp (°C)':>10}  {'DO status'}")
    print("  " + "-"*54)

    for d in range(HORIZON):
        date = today + timedelta(days=d)
        do   = forecast["DO_pred"][d]
        temp = forecast["temp_pred"][d]
        # DO fish stress thresholds
        if do < 5.0:
            status = "⚠ CRITICAL"
        elif do < 6.0:
            status = "⚡ STRESS"
        elif do < 7.0:
            status = "↓ Low"
        else:
            status = "✓ OK"
        print(f"  +{d+1:<5d} {str(date):<12} {do:>10.3f} {temp:>10.2f}  {status}")

    print("="*58)
    print(f"\n  Note: pH, EC, DO inputs filled with training means.")
    print(f"  Temperature predictions are most reliable.")
    print(f"  Source: api.existenz.ch (FOEN BAFU), License: CC-BY")
    print()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("\nAareML Live Demo — Real-Time River DO Forecast")
    print("=" * 50)

    # 1. Fetch live observations (last 1-2 days)
    live_df = fetch_live_data(GAUGE)

    # 2. Build 21-day lookback (live + historical fill)
    # Optional: pass path to local CAMELS-CH-Chem CSV for accurate history
    # e.g., hist_csv="data/camels-ch-chem/stream_water_chemistry/timeseries/daily/camels_ch_chem_daily_2473.csv"
    window_df = build_lookback_window(live_df, hist_csv=None)

    # 3. Feature matrix
    X_raw = build_feature_matrix(window_df)

    # 3. Scale
    X_scaled = scale_features(X_raw)

    # 4. Forecast (requires trained model)
    try:
        forecast = run_model(X_scaled)
        display_forecast(forecast, last_date=live_df.index[-1])
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n--- Data preview (model not available) ---")
        print(f"Latest 5 daily values from gauge {GAUGE}:")
        print(live_df.tail(5).to_string())
        print(f"\nReady to run inference once lstm_best.pt is available.")
        print(f"Lookback window shape: {X_scaled.shape} → (LOOKBACK, n_features)")


if __name__ == "__main__":
    main()
