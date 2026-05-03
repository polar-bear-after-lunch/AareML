#!/usr/bin/env python3
"""
AareML — Data Download & Preparation Script
============================================
Downloads and prepares all datasets required to run AareML notebooks.

Usage:
    python download_data.py           # download everything
    python download_data.py --camels  # CAMELS-CH-Chem only
    python download_data.py --lake    # LakeBeD-US Lake Mendota only

Datasets:
    1. CAMELS-CH-Chem  (~165 MB) — Swiss river sensor data (Zenodo)
    2. LakeBeD-US ME   (~194 MB) — Lake Mendota buoy data (Hugging Face)

Total download: ~360 MB
Estimated time: 5–10 min depending on connection speed.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
CAMELS_DIR  = DATA_DIR / "camels-ch-chem"
LAKE_DIR    = DATA_DIR / "lakebed-us"

# ── URLs ───────────────────────────────────────────────────────────────────
CAMELS_URL  = "https://zenodo.org/api/records/14980027/files/camels-ch-chem.zip/content"
LAKE_HF_BASE = "https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE/resolve/main/Data"
MENDOTA_URL  = f"{LAKE_HF_BASE}/HighFrequency/ME/ME_Mendota_2D.parquet"
NTL_URL      = f"{LAKE_HF_BASE}/HighFrequency/ME/ME_NTL_HF_2D.parquet"
LAKEINFO_URL = f"{LAKE_HF_BASE}/Lake_Info.csv"


# ── Helpers ────────────────────────────────────────────────────────────────

def run(cmd, desc):
    print(f"\n  {desc}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ERROR: command failed — {cmd}")
        sys.exit(1)


def check_tool(name):
    result = subprocess.run(f"which {name}", shell=True,
                            capture_output=True, text=True)
    return result.returncode == 0


def download(url, dest, desc):
    dest = Path(dest)
    if dest.exists():
        print(f"  Already exists: {dest.name} — skipping download")
        return
    print(f"\n  Downloading {desc}...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")
    t0 = time.time()
    if check_tool("wget"):
        run(f'wget -q --show-progress -O "{dest}" "{url}"', f"wget → {dest.name}")
    elif check_tool("curl"):
        run(f'curl -L --progress-bar -o "{dest}" "{url}"', f"curl → {dest.name}")
    else:
        # Pure Python fallback
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 1e6
    print(f"  Done: {size_mb:.1f} MB in {elapsed:.0f}s")


# ── CAMELS-CH-Chem ─────────────────────────────────────────────────────────

def download_camels():
    print("\n" + "="*60)
    print("  1/2  CAMELS-CH-Chem (Swiss river sensor data)")
    print("       Source: Zenodo record 14980027")
    print("       Reference: Nascimento et al. (2025)")
    print("="*60)

    CAMELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "camels-ch-chem.zip"

    # Check if already extracted
    sensor_dir = CAMELS_DIR / "stream_water_chemistry" / "timeseries" / "daily"
    if sensor_dir.exists() and len(list(sensor_dir.glob("*.csv"))) > 80:
        print(f"  Already extracted: {len(list(sensor_dir.glob('*.csv')))} CSV files found")
        print("  Skipping download and extraction.")
        return

    # Download
    download(CAMELS_URL, zip_path, "CAMELS-CH-Chem (~165 MB)")

    # Unzip
    print("\n  Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  {len(members)} files in archive")
        for i, member in enumerate(members):
            zf.extract(member, CAMELS_DIR)
            if i % 100 == 0:
                print(f"  Extracted {i+1}/{len(members)}...", end="\r")
    print(f"\n  Extracted {len(members)} files to {CAMELS_DIR}")

    # Remove zip to save space
    zip_path.unlink()
    print("  Removed zip file to save disk space")

    # Verify
    n_csv = len(list(sensor_dir.glob("*.csv")))
    if n_csv >= 80:
        print(f"\n  CAMELS-CH-Chem ready: {n_csv} daily sensor CSV files")
    else:
        print(f"\n  WARNING: expected 86 CSV files, found {n_csv}")


# ── LakeBeD-US ─────────────────────────────────────────────────────────────

def download_lake():
    print("\n" + "="*60)
    print("  2/2  LakeBeD-US — Lake Mendota surface data")
    print("       Source: Hugging Face (eco-kgml/LakeBeD-US-CSE)")
    print("       Reference: McAfee et al. (2025)")
    print("="*60)

    LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Lake info + NTL file (small)
    download(LAKEINFO_URL,
             LAKE_DIR / "Lake_Info.csv",
             "Lake_Info.csv (~5 KB)")
    download(NTL_URL,
             LAKE_DIR / "ME_NTL_HF_2D.parquet",
             "ME_NTL_HF_2D.parquet (~870 KB)")

    # Main Mendota file (large)
    mendota_path = LAKE_DIR / "ME_Mendota_2D.parquet"
    download(MENDOTA_URL, mendota_path, "ME_Mendota_2D.parquet (~194 MB)")

    # Pre-process to daily surface CSV
    surface_csv = LAKE_DIR / "ME_daily_surface.csv"
    if surface_csv.exists():
        import pandas as pd
        n = len(pd.read_csv(surface_csv))
        print(f"\n  Daily surface CSV already exists ({n} rows) — skipping pre-processing")
        return

    print("\n  Pre-processing: aggregating to daily surface medians...")
    print("  (reading 101M rows in chunks — takes 2–3 min)")
    t0 = time.time()

    try:
        import pyarrow.parquet as pq
        import pandas as pd
        import numpy as np

        pf = pq.ParquetFile(str(mendota_path))
        print(f"  Row groups: {pf.metadata.num_row_groups} | "
              f"Total rows: {pf.metadata.num_rows:,}")

        chunks = []
        for i in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(i).to_pandas()
            surface = rg[(rg["depth"] >= 0.5) & (rg["depth"] <= 1.5)]
            if len(surface) == 0:
                continue
            surface["date"] = pd.to_datetime(surface["datetime"]).dt.date
            daily = surface.groupby("date")[
                ["do", "temp", "chla_rfu", "phyco", "par"]
            ].median()
            chunks.append(daily)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{pf.metadata.num_row_groups} row groups...",
                      end="\r")

        daily_all = pd.concat(chunks).groupby(level=0).median()
        daily_all.index = pd.to_datetime(daily_all.index)
        daily_all = daily_all[daily_all.index >= "2006-01-01"].sort_index()
        daily_all.to_csv(str(surface_csv))

        elapsed = time.time() - t0
        print(f"\n  Pre-processing complete: {len(daily_all)} daily rows "
              f"({elapsed:.0f}s)")
        print(f"  Saved: {surface_csv}")

    except ImportError:
        print("  ERROR: pyarrow not installed. Run: pip install pyarrow")
        print("  After installing, re-run: python download_data.py --lake")
        sys.exit(1)


# ── Swiss Lakes (Bärenbold et al. 2026) ──────────────────────────────────────────────

def download_swiss_lakes():
    """
    Download the Bärenbold et al. (2026) harmonised Swiss lakes dataset.
    21 Swiss lakes, temperature + DO + conductivity + Secchi depth,
    1938/1980s–2023, at bi-weekly to monthly resolution.
    Source: Eawag Open Data Portal
    DOI: https://doi.org/10.25678/0009KJ
    """
    print("\n" + "="*60)
    print("  3/3  Swiss Lakes (Bärenbold et al. 2026)")
    print("       Source: Eawag Open Data Portal")
    print("       Reference: Bärenbold et al. (2026), ESSD")
    print("="*60)

    SWISS_LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Try the Eawag open data portal
    url = "https://opendata.eawag.ch/dataset/3b9b8bc3-f7bb-4e59-8b08-079b073d7dc9/resource/a1f7bb98-5af2-4947-b77e-c9d8b1bb3b4f/download/swiss_lakes_long_term.zip"
    dest = SWISS_LAKE_DIR / "swiss_lakes_long_term.zip"

    if any(SWISS_LAKE_DIR.glob("*.csv")) or any(SWISS_LAKE_DIR.glob("*.parquet")):
        print(f"  Swiss lakes data already present in {SWISS_LAKE_DIR} — skipping download.")
        return

    print(f"  Downloading Swiss Lakes dataset...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    try:
        _wget(url, dest)
        print("  Extracting...")
        import zipfile
        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall(SWISS_LAKE_DIR)
        dest.unlink()
        n_files = len(list(SWISS_LAKE_DIR.rglob("*.csv")))
        print(f"  Swiss Lakes ready: {n_files} CSV files in {SWISS_LAKE_DIR}")
    except Exception as e:
        print(f"  WARNING: Could not download Swiss Lakes dataset: {e}")
        print(f"  You can download it manually from:")
        print(f"  https://opendata.eawag.ch/dataset/long-term-temperature-oxygen-and-water-clarity-trends-in-swiss-lakes")
        print(f"  Place the extracted files in: {SWISS_LAKE_DIR}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download AareML datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--camels",       action="store_true",
                        help="Download CAMELS-CH-Chem only")
    parser.add_argument("--lake",          action="store_true",
                        help="Download LakeBeD-US Lake Mendota only")
    parser.add_argument("--swiss-lakes",   action="store_true",
                        help="Download Bärenbold et al. 2026 Swiss Lakes dataset only")
    args = parser.parse_args()

    do_camels      = args.camels or      (not args.camels and not args.lake and not args.swiss_lakes)
    do_lake        = args.lake   or      (not args.camels and not args.lake and not args.swiss_lakes)
    do_swiss_lakes = args.swiss_lakes or (not args.camels and not args.lake and not args.swiss_lakes)

    print("\nAareML — Data Download & Preparation")
    print(f"Data directory: {DATA_DIR.resolve()}")
    print(f"Datasets: {'CAMELS-CH-Chem ' if do_camels else ''}"
          f"{'LakeBeD-US' if do_lake else ''}")

    t_start = time.time()

    if do_camels:
        download_camels()
    if do_lake:
        download_lake()
    if do_swiss_lakes:
        download_swiss_lakes()

    total = time.time() - t_start
    print("\n" + "="*60)
    print(f"  All done in {total/60:.1f} min")
    print("  You can now run the notebooks in order: 01 → 02 → 03 → 04 → 04b → 05 → 06 → 07 → 08 → 09 → 10")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
