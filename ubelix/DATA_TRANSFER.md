# AareML — Getting Your Data onto UBELIX

## What needs to go there

| Item | Size | Method |
|------|------|--------|
| `AareML/` project folder | ~7 MB (code + notebooks) | `scp` or `rsync` |
| CAMELS-CH-Chem dataset | ~165 MB unzipped | Download directly on UBELIX from Zenodo |
| LakeBeD-US Lake Mendota | ~194 MB | Download directly on UBELIX from Hugging Face |

The datasets are large — it's faster and cleaner to download them directly on UBELIX
rather than uploading from your Mac.

---

## Step 1 — Upload the project code

On your Mac terminal (replace `your-campus-id` with your UniBE campus account):

```bash
# Navigate to the parent folder of AareML
cd ~/Documents   # or wherever your AareML folder lives

# Upload the whole project (excludes data/ automatically via rsync)
rsync -avz --exclude='data/' --exclude='.git/' --exclude='__pycache__/' \
    AareML/ your-campus-id@submit01.unibe.ch:~/AareML/
```

This takes about 10–30 seconds on a normal connection.

---

## Step 2 — Connect to UBELIX

```bash
ssh your-campus-id@submit01.unibe.ch
```

Note: you need to be on the UniBE VPN or campus network.
VPN client: [vpn.unibe.ch](https://vpn.unibe.ch)

---

## Step 3 — Download CAMELS-CH-Chem directly on UBELIX

Once logged in to UBELIX, run these commands:

```bash
cd ~/AareML
mkdir -p data/camels-ch-chem

# Download from Zenodo (~165 MB, takes 1-2 min on UBELIX's fast connection)
wget -O data/camels-ch-chem.zip \
    "https://zenodo.org/api/records/14980027/files/camels-ch-chem.zip/content"

# Unzip
unzip data/camels-ch-chem.zip -d data/camels-ch-chem/

# Remove zip to save space
rm data/camels-ch-chem.zip

echo "CAMELS-CH-Chem ready."
ls data/camels-ch-chem/
```

---

## Step 4 — Download LakeBeD-US Lake Mendota directly on UBELIX

```bash
mkdir -p data/lakebed-us

# Lake Info
wget -q -O data/lakebed-us/Lake_Info.csv \
    "https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE/resolve/main/Data/Lake_Info.csv"

# NTL high-frequency file (~870 KB)
wget -q -O data/lakebed-us/ME_NTL_HF_2D.parquet \
    "https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE/resolve/main/Data/HighFrequency/ME/ME_NTL_HF_2D.parquet"

# Main Mendota file (~194 MB) — this takes ~1 min
wget -O data/lakebed-us/ME_Mendota_2D.parquet \
    "https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE/resolve/main/Data/HighFrequency/ME/ME_Mendota_2D.parquet"

echo "LakeBeD-US data ready."
ls -lh data/lakebed-us/
```

---

## Step 5 — Pre-process Lake Mendota (if not already done)

The daily surface CSV is already in `data/lakebed-us/ME_daily_surface.csv` inside
the ZIP. If it is missing, re-run the preprocessing:

```bash
cd ~/AareML
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate aareml
python -c "
import pyarrow.parquet as pq, pandas as pd, numpy as np
pf = pq.ParquetFile('data/lakebed-us/ME_Mendota_2D.parquet')
chunks = []
for i in range(pf.metadata.num_row_groups):
    rg = pf.read_row_group(i).to_pandas()
    surface = rg[(rg['depth'] >= 0.5) & (rg['depth'] <= 1.5)]
    if len(surface) == 0: continue
    surface['date'] = pd.to_datetime(surface['datetime']).dt.date
    daily = surface.groupby('date')[['do','temp','chla_rfu','phyco','par']].median()
    chunks.append(daily)
daily_all = pd.concat(chunks).groupby(level=0).median()
daily_all.index = pd.to_datetime(daily_all.index)
daily_all = daily_all[daily_all.index >= '2006-01-01'].sort_index()
daily_all.to_csv('data/lakebed-us/ME_daily_surface.csv')
print(f'Done: {len(daily_all)} days saved.')
"
```

---

## Step 6 — Set up environment and run jobs

```bash
# One-time setup
bash ubelix/setup_env.sh

# Submit all jobs (takes ~1 hour total, you can log out)
bash ubelix/run_all.sh
```

---

## Step 7 — Download results back to your Mac

After jobs finish, from your Mac:

```bash
# Download executed notebooks and results
rsync -avz \
    your-campus-id@submit01.unibe.ch:~/AareML/notebooks/ \
    ~/Documents/AareML/notebooks/

rsync -avz \
    your-campus-id@submit01.unibe.ch:~/AareML/results/ \
    ~/Documents/AareML/results/

rsync -avz \
    your-campus-id@submit01.unibe.ch:~/AareML/figures/ \
    ~/Documents/AareML/figures/
```

Then share the results with me and I'll update the report.

---

## Useful UBELIX commands

```bash
squeue --me              # see your running/pending jobs
scancel <job_id>         # cancel a job
sinfo -p gpu             # check GPU partition availability
tail -f logs/job_03_lstm_<id>.out   # watch live job output
sacct -j <job_id>        # see completed job stats (runtime, memory)
```
