#!/bin/bash
# Rerun nb06, nb07, nb14, nb17 to restore executed notebook outputs
cd /storage/homefs/tn20y076/AareML

if squeue --me --noheader 2>/dev/null | grep -q "aareml"; then
    echo "ERROR: AareML jobs already in queue. Run: scancel --me first."
    exit 1
fi

echo "Submitting nb06, nb07, nb14, nb17..."
JOB1=$(sbatch --parsable ubelix/job_06_lake_mendota.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 ubelix/job_07_lake_eda.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 ubelix/job_14_ar_baseline.sh)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 ubelix/job_17_neuralhydrology.sh)
echo "Submitted: $JOB1, $JOB2, $JOB3, $JOB4"
echo "Expected runtime: ~30 min total"
