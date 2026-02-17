#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=job_output_%x_%j.txt
#SBATCH --error=job_error_%x_%j.txt
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=24Gb
#SBATCH --time=60:00:00

# ---------- Metadata ----------
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# ---------- Env / Modules ----------
module load anaconda/3
conda activate py3.10.4
module load python/3.10

# ---------- Data staging ----------
# Your train_tlens.py expects $SLURM_TMPDIR/data
mkdir -p "$SLURM_TMPDIR/data"
cp -r [path_to_your_data] "$SLURM_TMPDIR/data/"

# ---------- Run training ----------
python train_model.py "$@"