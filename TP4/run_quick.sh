#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH -J music-quick
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --chdir=/mnt/hdd/homes/nelhaddad/TP4

set -euo pipefail
mkdir -p logs generated runs_rnn figures

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning
set -u

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi

python -u train_music.py --mode quick
