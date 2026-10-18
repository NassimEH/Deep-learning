#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -J alexnet-full
#SBATCH --chdir=/mnt/hdd/homes/nelhaddad/TP3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eo pipefail
cd /mnt/hdd/homes/nelhaddad/TP3
mkdir -p logs checkpoints runs figures

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning
set -u
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

echo "Début du job $SLURM_JOB_ID sur le nœud $SLURM_NODELIST"
nvidia-smi

python -u train_imagenet.py --model alexnet --num_epochs 50 --batch_size 256
