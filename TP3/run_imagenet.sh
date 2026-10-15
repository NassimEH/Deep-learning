#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -J alexnet-full
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs checkpoints

# Activation de l'environnement virtuel Mamba/Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Début du job $SLURM_JOB_ID sur le nœud $SLURM_NODELIST"
nvidia-smi

# Entraînement long : AlexNet sur ImageNet complet (subset_size=None par défaut), 50 époques
python -u train_imagenet.py --model alexnet --num_epochs 50
