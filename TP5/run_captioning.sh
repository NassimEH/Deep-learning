#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH -J captioning
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --chdir=/mnt/hdd/homes/nelhaddad/TP5

set -euo pipefail
mkdir -p logs runs

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning
set -u

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.__version__)"

python -u train.py --num_epochs 15 --lr 0.001 --hidden_dim 512
