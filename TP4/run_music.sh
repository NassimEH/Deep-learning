#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH -J music-lstm
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
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.__version__)"

python -u train_music.py \
  --mode full \
  --rnn_type lstm \
  --subset_size 50000 \
  --num_epochs 30 \
  --hidden_size 1024 \
  --embedding_dim 256 \
  --max_grad_norm 1.0 \
  --patience 5 \
  --checkpoint best_lstm.pth
