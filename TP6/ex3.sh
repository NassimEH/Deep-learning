#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --account=csc_8607
#SBATCH -J tp6-ex3
#SBATCH -o logs/%x-%j.out
#SBATCH --chdir=/mnt/hdd/homes/nelhaddad/TP6

set -euo pipefail
mkdir -p logs
source tp6_env.sh
python -u bench.py --tag fp32-synthetic
python -u bench.py --tag tf32-synthetic --precision tf32
python -u bench.py --tag bf16-synthetic --precision bf16
python -u bench.py --tag bf16-cl-synthetic --precision bf16 --channels-last
python -u bench.py --tag bf16-cl-compile-synthetic --precision bf16 --channels-last --compile
for D in raw small; do
    for W in 1 3; do
        R=$([ "$W" = 1 ] && echo 5 || echo 3)
        python -u bench.py --tag bf16-cl-$D-w$W --precision bf16 --channels-last --data $D \
               --workers $W --repeat $R
        python -u bench.py --tag bf16-cl-compile-$D-w$W --precision bf16 --channels-last --compile \
               --data $D --workers $W --repeat $R
    done
done
python -u bench.py --tag maxbatch-bf16 --precision bf16 --channels-last --find-max-batch
