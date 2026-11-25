#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --account=csc_8607
#SBATCH -J tp6-ddp
#SBATCH -o logs/%x-%j.out
#SBATCH --chdir=/mnt/hdd/homes/nelhaddad/TP6

set -euo pipefail
mkdir -p logs
export TP6_NO_COPY=1
source tp6_env.sh

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export MASTER_ADDR=${NODES[0]}
PORT=$((20000 + SLURM_JOB_ID % 10000))
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
echo "Nœuds : ${NODES[*]} ; rendez-vous sur $MASTER_ADDR"

run2() {
    local port=$1; shift
    srun torchrun --nnodes=2 --nproc_per_node=1 --rdzv_backend=c10d \
         --rdzv_id="$SLURM_JOB_ID-$port" --rdzv_endpoint="$MASTER_ADDR:$port" "$@"
}

export NCCL_DEBUG=INFO
if ! run2 "$PORT" check_dist.py > "logs/nccl-$SLURM_JOB_ID.log" 2>&1; then
    echo "ÉCHEC du test de communication, fin du log :"; tail -n 30 "logs/nccl-$SLURM_JOB_ID.log"; exit 1
fi
grep -E "rang|all-reduce|Using network|NET/" "logs/nccl-$SLURM_JOB_ID.log" | sort -u | head -n 20 || true
export NCCL_DEBUG=WARN

srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
     torchrun --standalone --nproc_per_node=1 ddp_bench.py --tag l4-x1 --channels-last

run2 $((PORT + 1)) ddp_bench.py --tag l4-x2 --channels-last

run2 $((PORT + 2)) ddp_bench.py --tag l4-x2-bf16hook --channels-last --bf16-hook

srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
     torchrun --standalone --nproc_per_node=1 ddp_bench.py --tag l4-x1-accum8 --channels-last \
     --accum 8 --warmup 3 --steps 10

run2 $((PORT + 3)) ddp_bench.py --tag l4-x2-accum8-bf16hook --channels-last --bf16-hook \
     --accum 8 --warmup 3 --steps 10
