#!/bin/bash
# One GPU per rank, round-robin over what the node has.
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
export CUDA_VISIBLE_DEVICES=$(( local_rank % $(nvidia-smi -L | wc -l) ))
export OMP_NUM_THREADS=4
exec "$@"
