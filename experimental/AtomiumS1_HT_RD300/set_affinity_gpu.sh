#!/bin/bash
rank="${OMPI_COMM_WORLD_RANK:-$SLURM_PROCID}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-$SLURM_LOCALID}"
export OMP_NUM_THREADS=4
ngpus=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=$(( $local_rank % $ngpus ))
echo $rank" uses "${CUDA_VISIBLE_DEVICES}" of "$ngpus "  " `hostname`
$*
