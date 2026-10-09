#!/bin/bash
# Pin one GPU per rank.  Reads whichever rank variables the launcher exports:
# Open MPI (OMPI_*), SLURM (SLURM_*), or PMI/PALS as used by Cray MPICH on
# Polaris (PMI_*).
rank="${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-${PMI_RANK:-?}}}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_LOCAL_RANK:-}}}"
if [ -z "$local_rank" ]; then
    echo "set_affinity_gpu.sh: no local-rank variable in the environment (looked for" \
         "OMPI_COMM_WORLD_LOCAL_RANK, SLURM_LOCALID, PMI_LOCAL_RANK) -- every rank" \
         "would land on GPU 0, refusing to start" >&2
    exit 1
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
ngpus=$(nvidia-smi -L | wc -l)
dev=$(( local_rank % ngpus ))
# Polaris numbers its GPUs opposite to the CPU NUMA nodes, so local rank 0
# belongs on GPU 3.  Set GPU_REVERSE=1 there (see Readme.md, Polaris section).
[ "${GPU_REVERSE:-0}" = "1" ] && dev=$(( ngpus - 1 - dev ))
export CUDA_VISIBLE_DEVICES=$dev
echo "$rank uses $CUDA_VISIBLE_DEVICES of $ngpus   $(hostname)"
exec "$@"
