#!/bin/bash
# One GPU per MPI rank, selected by UUID rather than by ordinal.
#
# An ordinal only means something inside one enumeration, and there are two:
# nvidia-smi lists by PCI bus, while CUDA lists by CUDA_DEVICE_ORDER, whose
# default is FASTEST_FIRST -- so on a node with unlike cards "rank 2 -> device 2"
# can land on a different card than the one nvidia-smi calls 2.  Handing CUDA the
# UUID nvidia-smi reported removes the question entirely: whatever the ordering,
# the rank runs on the card printed below.  PCI_BUS_ID is pinned as well, so that
# anything downstream that still enumerates by number agrees with nvidia-smi.
#
# To check where the ranks actually landed, while the job runs:
#     nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
rank="${OMPI_COMM_WORLD_RANK:-$SLURM_PROCID}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-$SLURM_LOCALID}"
export OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # an outer restriction (a scheduler, or the caller picking cards by hand)
    IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
fi
ngpus=${#gpus[@]}
if [ "$ngpus" -eq 0 ]; then
    echo "set_affinity_gpu.sh: no GPU found (nvidia-smi returned nothing)" >&2
    exit 1
fi
idx=$(( local_rank % ngpus ))
if [ "$local_rank" -ge "$ngpus" ]; then
    # more ranks than cards: two ranks then share one GPU and each sees half the
    # memory, which looks from nvidia-smi like the ranks landed on the wrong cards
    echo "set_affinity_gpu.sh: local rank $local_rank on a node with $ngpus GPU(s)" \
         "-- sharing gpu $idx with rank $(( local_rank % ngpus ))" >&2
fi
export CUDA_VISIBLE_DEVICES=${gpus[$idx]}
# index is the position in the list above -- which is nvidia-smi's index unless
# the caller restricted the set -- and the UUID identifies the card outright
echo "rank $rank -> gpu $idx of $ngpus  ${CUDA_VISIBLE_DEVICES}  $(hostname -s)"
"$@"
