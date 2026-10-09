#!/usr/bin/env bash
# Multi-node analog of ./run.sh -- geometry + synthetic mosaic data.
#
#   ./run_mn.sh                              # all GPUs on tomo2..tomo5
#   NODES="tomo2 tomo4" ./run_mn.sh          # subset of nodes
#   NP=6 ./run_mn.sh                         # fewer ranks than GPUs
#   CONF=config_gen.conf ./run_mn.sh
#   ./run_mn.sh --plan                       # sizes only, local, no GPU
#
# Launch from any of tomo2..tomo5 (handyn has no mpirun and no /data3).
# path_out must be on /data3 -- it is NFS-shared across all four nodes.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CONF=${CONF:-config_gen.conf}

if [ "${1:-}" = "--plan" ]; then
    PY=${PY:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}
    "$PY" gen_data.py "$CONF" --plan
    exit 0
fi

source ./mpi_multinode.sh
mn_banner

# Serial, on this node only: writes shift_dir/, read by every rank afterwards.
"$PY" make_geometry.py "$CONF" ${FROM_H5:+--from-h5 "$FROM_H5"}

"$MPIRUN" -n "$NP" $MPI_ARGS ./set_affinity_gpu.sh "$PY" gen_data.py "$CONF"
