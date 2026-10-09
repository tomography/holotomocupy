#!/usr/bin/env bash
# Smoke-test the multi-node setup without running a real job: one rank per GPU,
# each reports its host, pinned device, cupy visibility, and /data3 access,
# then the ranks do a collective so the interconnect is exercised too.
#
#   ./check_nodes_mn.sh
#   NODES="tomo2 tomo4" ./check_nodes_mn.sh
set -eu
cd "$(dirname "$(readlink -f "$0")")"

source ./mpi_multinode.sh
mn_banner

"$MPIRUN" -n "$NP" $MPI_ARGS ./set_affinity_gpu.sh "$PY" _check_rank.py
