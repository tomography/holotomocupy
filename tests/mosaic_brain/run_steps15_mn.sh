#!/usr/bin/env bash
# Multi-node analog of ./run_steps15.sh -- stitch -> Paganin -> FBP -> {pfile}_obj.h5.
#
#   ./run_steps15_mn.sh
#   NODES="tomo2 tomo5" ./run_steps15_mn.sh
#   NP=8 ./run_steps15_mn.sh
#   CONF=config_steps15.conf ./run_steps15_mn.sh
#
# Run after ./run_mn.sh and before ./run_step6_mn.sh.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CONF=${CONF:-config_steps15.conf}

source ./mpi_multinode.sh
mn_banner

"$MPIRUN" -n "$NP" $MPI_ARGS ./set_affinity_gpu.sh "$PY" steps15.py "$CONF"
