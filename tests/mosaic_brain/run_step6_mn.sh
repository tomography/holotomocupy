#!/usr/bin/env bash
# Multi-node analog of ./run_step6.sh -- the BH solver.
#
#   ./run_step6_mn.sh
#   NODES="tomo4 tomo5" ./run_step6_mn.sh
#   NP=11 ./run_step6_mn.sh
#   CONF=config_step6.conf ./run_step6_mn.sh
#
# Run after ./run_steps15_mn.sh, which writes the initial object.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CONF=${CONF:-config_step6.conf}

source ./mpi_multinode.sh
mn_banner

"$MPIRUN" -n "$NP" $MPI_ARGS ./set_affinity_gpu.sh "$PY" step6.py "$CONF"
