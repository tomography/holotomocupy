#!/usr/bin/env bash
# Reconstruct the synthetic mosaic dataset (STEP 6, the BH solver).
#
#   ./run_step6.sh                 # NP=1
#   NP=8 ./run_step6.sh
#   CONF=config_step6.conf ./run_step6.sh
#
# Run after ./run_steps15.sh, which writes the initial object.
# One rank per GPU; set_affinity_gpu.sh pins each rank to its own device.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CONF=${CONF:-config_step6.conf}
NP=${NP:-1}
PY=${PY:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}
MPIRUN=${MPIRUN:-mpirun}

"$MPIRUN" -n "$NP" ./set_affinity_gpu.sh "$PY" step6.py "$CONF"
