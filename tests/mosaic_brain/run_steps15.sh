#!/usr/bin/env bash
# STEP 5 for the synthetic mosaic: stitch -> Paganin -> FBP -> {pfile}_obj.h5.
#
#   ./run_steps15.sh               # NP=1
#   NP=8 ./run_steps15.sh
#   CONF=config_steps15.conf ./run_steps15.sh
#
# Run after ./run.sh (make_geometry.py + gen_data.py) and before ./run_step6.sh.
# One rank per GPU; set_affinity_gpu.sh pins each rank to its own device.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CONF=${CONF:-config_steps15.conf}
NP=${NP:-1}
PY=${PY:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}
MPIRUN=${MPIRUN:-mpirun}

"$MPIRUN" -n "$NP" ./set_affinity_gpu.sh "$PY" steps15.py "$CONF"
