#!/usr/bin/env bash
# Generate the synthetic mosaic dataset -- the first stage of
#
#   ./run.sh          make_geometry.py + gen_data.py   -> {pfile}_{tile}.h5, _prb.h5
#   ./run_steps15.sh  steps15.py                       -> {pfile}_obj.h5  (Paganin+FBP)
#   ./run_step6.sh    step6.py                         -> BH reconstruction
#
#   ./run.sh                       # geometry + data
#   ./run.sh --plan                # sizes only, no GPU
#   NP=4 ./run.sh
#   CONF=config_gen.conf ./run.sh
#
# All angles are generated in one pass; ./run.sh --plan prints the pinned host
# memory that needs, per rank, before anything is allocated.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CONF=${CONF:-config_gen.conf}
NP=${NP:-1}
PY=${PY:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}
MPIRUN=${MPIRUN:-mpirun}

if [ "${1:-}" = "--plan" ]; then
    "$PY" gen_data.py "$CONF" --plan
    exit 0
fi

"$PY" make_geometry.py "$CONF" ${FROM_H5:+--from-h5 "$FROM_H5"}
"$MPIRUN" -n "$NP" ./set_affinity_gpu.sh "$PY" gen_data.py "$CONF"
