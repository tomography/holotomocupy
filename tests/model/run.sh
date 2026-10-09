#!/usr/bin/env bash
# Run the misfit-model comparison on a local GPU.  See tests/shift/run.sh for
# why the module load and the PYTHONPATH are needed.
#
#   ./run.sh                          # test_model_compare.py
#   ./run.sh test_model_compare.py --niter 96
set -eu
here="$(dirname "$(readlink -f "$0")")"
repo="$(cd "${here}/../.." && pwd)"

if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
    module load nvhpc-hpcx-cuda13/26.3 2>/dev/null || true
fi

export PYTHONPATH="${repo}/src"
# Never /tmp: a full /tmp on handyn breaks everything else running there.
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/local/ssd/vnikitin/.cupy}"

exec /home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python \
     "${here}/${1:-test_model_compare.py}" "${@:2}"
