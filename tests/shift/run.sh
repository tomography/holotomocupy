#!/usr/bin/env bash
# Run the shift tests on a local GPU.
#
# A non-interactive shell skips the module lines in .bashrc, so libmpi.so is
# missing and `import holotomocupy.shift` dies in logger_config's mpi4py
# import; the conda holotomocupy env also has an older holotomocupy installed
# than this repo, so PYTHONPATH has to point at src/.
#
#   ./run.sh                     # test_shift_fft.py
#   ./run.sh test.py             # any other script in this directory
set -eu
here="$(dirname "$(readlink -f "$0")")"
repo="$(cd "${here}/../.." && pwd)"

# Best effort: tomo5 needs this for libmpi.so, handyn already has it and does
# not carry the modulefile, so a failure here is not fatal.
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
    module load nvhpc-hpcx-cuda13/26.3 2>/dev/null || true
fi

export PYTHONPATH="${repo}/src"
# Never /tmp: a full /tmp on handyn breaks everything else running there.
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/local/ssd/vnikitin/.cupy}"

exec /home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python \
     "${here}/${1:-test_shift_fft.py}" "${@:2}"
