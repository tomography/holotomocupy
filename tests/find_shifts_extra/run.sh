#!/usr/bin/env bash
# Run one of the autofocus tests on a local GPU.  Everything is synthetic, so
# no mounted data is needed and any GPU host will do.
#
#   ./run.sh                               # test_find_shifts.py, defaults
#   ./run.sh test_find_shifts_null.py
set -eu
here="$(dirname "$(readlink -f "$0")")"
repo="$(cd "${here}/../.." && pwd)"

if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
    module load nvhpc-hpcx-cuda13/26.3 2>/dev/null || true
fi

export PYTHONPATH="${repo}/src"
# Never /tmp: a full /tmp breaks everything else on the host.
if [ -z "${CUPY_CACHE_DIR:-}" ]; then
    for d in /local/ssd/vnikitin /local/vnikitin "${HOME}"; do
        if mkdir -p "$d/.cupy" 2>/dev/null; then
            export CUPY_CACHE_DIR="$d/.cupy"
            break
        fi
    done
fi

script="${1:-test_find_shifts.py}"
case "$script" in -*) script=test_find_shifts.py ;; *) shift || true ;; esac

exec /home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python \
     "${here}/${script}" "$@"
