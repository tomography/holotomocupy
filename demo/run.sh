#!/usr/bin/env bash
# One GPU.   ./run.sh 02_holotomography.py [options]
set -eu
here="$(dirname "$(readlink -f "$0")")"
env="/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy"
export PYTHONPATH="${here}/../src"
exec "${env}/bin/python" "${here}/${1:-02_holotomography.py}" "${@:2}"
