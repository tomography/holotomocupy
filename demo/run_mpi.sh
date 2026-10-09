#!/usr/bin/env bash
# N GPUs.   ./run_mpi.sh 4 02_holotomography.py [options]
set -eu
here="$(dirname "$(readlink -f "$0")")"
env="/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy"
export PYTHONPATH="${here}/../src"
exec "${env}/bin/mpirun" -np "${1:-4}" \
     "${here}/bind.sh" "${env}/bin/python" "${here}/${2:-02_holotomography.py}" "${@:3}"
