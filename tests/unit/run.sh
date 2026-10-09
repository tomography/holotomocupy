#!/usr/bin/env bash
# Every unit test: the serial ones, then the MPI ones on 2 ranks.
#
#   ./run.sh              # all of it
#   ./run.sh operators    # only files whose name contains "operators"
#   NP=4 ./run.sh         # a different rank count for the MPI part
set -u
here="$(dirname "$(readlink -f "$0")")"
repo="$(cd "${here}/../.." && pwd)"
export PYTHONPATH="${repo}/src"
PY="${HTC_PYTHON:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}"
MPIRUN="${MPIRUN:-$(dirname "${PY}")/mpirun}"
command -v "${MPIRUN}" >/dev/null 2>&1 || MPIRUN=mpirun
NP="${NP:-2}"
# Never /tmp: a full /tmp breaks everything else on the host.
for d in /local/ssd/vnikitin /local/vnikitin "${HOME}"; do
    [ -n "${HTC_SCRATCH:-}" ] && break
    mkdir -p "$d" 2>/dev/null && export HTC_SCRATCH="$d"
done
[ -n "${CUPY_CACHE_DIR:-}" ] || export CUPY_CACHE_DIR="${HTC_SCRATCH}/.cupy"

want="${1:-}"
rc=0
for f in test_env.py test_operators.py test_derivatives.py test_shift_derivatives.py test_cascade.py; do
    case "$f" in *"${want}"*) ;; *) [ -n "$want" ] && continue ;; esac
    echo; echo "================ $f"
    "${PY}" "${here}/$f" || rc=1
done
case "test_mpi.py" in *"${want}"*)
    echo; echo "================ test_mpi.py   (${NP} ranks)"
    "${MPIRUN}" -np "${NP}" "${here}/../../demo/bind.sh" "${PY}" \
        "${here}/test_mpi.py" || rc=1 ;;
esac
echo
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "SOME UNIT TESTS FAILED"
exit $rc
