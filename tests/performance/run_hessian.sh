#!/usr/bin/env bash
# Fused-vs-classic Hessian benchmark across detector sizes.
#
#   ./run_hessian.sh --plan      # sizes + memory for every N below, no GPU, no run
#   ./run_hessian.sh             # accuracy + timing sweep, then the cross-N table
#   MODE=accuracy ./run_hessian.sh
#
# One mpirun per N (each needs the full pinned working set, so they cannot share
# a process), each writing loghess_<N>; the table at the end is built from the
# RESULT line in those logs.
#
# N must be a size cuFFTDx can build a 2N x 2N kernel for -- powers of two in
# practice.  n = 192 (384 x 384), say, fails to compile whatever EPT/FPB it is
# given.  A size that dies takes no row in the tables below; the sweep carries on
# and the exit status reports how many failed.
#
# nobj and ntheta scale with N, off the production n = 2048 geometry
# (nobj = 3072, ntheta = 1800), so every column is a realistic reconstruction of
# that detector size rather than a detector-only sweep.  Set NOBJ / NTHETA to
# pin them across N instead, which isolates the detector-plane cost of a cascade
# sweep from the object-plane cost that dominates it.

# On tomo5 the cuFFTDx JIT cannot find cufft.h (mathDX is the cuda13 build, nvcc
# is the conda 12.9 one) and Rec.__init__ dies before the benchmark starts.  Point
# CPATH at the conda cuFFT headers to get past it; there are no prebuilt sm80 .so
# files, so every size compiles from scratch the first time (~5 min, then cached
# next to conv2d_cufftdx.py).  Harmless where the JIT already works.
export CPATH="${CPATH:-}${CPATH:+:}$(python - <<'PY' 2>/dev/null || true
# nvidia.cufft is a namespace package, so __file__ is None -- use __path__.
import nvidia.cufft, pathlib
print(pathlib.Path(list(nvidia.cufft.__path__)[0]) / 'include')
PY
)"
set -eu
cd "$(dirname "$(readlink -f "$0")")"

NP=${NP:-4}                          # ranks = GPUs
SIZES=${SIZES:-"512 1024 2048"}      # detector sizes to sweep
NDIST=${NDIST:-4}                    # propagation distances
MODE=${MODE:-both}                   # accuracy | timing | both
NREP=${NREP:-5}                      # timed BH iterations per route
CHECK_ITERS=${CHECK_ITERS:-3}        # BH iterations compared in the accuracy pass
LAM_LAPLACIAN=${LAM_LAPLACIAN:-0}    # >0 also exercises LaplacianTerm.hessian3
NDISTCHUNK=${NDISTCHUNK:-0}          # 0 = all distances share one proj upload

# nobj = 3072*N/2048 and ntheta = 1800*N/2048 unless pinned here (empty = let
# test_hessian_fused.py scale them).
NOBJ=${NOBJ:-}
NZOBJ=${NZOBJ:-}
NTHETA=${NTHETA:-}

# Theta chunk size per detector size, the main perf knob.  The chunking pool
# scales as nchunk*nobj^2 and nobj grows with N, so it has to come down; these
# are run.sh's numbers for an 80 GB card.  Override with NCHUNK to force one
# value at every size.
nchunk_for() {
    [ -n "${NCHUNK:-}" ] && { echo "$NCHUNK"; return; }
    case "$1" in
        512)  echo 8  ;;
        1024) echo 16 ;;
        2048) echo 16 ;;
        4096) echo 4  ;;
        8192) echo 1  ;;
        *)    echo 8  ;;
    esac
}

# Only pass --nobj / --ntheta when they were pinned; otherwise the python
# defaults (which scale with N) apply.
size_args() {
    [ -n "$NOBJ" ]   && printf ' --nobj %s'   "$NOBJ"
    [ -n "$NZOBJ" ]  && printf ' --nzobj %s'  "$NZOBJ"
    [ -n "$NTHETA" ] && printf ' --ntheta %s' "$NTHETA"
    return 0
}

if [ "${1:-}" = "--plan" ]; then
    for N in $SIZES; do
        # shellcheck disable=SC2046
        python test_hessian_fused.py --n "$N" $(size_args) --ndist "$NDIST" \
            --nchunk "$(nchunk_for "$N")" --ndistchunk "$NDISTCHUNK" \
            --lam-laplacian "$LAM_LAPLACIAN" --nranks "$NP" --plan
        echo
    done
    exit 0
fi

# One size failing (an unbuildable FFT, an OOM) should not throw away the sizes
# that did run, so the loop keeps going -- but `mpirun | tee` reports tee's
# status, so the real one has to come out of PIPESTATUS or the failure is
# invisible.  It is re-raised as the exit status at the very end.
FAILED=""
for N in $SIZES; do
    NCH=$(nchunk_for "$N")
    LOG="loghess_${N}"
    echo "=== n $N  ndist $NDIST  nchunk $NCH  np $NP  mode $MODE"
    # shellcheck disable=SC2046
    mpirun -np "$NP" ./set_affinity_gpu.sh \
        python test_hessian_fused.py --n "$N" $(size_args) --ndist "$NDIST" \
            --nchunk "$NCH" --ndistchunk "$NDISTCHUNK" --mode "$MODE" \
            --nrep "$NREP" --check-iters "$CHECK_ITERS" \
            --lam-laplacian "$LAM_LAPLACIAN" --log "$LOG" "$@" \
        2>&1 | tee "${LOG}.out"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && { echo "!!! n $N exited $rc -- see ${LOG}.out"; FAILED="$FAILED $N"; }
done

echo
if [ "$MODE" != "timing" ]; then
    echo "=== accuracy: one hessian3 sweep vs three independent hessian sweeps"
    echo "    worst relative difference over $CHECK_ITERS BH iteration(s)"
    # Sweep order, not glob order (loghess_1024 sorts before loghess_512).
    for N in $SIZES; do grep -h '^ACCURACY ' "loghess_${N}.out" 2>/dev/null || true; done | awk '
    BEGIN { printf "%6s %10s %10s %10s %10s %10s %10s %10s %9s\n",
                    "n","B(g,e)","B(e,e)","B(g,g)","symmetry","beta","bottom","alpha","verdict" }
    { delete v; for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
      printf "%6s %10s %10s %10s %10s %10s %10s %10s %9s\n",
             v["n"], v["Bge"], v["Bee"], v["Bgg"], v["symmetry"],
             v["beta"], v["bottom"], v["alpha"], v["verdict"] }'
    echo
fi

if [ "$MODE" != "accuracy" ]; then
    echo "=== timing: fused vs classic hessian, ndist=$NDIST np=$NP  [seconds, min over reps]"
    echo "    beta_old / alpha_old / step_old : classic route, 3 cascade sweeps"
    echo "    step_new                        : fused route, 1 cascade sweep"
    echo "    grad                            : compute_gradient, identical on both"
    for N in $SIZES; do grep -h '^RESULT ' "loghess_${N}.out" 2>/dev/null || true; done | awk '
    BEGIN { printf "%6s %7s %7s %10s %10s %10s %10s %10s %9s %9s\n",
                    "n","nobj","ntheta","beta_old","alpha_old","step_old","step_new","grad","sp_step","sp_iter" }
    { delete v; for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
      printf "%6s %7s %7s %10s %10s %10s %10s %10s %9s %9s\n",
             v["n"], v["nobj"], v["ntheta"], v["beta_old"], v["alpha_old"],
             v["step_old"], v["step_new"], v["grad"],
             v["speedup_step"], v["speedup_iter"] }'
fi

if [ -n "$FAILED" ]; then
    echo
    echo "!!! no results for n:$FAILED"
    exit 1
fi
