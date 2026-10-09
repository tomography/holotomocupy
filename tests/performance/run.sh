#!/usr/bin/env bash
# Single-tile perf benchmark: BH iterations on a square detector of size N,
# ndist = 4 propagation distances, one tile.  The mosaic counterpart is
# run_mosaic.sh, the Polaris job script is run_polaris.sh.
#
#   ./run.sh --plan     # sizes + memory for the settings below, no GPU, no run
#   ./run.sh            # run it, then print the timing breakdown
#
# Everything is set here in the script -- edit and rerun.  The big sizes need
# many ranks; see "Running on a cluster" in Readme.md for the node counts and
# the per-size nchunk / ndistchunk that fit an 80 GB card.

set -eu
cd "$(dirname "$(readlink -f "$0")")"

NP=4                                # ranks = GPUs
N=2048                               # detector size (nz = n)
NTHETA=$(( 1800 * N / 2048 ))       # projection angles (1800 at n = 2048)
# Theta chunk size, the main perf knob.  The chunking pool scales as
# nchunk * nobj^2 (nobj = 1.59n), so it has to come down fast as n grows.
# Powers of two, from the 80 GB / 8-rank-node table in Readme.md:
#     n         512   1024   2048   4096   8192
#     nchunk      8     16     16      4      1
NCHUNK=8
# 0 = all ndist distances share one upload of a theta chunk of proj.  Here
# nzobj == nobj, so the pool has no headroom and every extra resident distance
# costs memory from the first one; at n = 8192 use 2, which is the difference
# between fitting an 80 GB card and not.
NDISTCHUNK=0
LOG="log${N}_${NCHUNK}"

if [ "${1:-}" = "--plan" ]; then
    python test.py --n "$N" --ntheta "$NTHETA" --nchunk "$NCHUNK" \
                   --ndistchunk "$NDISTCHUNK" --nranks "$NP" --plan
    exit 0
fi

echo "=== n $N  ntheta $NTHETA  nchunk $NCHUNK  ndistchunk $NDISTCHUNK  np $NP"
mpirun -np "$NP" ./set_affinity_gpu.sh \
    python test.py --n "$N" --ntheta "$NTHETA" --nchunk "$NCHUNK" \
                   --ndistchunk "$NDISTCHUNK" --log "$LOG" "$@"
python parse_perf_log.py "$LOG"
