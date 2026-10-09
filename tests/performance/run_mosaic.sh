#!/usr/bin/env bash
# Mosaic perf benchmark: BH iterations of an NTILE_V x NTILE_H x 4-distance
# scan at binning level BIN.  Two mosaic shapes are interesting:
#
#   1 x 5 tiles -> ndist = 20   (the default: a single row, wide flat object)
#   2 x 5 tiles -> ndist = 40   (two rows, 1.5x taller object)
#
#   ./run_mosaic.sh --plan   # sizes + memory for the settings below, no GPU
#   ./run_mosaic.sh          # run it, then print the timing breakdown
#
# Everything is set here in the script -- edit and rerun.  Node counts per bin
# are in "Running on a cluster" in Readme.md.

set -eu
cd "$(dirname "$(readlink -f "$0")")"

NP=4                                # ranks = GPUs
BIN=2                               # n = nz = 2048>>bin, ntheta = 6000>>bin
NTILE_V=1                           # tile rows
NTILE_H=5                           # tiles per row
NDIST_TILE=4                        # distances per tile -> ndist = v*h*4
# Theta chunk size, the main perf knob.  The chunking pool scales as
# nchunk * nzobj * nobj, so it has to come down as the object grows.
# Powers of two, from the 80 GB / 8-rank-node tables in Readme.md:
#     bin       3    2    1    0
#     nchunk   16   16    4    1
NCHUNK=4
# 0 = all ndist distances share one upload of a theta chunk of proj (the
# default); 1 = the old outer-distance loop.  For these mosaic shapes the
# chunking pool has enough object-plane headroom that all of them are free
# (the crossover is 144 distances at 1x5, 108 at 2x5), so leave it at 0.
NDISTCHUNK=0
LOG="logmosaic${NTILE_V}x${NTILE_H}_${BIN}_${NP}"

if [ "${1:-}" = "--plan" ]; then
    python test_mosaic.py --bin "$BIN" --ntile-v "$NTILE_V" --ntile-h "$NTILE_H" \
                          --ndist-tile "$NDIST_TILE" --nchunk "$NCHUNK" \
                          --ndistchunk "$NDISTCHUNK" --nranks "$NP" --plan
    exit 0
fi

echo "=== bin $BIN  tiles ${NTILE_V}x${NTILE_H}  nchunk $NCHUNK  ndistchunk $NDISTCHUNK  np $NP"
mpirun -np "$NP" ./set_affinity_gpu.sh \
    python test_mosaic.py --bin "$BIN" --ntile-v "$NTILE_V" --ntile-h "$NTILE_H" \
                          --ndist-tile "$NDIST_TILE" --nchunk "$NCHUNK" \
                          --ndistchunk "$NDISTCHUNK" --log "$LOG" "$@"
python parse_perf_log.py "$LOG"
