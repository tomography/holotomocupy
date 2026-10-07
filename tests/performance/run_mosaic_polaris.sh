#!/bin/bash
#PBS -N holotomo_mosaic
#PBS -l select=32:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -A 17445
#PBS -j oe
# Mosaic perf benchmark on Polaris (4 x A100 40 GB, 512 GB per node).  Same
# measurement as run_mosaic.sh; nchunk is sized for a 40 GB card and the ranks
# are started by PALS with the Polaris GPU-affinity wrapper.
#
#   qsub run_mosaic_polaris.sh   # after editing the settings below and -A/select
#   ./run_mosaic_polaris.sh --plan   # memory check on a login node
#
# select= above and BIN / NCHUNK below have to be edited together -- the
# Polaris table in Readme.md gives the node count for each bin and tile shape.
# bin 0 does not fit a 40 GB card at any node count.

set -eu
cd "${PBS_O_WORKDIR:-$(dirname "$(readlink -f "$0")")}"
[ -f ./env_polaris.sh ] && . ./env_polaris.sh

BIN=0                               # n = nz = 2048>>bin, ntheta = 6000>>bin
NTILE_V=1                           # tile rows
NTILE_H=5                           # tiles per row
NDIST_TILE=4                        # distances per tile -> ndist = v*h*4
# Theta chunk size for a 40 GB A100 -- powers of two, with the node count the
# Polaris table in Readme.md pairs them with:
#     bin              3    2    1
#     select (1x5)     1    1    8
#     select (2x5)     1    2   16
#     nchunk          32    8    2
NCHUNK=4
NDISTCHUNK=0                        # 0 = all distances share one proj upload
NODES=1                             # only used outside a job (--plan); inside
                                    # one the node count comes from PBS

if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
    NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
fi
NP=$(( NODES * 4 ))                 # 4 A100s per Polaris node
LOG="logmosaic${NTILE_V}x${NTILE_H}_${BIN}_${NP}"

if [ "${1:-}" = "--plan" ]; then
    python test_mosaic.py --bin "$BIN" --ntile-v "$NTILE_V" --ntile-h "$NTILE_H" \
                          --ndist-tile "$NDIST_TILE" --nchunk "$NCHUNK" \
                          --ndistchunk "$NDISTCHUNK" --nranks "$NP" --plan
    exit 0
fi

echo "=== bin $BIN  tiles ${NTILE_V}x${NTILE_H}  nchunk $NCHUNK  np $NP on $NODES node(s)"
mpiexec -n "$NP" --ppn 4 --depth=8 --cpu-bind depth --env OMP_NUM_THREADS=4 \
    ./set_affinity_gpu_polaris.sh \
    python test_mosaic.py --bin "$BIN" --ntile-v "$NTILE_V" --ntile-h "$NTILE_H" \
                          --ndist-tile "$NDIST_TILE" --nchunk "$NCHUNK" \
                          --ndistchunk "$NDISTCHUNK" --log "$LOG" "$@"
python parse_perf_log.py "$LOG"
