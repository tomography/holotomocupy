#!/bin/bash
#PBS -N holotomo_perf
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A 17445
#PBS -j oe
# Single-tile perf benchmark on Polaris (4 x A100 40 GB, 512 GB per node).
# Same measurement as run.sh; nchunk is sized for a 40 GB card and the ranks
# are started by PALS with the Polaris GPU-affinity wrapper.
#
#   qsub run_polaris.sh          # after editing the settings below and -A/select
#   ./run_polaris.sh --plan      # memory check on a login node, nothing queued
#
# select= above and N / NCHUNK below have to be edited together -- the Polaris
# table in Readme.md gives the node count for each size.  n = 8192 does not fit
# a 40 GB card at any node count.

set -eu
cd "${PBS_O_WORKDIR:-$(dirname "$(readlink -f "$0")")}"
[ -f ./env_polaris.sh ] && . ./env_polaris.sh


N=512                               # detector size (nz = n)
NTHETA=$(( 1800 * N / 2048 ))       # projection angles (1800 at n = 2048)
# Theta chunk size for a 40 GB A100 -- powers of two, with the node count the
# Polaris table in Readme.md pairs them with:
#     n         512   1024   2048   4096
#     select      1      1      4     32
#     nchunk     16     32      8      2
NCHUNK=16
NODES=1                             # only used outside a job (--plan); inside
                                    # one the node count comes from PBS

if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
    NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
fi
NP=$(( NODES * 4 ))                 # 4 A100s per Polaris node
LOG="log${N}_${NCHUNK}_${NODES}n"

if [ "${1:-}" = "--plan" ]; then
    python test.py --n "$N" --ntheta "$NTHETA" --nchunk "$NCHUNK" \
                   --nranks "$NP" --plan
    exit 0
fi

echo "=== n $N  ntheta $NTHETA  nchunk $NCHUNK  np $NP on $NODES node(s)"
mpiexec -n "$NP" --ppn 4 --depth=8 --cpu-bind depth --env OMP_NUM_THREADS=4 \
    ./set_affinity_gpu_polaris.sh \
    python test.py --n "$N" --ntheta "$NTHETA" --nchunk "$NCHUNK" \
                   --log "$LOG" "$@"
python parse_perf_log.py "$LOG"
