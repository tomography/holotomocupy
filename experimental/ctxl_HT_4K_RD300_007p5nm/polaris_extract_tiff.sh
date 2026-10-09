#!/bin/bash
#PBS -A 17445
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -N ctxl0075tiff
#PBS -j oe
# ===========================================================================
# Re(obj) of the three ctxl step-6 reconstructions, checkpoint 1472, out as
# cropped TIFF stacks:
#
#     qsub polaris_extract_tiff.sh
#
#     full  <- ..._rec6      (all 4000 projections)   -> <rec6_results>/full/
#     even  <- ..._rec6_p0   (projections 0,2,...)    -> <rec6_results>/even/
#     odd   <- ..._rec6_p1   (projections 1,3,...)    -> <rec6_results>/odd/
#
# 768 px trimmed off each y/x border and 256 slices off the top and bottom in
# z, so 4544 slices of 3520 x 3520 float32 = 47 MiB a slice, 210 GiB a set,
# ~629 GiB for the three.  Eagle had 189 T free and APS_IRI 189 T of headroom
# under its 500 T quota when this was written; check `myprojectquotas` if that
# is no longer true.
#
# File numbering restarts at zero: 0000.tiff is global z = 256 and 4543.tiff is
# global z = 4799.  All three sets are cropped the same way, so full/even/odd
# stay slice-aligned and an FSC can be taken file-by-file.
#
# NO GPU IS USED -- this is pure Lustre I/O, which is why it is one node in
# the debug queue rather than the GPU ladder's 24.  24 ranks, 8 per set.  The
# three checkpoints sit on three different OSTs (7, 105, 35), so the reads do
# not contend.  The z crop skips 10% of each volume and the tighter y/x crop
# writes a third less than the 512 px version, so expect roughly 15-30 min.
#
# RESTART: rerunning skips every slice whose TIFF is already there with the
# right shape, so if the hour runs out just qsub it again.  Changing --crop or
# --zcrop makes the existing stack the wrong shape and it is rewritten.
#
# --clean below deletes numbered slices PAST THE END of the current stack --
# what a smaller --zcrop left behind, e.g. 4544.tiff..5055.tiff from a run with
# no z crop.  It is there so a plain `qsub` after a crop change just works.
# Nothing outside <rec6_results>/{full,even,odd}/NNNN.tiff is ever touched.
# ===========================================================================

HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}
NRANKS=${NRANKS:-24}          # ranks per node; 8 per set with the default 3 sets
NDEPTH=2
NTHREADS=1                    # numpy/tifffile here are single-threaded copies

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
exec > >(tee "${SCRIPT_DIR}/tiff-${PBS_JOBID}.out") 2>&1

echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "NRANKS=${NRANKS}"

[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

echo "=== START $(date) ==="
mpiexec -n ${NRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth \
    --env OMP_NUM_THREADS=${NTHREADS} \
    python "${SCRIPT_DIR}/extract_tiff.py" --iter 1472 --crop 768 --zcrop 256 \
    --clean || exit $?
echo "=== DONE $(date) ==="
