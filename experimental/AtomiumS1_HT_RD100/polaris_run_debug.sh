#!/bin/bash
#PBS -A 17445
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:59:00
#PBS -q debug-scaling
#PBS -N AtS1HT100d
#PBS -j oe
# ===========================================================================
# Atomium S1, 4-distance HT, +-100 px random displacement, 4.5 nm voxels
# -- ESRF ID16A, visit blc17322, beamtime 20260825, collected 2026-08-29.
# FIRST END-TO-END EXERCISE of the RD100 data path -- NOT a reconstruction.
#
#     qsub polaris_run_debug.sh
#
# NOTHING CROSS-CHECKS THE ESTIMATORS ON THIS SCAN.  ESRF left no nabu drop,
# so there is no outside rotation axis, and no rhapp.mat to validate
# estimate_rhapp against.  RD025 is the scan that checks rhapp; RD300 is the
# one that checks the axis.  Run both of those first -- this one inherits
# their verdict, it does not produce its own.
#
# PER-ANGLE DRIFT IS NOT MEASURED HERE AT ALL.  Step 7 fits it from a
# position-frozen pass-1 volume, and this debug job stops at bin 2, well short
# of it.  polaris_run.sh is the script that reaches step 7.
#
# WHAT TO READ IN THE LOG, in order:
#   1. `Step 3: rhapp per-plane mean, object px:` -- the inter-plane residual.
#      ~182 px on this sample; a few px would mean the estimator failed.
#   2. `dy over all N estimates: ...  max_dy=200` -- the vertical spread of
#      the opposed pairs, which is the slow sample drift and is NOT undone.
#      If it approaches 200 the filter is about to start cutting real pairs.
#   3. `Step 3: measured axis ...` -- measured, with no ESRF number to print it
#      against on this scan.  Compare it by hand with RD025's and RD300's: same
#      sample, different mounting, so agreement is reassuring but not required.
#   4. shifts.png -- every shift term and their sum, written by step 3.
#
# Step 3 caches both estimates under <path_out>/measured/ and reuses them on a
# re-run; delete that directory to force a re-measurement.
#
# debug-scaling is capped at 1 hour and 10 nodes, which cannot carry the
# two-pass ladder in polaris_run.sh.  Stage 2 runs bin 2 for as long as the
# clock allows; it checkpoints every 32 iterations, so whatever it reaches is
# on disk and polaris_run.sh can be submitted afterwards with the steps15 line
# commented out.
# ===========================================================================
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/slurm-${PBS_JOBID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Jobid: $PBS_JOBID"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# Drop nodes whose GPUs cannot take a CUDA context; PBS cannot negate a host.
HOSTOPT=""
if [ "${HEALTHCHECK:-1}" = "1" ]; then
    GOOD="${SCRIPT_DIR}/nodes.good.${PBS_JOBID}"
    bash "${rec_dir}/gpu_healthcheck.sh" "${GOOD}" "${NRANKS}" "${RUN_NODES:-1}" || { echo "ERROR: too few healthy nodes in this allocation; aborting."; exit 1; }
    head -n "${RUN_NODES:-$(wc -l < "${GOOD}")}" "${GOOD}" > "${GOOD}.run"
    NNODES=$(wc -l < "${GOOD}.run")
    export NTOTRANKS=$(( NNODES * NRANKS ))
    HOSTOPT="--hostfile ${GOOD}.run"
    echo "Running on ${NNODES} healthy nodes  TOTAL_NUM_RANKS=${NTOTRANKS}"
fi

echo "=== preflight START $(date) ==="
python "${SCRIPT_DIR}/preflight.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# --- STAGE 1: the pipeline proper ----------------------------------------
# Step 3 measures rhapp and the rotation axis itself, by importing
# estimate_rhapp and estimate_center -- they are parts of steps15, not stages
# to run first.  The numbers and the figures come out inside the step-3 log.
echo "=== steps15 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# --- STAGE 2: whatever the clock allows ----------------------------------
echo "=== bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2_nopos.conf" || exit $?

echo "=== DEBUG RUN DONE $(date) ==="
