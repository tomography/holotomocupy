#!/bin/bash
#PBS -A 17445
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:59:00
#PBS -q debug-scaling
#PBS -N AtS1HT300d
#PBS -j oe
# ===========================================================================
# STEP 3 ONWARDS, to exercise the new drift term -- NOT a reconstruction.
#
#     qsub polaris_run_debug.sh
#
# config_steps15.conf already has start_step=3, so this picks up the h5 that
# steps 1-2 wrote and runs 3 (shifts), 4 (resample) and 5 (Paganin + FBP).
# It stops there: the iterative ladder is polaris_run.sh's job, and bin 2
# would not finish inside the hour this queue allows.
#
# WHAT IT IS CHECKING.  Step 3 now carries THREE measured terms, not two:
#
#   random        the commanded displacement, read from the tables
#   motion        NEW -- sample drift from the post-scan retakes,
#                 estimate_quali_motion.py
#   rhapp         inter-plane residual, estimate_rhapp.py
#
# plus the rotation axis in step 4b.  The drift is the new one and this scan
# is where it can be judged: HT_RD300 has both a reference_motion.mat
# (8.27 binned = 16.54 raw px vertical) and a quali.mat, so step 3's
# `--validate` block prints ours against both.
#
# Run the estimator HERE, not on a login node: it is a handful of 4096^2 FFTs
# per plane and the login nodes sit at load ~70, where one of them costs 69 s.
#
# RHAPP IS MEASURED WITH THE DRIFT ALREADY UNDONE.  It re-reads the shift
# tables itself rather than taking cshifts_final, so the motion array is
# passed into it; without that the drift would be measured twice and added
# twice.  The per-plane numbers to read back are in the step-3 log, and
# shifts.png next to the config draws all three terms against each other.
#
# STEP 7 STILL OWNS THE RESIDUAL DRIFT.  Three retake points catch the bulk,
# not the structure between them; step 7 fits what is left out of a
# position-frozen pass-1 volume, and this job stops well short of that.
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

# All three estimators run inside step 3 and cache to <path_out>/measured/.
# Delete motion_measured.npy to force a re-measure; the rhapp cache is then
# rejected as stale on purpose, because it was measured against the old drift.
echo "=== steps15 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

echo "=== DEBUG RUN DONE $(date) ==="
echo "Read back: the 'motion per-plane ptp' and 'rhapp per-plane mean' lines,"
echo "the 'Step 4b: axis verify -- residual' line, and ${SCRIPT_DIR}/shifts.png"
