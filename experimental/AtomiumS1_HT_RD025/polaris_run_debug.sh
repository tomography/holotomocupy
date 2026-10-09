#!/bin/bash
#PBS -A 17445
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:00:00
#PBS -q prod
#PBS -N AtS1HT025d
#PBS -j oe
# ===========================================================================
# FIRST END-TO-END EXERCISE of the RD025 data path -- NOT a reconstruction.
#
#     qsub polaris_run_debug.sh
#
# debug-scaling is capped at 1 hour and 10 nodes, which cannot carry the
# two-pass ladder in polaris_run.sh.  This script runs preflight, steps15 and
# then bin 2 for as long as the clock allows; bin 2 checkpoints every 32
# iterations, so whatever it reaches is on disk and polaris_run.sh can be
# submitted afterwards with the steps15 line commented out.
#
# What it is actually testing: the esrf_layout link-table / ewoks-fallback
# path on a scan with no master .nx, the preflight, and step 3's two
# estimators -- rhapp from the frames and the rotation axis from opposed
# pairs.  Neither is inherited from ESRF.
#
# THIS IS THE SCAN THAT CHECKS THE RHAPP ESTIMATOR.  RD025 is the one HT scan
# with an ESRF rhapp.mat, so it is where the measurement can be compared with
# an outside number -- by hand, with estimate_rhapp.py --validate; step 3
# itself never reads the .mat.  Expect ~182 object px of inter-plane residual.
#
# PER-ANGLE DRIFT IS NOT MEASURED HERE AT ALL.  Step 7 fits it from a
# position-frozen pass-1 volume, and this debug job stops at bin 2, well
# short of it.  polaris_run.sh is the script that reaches step 7.
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

# steps15 measures rhapp and the rotation axis itself, inside step 3, caching
# both to <path_out>/measured/.  No separate stage here: the estimators are
# part of the pipeline, not something to remember to run first.  Run them by
# hand only to inspect or validate, e.g.
#     python estimate_rhapp.py config_steps15.conf --validate
echo "=== steps15 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# bin 2 with positions frozen (rho[2]=0) -- pass 1 of the two-pass ladder,
# matching polaris_run.sh.  Position refinement needs the step-7 correct3d
# that does not exist yet.
echo "=== bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2_nopos.conf" || exit $?

echo "=== DEBUG RUN DONE $(date) ==="
