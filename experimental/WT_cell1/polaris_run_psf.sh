#!/bin/bash
#PBS -A 17445
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -N WTpsf
#PBS -j oe
# ===========================================================================
# One arm of the WT_cell1 psf_sigma sweep.  Same as polaris_run.sh except the
# config comes in from the environment and the job fits the debug queue:
#
#     qsub -v CFG=config_psf12.conf polaris_run_psf.sh
#
# 1025 iterations take ~7 min on 2 nodes, so 30 min is ample.  Use
# sweep_driver_psf.sh to walk all six arms -- the debug queue only lets one
# job per user be queued at a time.  See README.md.
# ===========================================================================

# --- user configuration ---
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}
CFG=${CFG:-config_psf00.conf}
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Config:      ${CFG}"
echo "Jobid: $PBS_JOBID"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# Drop nodes whose GPUs cannot take a CUDA context; PBS cannot negate a host,
# so this can only be filtered from inside the job.  Needs cupy, so it runs
# after polaris_env.sh.
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

echo "=== nfp START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step0.py" "${SCRIPT_DIR}/${CFG}" || exit $?

echo "=== ALL STAGES DONE $(date) ==="
