#!/bin/bash
#PBS -A 17445
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:20:00
#PBS -q prod
#PBS -N Y350cp12ip1
#PBS -j oe

# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_run.sh testing).
# Plain $(pwd) does NOT work: PBS starts the job in $HOME, not where you qsub'd.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/slurm-${PBS_JOBID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

# Modules + conda env. env.sh loads PrgEnv-gnu, cray-mpich, cudatoolkit,
# cray-hdf5-parallel and activates the holotomocupy env; it must be sourced
# inside the job, not just at install time, or the cray-mpich-linked mpi4py
# and h5py will not find their libraries.
[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# PBS can pin a host but cannot negate one, so a node that comes up with
# cudaErrorDevicesUnavailable can only be filtered from inside the job.
# Must run AFTER the env is sourced: the probe needs cupy.
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

# NFP probe retrieval -- OPT-IN.  Uncomment prb_file in config_step6.conf after.
# echo "=== nfp START $(date) ==="
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step0.py" "${SCRIPT_DIR}/config_step0.conf" || exit $?

# INTENSITY misfit, psf_sigma = 1.2 raw px -- the model twin of psf12.
#
#   amplitude  F0 = 1/N sum W (a        - sqrt(d))^2     <- psf12
#   intensity  F0 = 1/N sum W (a^2      - d      )^2     <- here,  a = sqrt(K|psi|^2)
#
# Same psf_sigma, same rho, same lam_prbfit, same ladder.  lam_laplacian
# is x4 (3e-5 -> 1.2e-4) because the intensity F0 is ~4x larger: the
# object penalty has to be rescaled or it would be a 4x lighter
# regularizer than in the amplitude arm.  err is NOT comparable between
# the two models for the same reason.
#
# HALF SET p1 = odd angles; positions frozen at this
# arm's OWN full run, checkpoint_1536.h5 in ..._psf12_int.
#
#     qsub polaris_run_psf12_int_p1.sh
#
# RUN ONLY AFTER the full psf12_int arm has written checkpoint_1536.h5.

# bin 2: 4x4  n=512   nobj=640   iters    0 -> 1024
echo "=== psf12_int p1 bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_psf12_int_p1_bin2.conf" || exit $?

# bin 1: 2x2  n=1024  nobj=1280  iters 1024 -> 1280
echo "=== psf12_int p1 bin1 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_psf12_int_p1_bin1.conf" || exit $?

# bin 0: 1x1  n=2048  nobj=2560  iters 1280 -> 1536
echo "=== psf12_int p1 bin0 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_psf12_int_p1_bin0.conf" || exit $?

echo "=== psf12_int p1 DONE $(date) ==="
