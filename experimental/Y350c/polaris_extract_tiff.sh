#!/bin/bash
#PBS -A 17445
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -N Y350ctiff
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

# Re(obj) at iteration 1536 -> cropped TIFF stacks, for FSC.
#
# Two psf arms, each as even (p0) / odd (p1):
#   /eagle/APS_IRI/vnikitin/20240515/Y350c_rec6_psf00_results/{even,odd}
#   /eagle/APS_IRI/vnikitin/20240515/Y350c_rec6_psf12_results/{even,odd}
#
# 2560^3 cropped by 30 px a side to 2500^3: 23.8 MiB a slice, 58 GiB a set,
# 233 GiB for all four.  Uncompressed -- these feed FSC, and zlib costs more
# on read than the disk.
#
# No GPU is used, so set_affinity_gpu_polaris.sh is NOT in the mpiexec line;
# ranks are dealt over the two sets and then over contiguous z blocks.
# Restartable: a slice is skipped if its TIFF exists with the right shape, so
# if this hits the walltime just qsub it again.
#
#     qsub polaris_extract_tiff.sh

for ARM in psf00 psf12; do
    echo "=== $ARM  $(date) ==="
    mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} python "${SCRIPT_DIR}/extract_tiff.py" \
        --iter 1536 --crop 30 --zcrop 30 \
        --dir even=rec6_${ARM}_p0 --dir odd=rec6_${ARM}_p1 \
        --out /eagle/APS_IRI/vnikitin/20240515/Y350c_rec6_${ARM}_results || exit $?
done
echo "=== DONE $(date) ==="
