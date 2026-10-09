#!/bin/bash
#PBS -A 17445
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=0:59:00
#PBS -q debug
#PBS -N holotomo
#PBS -j oe

# --- user configuration ---
# Hierarchical reconstruction: run the three levels in order.
#     qsub -v CONFIG=config_step6_bin2.conf polaris_run.sh   # 4x4, iters   0- 512
#     qsub -v CONFIG=config_step6_bin1.conf polaris_run.sh   # 2x2, iters 512- 768
#     qsub -v CONFIG=config_step6_bin0.conf polaris_run.sh   # 1x1, iters 768-1024
# All three share one path_out, so each level picks up the previous level's
# checkpoint automatically (start_iter) and upsamples obj/prb/pos onto its grid.
# (steps 1-5 first:  qsub -v CONFIG=config_steps15.conf,SCRIPT=steps15.py polaris_run.sh)
CONFIG=${CONFIG:-config_step6_bin2.conf}
SCRIPT=${SCRIPT:-step6.py}
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

# --- drop nodes whose GPUs cannot take a CUDA context ----------------------
# PBS has no Slurm-style --exclude: `-l select=` can pin a host (host=/vnode=)
# but cannot negate one, so a node that comes up with
#   cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or unavailable
# can only be filtered from inside the job.  This must run AFTER env.sh is
# sourced -- the probe needs cupy.
#
#   RUN_NODES=N   run on exactly N healthy nodes (over-request N+1 in select=
#                 so one bad node costs a node-hour, not the whole job).
#                 Default: use every healthy node.
#   HEALTHCHECK=0 skip the probe entirely (~30 s).
HOSTOPT=""
if [ "${HEALTHCHECK:-1}" = "1" ]; then
    GOOD="${SCRIPT_DIR}/nodes.good.${PBS_JOBID}"
    if bash "${rec_dir}/gpu_healthcheck.sh" "${GOOD}" "${NRANKS}" "${RUN_NODES:-1}"; then
        NEED=${RUN_NODES:-$(wc -l < "${GOOD}")}
        head -n "${NEED}" "${GOOD}" > "${GOOD}.run"
        NNODES=$(wc -l < "${GOOD}.run")
        export NTOTRANKS=$(( NNODES * NRANKS ))
        HOSTOPT="--hostfile ${GOOD}.run"
        echo "Running on ${NNODES} healthy nodes  TOTAL_NUM_RANKS=${NTOTRANKS}"
    else
        echo "ERROR: too few healthy nodes in this allocation; aborting."
        exit 1
    fi
fi

mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf"
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf"
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf"
