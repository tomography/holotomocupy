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
# NOTE: the AtomiumL1 raw data lives on tomo5 at
#     /data3/vnikitin/ESRF/atomium/20250607/AtomiumL1
# NOT on eagle.  Before using this script, stage the EDF directories to eagle
# and repoint `path`/`path_out` in every conf file; until then use
# ./local_run.sh on tomo5 instead.
#
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

# step0.py is disabled for this dataset -- see config_step0.conf.
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/${SCRIPT}" "${SCRIPT_DIR}/${CONFIG}"
