#!/bin/bash
#PBS -A 17445
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=18:00:00
#PBS -q preemptable
#PBS -N AtomS145
#PBS -j oe
# ===========================================================================
# Atomium S1, SINGLE-distance FT, +-300 px random displacement, 4.5 nm voxels
# -- ESRF ID16A, proposal blc17322, beamtime 20260825, scan _0001.
# THE WHOLE PIPELINE IN ONE JOB:
#
#     qsub polaris_run.sh
#
# To run only part of it -- steps 1-5 already done, or resuming after a
# preemption -- COMMENT OUT the mpiexec lines at the bottom that you do not
# want.  Each line ends in `|| exit $?` so a failed stage stops the job
# instead of letting the next level seed itself from a checkpoint that was
# never written.
#
# The three optional diagnostic stages at the very bottom (centre sweep,
# x-shift fit) stay commented; they are one-off measurements, not part of a
# reconstruction.  For those the debug queues are enough -- add
# `-q debug -l select=2 -l walltime=01:00:00` on the qsub line.
#
# What each stage does, how to resume one that ran out of walltime, the
# measured walltimes and disk sizes, and what is and is not corrected: see
# README.md, "Running it on Polaris".
# ===========================================================================

# --- user configuration ---
# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
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

# Drop nodes whose GPUs cannot take a CUDA context.  PBS has no Slurm-style
# --exclude -- `-l select=` can pin a host but cannot negate one -- so a node
# that comes up with cudaErrorDevicesUnavailable can only be filtered from
# inside the job.  Must run AFTER env.sh: the probe needs cupy.
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

# --- the pipeline; comment out a line to skip that stage ---------------------

# NFP probe retrieval -- OPT-IN, not part of the default flow.  Needs
# RAW_DATA/Atomium_S1/<pfile>/scan000[45]/balor_0000.h5 on eagle (there since
# 2026-09-12); step0.py refuses rather than read the zeros HDF5 returns for an
# unresolvable virtual source.  Check first with `python nx_frames.py <the NFP
# .nx>`; afterwards uncomment prb_file in config_step6_bin2.conf.  This is the
# only stage here that touches RAW_DATA.
#
# NOTE THE RANK COUNT.  This is 50 frames: get_local_chunk over 96 ranks
# leaves 46 ranks with ZERO.  Run it with RUN_NODES=2, not in this 24-node job.
# echo "=== nfp START $(date) ==="
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step0.py" "${SCRIPT_DIR}/config_step0.conf" || exit $?

# shifts, binned data, Paganin+FBP.  config_steps15.conf has start_step=3, so
# the EDF->HDF5 conversion and the preprocessing are NOT redone -- set
# start_step=1 if the _rec/<pfile>.h5 has to be rebuilt from the EDFs.
echo "=== steps15 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# bin 2: 4x4  n=1024  iters    0 ->1024
echo "=== bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" || exit $?

# bin 1: 2x2  n=2048  iters 1024 ->1280
echo "=== bin1 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" || exit $?

# bin 0: 1x1  n=4096  iters 1280 ->1800
echo "=== bin0 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" || exit $?

echo "=== ALL STAGES DONE $(date) ==="

# --- optional one-off diagnostics; not part of a reconstruction --------------
# Run these with -q debug -l select=2 -l walltime=01:00:00 and everything
# above commented out.

# rotation-centre sweep, needs GPUs (stitch + Paganin per candidate)  24 s
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step5_center_sweep.py" "${SCRIPT_DIR}/config_steps15.conf" --start 3 --stop 23 --step 1

# x and z drift through the volume; its own config, see README.md
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/estimate_xshift_ls.py" "${SCRIPT_DIR}/config_xshift.conf"
