#!/bin/bash
#PBS -A 17445
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:00:00
#PBS -q prod
#PBS -N AtS1FT300
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
# README.md, "Running it on Polaris".#
# RE-SUBMITTING A FINISHED RUN SILENTLY LOSES WORK.  On a second submission
# correct_correct3D_extra.txt already exists, so pass 1 applies it, and step 7
# then re-measures from an already-corrected volume and OVERWRITES the file
# with the residual.  Step 6 ADDS the file to cshifts_final rather than
# accumulating, so the original correction is lost, not doubled -- the second
# run is worse than the first and nothing errors.  The guard below refuses to
# start pass 1 when the file is present.  After a mid-pass-2 preemption,
# moving the file aside is the WRONG answer: comment out pass 1 and step 7 and
# resubmit pass 2 alone.
#
# THE ROTATION AXIS IS MEASURED BY STEP 4b, not typed in.  center_src=measured
# in config_steps15.conf, so after step 4 and before step 5 steps15 correlates
# opposed (theta, theta+180) pairs of bin-0 PAGANIN projections -- phase, not
# frames -- and folds the answer into cshifts_final.  rotation_center_shift
# must stay 0 in config_steps15.conf AND in all five config_step6_*.conf:
# reader.py adds whatever is there on top, and steps15 cannot see those files.
#
# PER-ANGLE DRIFT COMES ONLY FROM STEP 7.  Nothing in step 3 carries it: the
# ESRF correct_motion.txt / correct_correct3D.txt terms and the step-3 drift
# correlator are all gone.  Pass 1 below reconstructs with the positions
# frozen, step 7 fits the drift from that volume, pass 2 applies it.  So
# losing correct_correct3D_extra.txt loses the whole drift correction, not
# just a refinement of it.
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

# See the re-submission hazard at the top of this file.
EXTRA="${SCRIPT_DIR}/correct_correct3D_extra.txt"
[ -f "${EXTRA}" ] && { echo "ERROR: ${EXTRA} already exists, so pass 1 would apply it and step 7 would then overwrite it with the residual, LOSING the correction. Either this run is finished, or you are resuming: comment out pass 1 and step 7 below and resubmit pass 2 alone."; exit 1; }

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
# start_step=1 if the _rec_paper/<pfile>.h5 has to be rebuilt from the EDFs.
echo "=== steps15 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# PASS 1 ---------------------------------------------------------------------
# Pass 1 runs the *_nopos.conf variants: rho[2]=0, so the positions stay at the
# cshifts_final step 3 computed and step7 below measures the sample's drift
# rather than a residue of a refinement that already happened (step7.py's own
# docstring warns about exactly that bias).  They also write to a separate
# path_out, _rec6_paper_nopos, so pass 2 cannot overwrite the volume step7 was
# derived from and step7 can be re-run or re-tuned without redoing pass 1.

# bin 2: 4x4  n=1024  iters    0 ->1024    positions frozen
echo "=== pass1 bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2_nopos.conf" || exit $?

# bin 1: 2x2  n=2048  iters 1024 ->1280    positions frozen
echo "=== pass1 bin1 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1_nopos.conf" || exit $?

# STEP 7 ---------------------------------------------------------------------
# Fits the per-angle drift and writes correct_correct3D_extra.txt next to the
# configs -- one file, picked up by every step-6 config in this directory via
# correct3d_extra=1, so pass 2 needs no edit.  Under mpiexec it splits its z
# slices over the ranks and allreduces the 256-bin histogram, so the answer
# does not depend on -n -- checked bit-for-bit, 1 rank against 4.
#   -n 8         ~6x one GPU, so ~5 min at --bin 1.  What does not scale is
#                the shift: a vertical shift mixes z, so every rank repeats it,
#                and that is 4% of an evaluation.  -n cannot exceed --nslice.
#   bin1_nopos   read the PASS-1 (frozen) volume out of _rec6_paper_nopos, not the
#                refined pass-2 one.  This is the whole point of the split.
#   --iter 1280  the pass-1 bin-1 hand-off.  The default (0) means "highest
#                checkpoint present", and path_out is shared by both rungs.
#   --bin 1      search on the 1024 grid while reading the better-converged
#                bin-1 volume.  Unit-safe: the written file is in correct3d_bin
#                px whatever grid the search ran on.
#   --ntheta     unset: the default searches a quarter of the scan's angles,
#                on an even stride.  The fit is a low-order polynomial and is
#                written back onto every angle, so the output is unchanged.
echo "=== step7 START $(date) ==="
# the healthcheck may have left fewer nodes than the 2 this wants
STEP7RANKS=$(( NTOTRANKS < 8 ? NTOTRANKS : 8 ))
mpiexec ${HOSTOPT} -n ${STEP7RANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step7.py" "${SCRIPT_DIR}/config_step6_bin1_nopos.conf" --iter 1280 --bin 1 || exit $?

# PASS 2 ---------------------------------------------------------------------
# bin 2 re-seeds from the step-5 Paganin+FBP volume (start_iter=0 makes
# find_latest_checkpoint return None) and this time applies the correction.
# Step 7's drift is only read at bin 2, the one rung with start_iter=0, where
# Reader.read_pos adds it to cshifts_final; bin 1 and bin 0 inherit it through
# the checkpoint, and re-adding would double count.
echo "=== pass2 bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" || exit $?

# bin 1: 2x2  n=2048  iters 1024 ->1280
echo "=== pass2 bin1 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" || exit $?

# bin 0: 1x1  n=4096  iters 1280 ->1800
echo "=== pass2 bin0 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" || exit $?

echo "=== ALL STAGES DONE $(date) ==="

# --- optional one-off diagnostics; not part of a reconstruction --------------
# Run these with -q debug -l select=2 -l walltime=01:00:00 and everything
# above commented out.

# rotation-centre sweep, needs GPUs (stitch + Paganin per candidate)  24 s
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step5_center_sweep.py" "${SCRIPT_DIR}/config_steps15.conf" --start 3 --stop 23 --step 1

# x and z drift through the volume; its own config, see README.md
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/estimate_xshift_ls.py" "${SCRIPT_DIR}/config_xshift.conf"
