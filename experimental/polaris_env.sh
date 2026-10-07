#!/bin/bash
# Software environment for holotomocupy on ALCF Polaris.  Source it INSIDE the
# PBS job -- batch jobs do not read ~/.bashrc:
#
#     source <repo>/experimental/polaris_env.sh
#
# HTC_VENV overrides the venv; HTC_ENV_CHECK=1 prints what actually resolved.
#
# The venv is layered on the `conda` module, so mpi4py and h5py come from
# ALCF's own build and nothing has to be compiled against the Cray wrappers.
# (It used to be a hand-built env under /eagle; that is retired.)

module use /soft/modulefiles
module load conda
conda activate base
CONDA_NAME=$(echo "${CONDA_PREFIX}" | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
source "${HTC_VENV:-/home/vvnikitin/venvs/${CONDA_NAME}}/bin/activate"

export MATHDX_ROOT=${MATHDX_ROOT:-/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12}

# GPU-aware MPI off: nothing hands MPI a device pointer (the Alltoallw buffer
# is pinned HOST memory, allreduces take host arrays or an explicit .get()).
export MPICH_GPU_SUPPORT_ENABLED=0

# --- parallel HDF5 / MPI-IO on Lustre ---------------------------------------
# Every hang seen on Polaris has been at an h5py driver="mpio" call.  None of
# this was needed on tomo5; all of it is /eagle being Lustre.
#
# HDF5 takes an flock() on open and Lustre serialises it across ranks, so a
# 40-96 rank open can block indefinitely.  We never mix readers and writers on
# one file, so turning it off risks nothing.
export HDF5_USE_FILE_LOCKING=FALSE
# reader.read_data indexes with a list of theta ids, an irregular selection.
# With ROMIO data sieving on, a scattered read of a few GB turns into tens of
# GB through a 512 KB sieve buffer.
export MPICH_MPIIO_HINTS="${MPICH_MPIIO_HINTS:-*:romio_ds_read=disable:romio_ds_write=disable:romio_cb_read=enable:romio_cb_write=enable}"
# Striping is a property of the DIRECTORY and must be set before the file is
# created -- eagle defaults to stripe_count=1, i.e. one OST for a 134 GB h5:
#     lfs setstripe -c 8 -S 16M <output dir>
# Changing the directory does not restripe files already in it.
#
# HTC_MPIIO_STATS=1 prints per-file MPI-IO counters at exit; noisy, but the
# fastest way to tell a blocked job from one doing 50x the I/O it should.
if [ "${HTC_MPIIO_STATS:-0}" = "1" ]; then
    export MPICH_MPIIO_STATS=1
    export MPICH_MPIIO_TIMERS=1
fi

if [ "${HTC_ENV_CHECK:-0}" = "1" ]; then
    echo "python  : $(which python)"
    # Import mpi4py.MPI, not mpi4py: the bare package is pure Python and
    # imports even when the compiled extension cannot find its libmpi.
    python -c "from mpi4py import MPI; import mpi4py, h5py, cupy; print('mpi4py', mpi4py.__version__, MPI.Get_library_version().split(chr(10))[0], '| h5py', h5py.__version__, 'mpi=', h5py.get_config().mpi, '| cupy', cupy.__version__)"
fi
