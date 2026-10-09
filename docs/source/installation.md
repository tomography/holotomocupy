# Installation

## Requirements

One NVIDIA GPU per MPI rank, and a CUDA 12 driver (use `cupy-cuda13x` for
CUDA 13).

Two dependencies are easy to get wrong and are the usual cause of a broken
install:

* **An MPI-enabled h5py.** `Writer` opens its checkpoints with
  `driver='mpio'`; the default serial h5py raises there. The conda build you
  want is `h5py=*=mpi_openmpi_*`.
* **mpi4py.** `logger_config` imports it, so it is needed even for a
  single-GPU run.

## Conda (recommended)

```bash
conda env create -f environment.yml
conda activate holotomocupy
pip install -e .
```

`environment.yml` pins OpenMPI and the MPI h5py build. `requirements.txt` is
the pip-only list, but you still have to supply MPI and a parallel h5py
yourself.

## Verify the install

```bash
tests/unit/run.sh
```

148 checks in about a minute on one GPU: imports, GPU, MPI and parallel HDF5;
the adjointness of every linear operator; and the gradients and Hessians of
the data misfit by Taylor test. If that passes, the install is good.

## Optional: the cuFFTDx propagator

The Fresnel propagator has two backends. cuPy is the default and works out of
the box; [NVIDIA mathDX](https://docs.nvidia.com/cuda/mathdx/installation.html)
enables a faster cuFFTDx one, selected automatically when found.

```bash
export MATHDX_ROOT=/opt/nvidia/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12
export NVCC=/usr/local/cuda/bin/nvcc
```

The first use of a new grid size JIT-compiles a small shared library and
caches it. Under MPI only rank 0 compiles; the others wait at a barrier.
If mathDX is missing the package falls back to cuPy with no loss of
correctness.
