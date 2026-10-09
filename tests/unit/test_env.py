#!/usr/bin/env python
"""Is the installation usable at all: packages, GPU, MPI, parallel HDF5, kernels.

    python tests/unit/test_env.py

Everything here must pass before any other test means anything.  A missing
piece is reported, not raised, so one run tells you everything that is wrong.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, run                                    # noqa: E402


def test_packages():
    for mod in ('numpy', 'scipy', 'cupy', 'h5py', 'mpi4py', 'tifffile', 'matplotlib'):
        try:
            m = __import__(mod)
            check(f'import {mod}', True, getattr(m, '__version__', ''))
        except Exception as e:
            check(f'import {mod}', False, str(e))


def test_holotomocupy_imports():
    for mod in ('config', 'chunking', 'logger_config', 'utils', 'tomo', 'shift',
                'shift_fft', 'propagation', 'psf', 'extra_terms', 'reader',
                'writer', 'mpi_functions', 'rec_mpi', 'rec_nfp_mpi',
                'autofocus', 'esrf_meta'):
        try:
            __import__(f'holotomocupy.{mod}')
            check(f'import holotomocupy.{mod}', True)
        except Exception as e:
            check(f'import holotomocupy.{mod}', False, str(e))


def test_gpu():
    import cupy as cp
    ndev = cp.cuda.runtime.getDeviceCount()
    check('a CUDA device is visible', ndev > 0, f'{ndev} device(s)')
    p = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    check('device memory reported', total > 0,
          f"{p['name'].decode()}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
    a = cp.arange(1024, dtype='float32')
    check('a kernel runs and returns', float(a.sum()) == 1024 * 1023 / 2)


def test_cuda_kernels_compile():
    """The raw kernels are JIT-compiled at import; building one proves nvrtc works."""
    import cupy as cp
    from holotomocupy.tomo import Tomo
    cl = Tomo(32, 2, __import__('numpy').linspace(0, 3.14, 8, dtype='float32'), -1.0)
    d = cl.R(cp.ones((2, 32, 32), dtype='complex64'))
    check('Tomo.R compiles and runs', bool(cp.isfinite(d).all()), f'{tuple(d.shape)}')


def test_fft_backend():
    from holotomocupy.propagation import cufftdx_available
    ok = cufftdx_available()
    # not a failure: the cuPy backend is correct, cuFFTDx is only faster
    check('FFT backend resolved', True,
          'cuFFTDx' if ok else 'cuPy (set MATHDX_ROOT for cuFFTDx)')


def test_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    n = comm.Get_size()
    tot = comm.allreduce(comm.Get_rank() + 1, op=MPI.SUM)
    check('MPI allreduce', tot == n * (n + 1) // 2,
          f'{n} rank(s), sum of ranks+1 = {tot}')


def test_h5py_parallel():
    import h5py
    mpi = h5py.get_config().mpi
    check('h5py built with MPI', mpi,
          'driver="mpio" available' if mpi else
          'SERIAL h5py: step6 and steps15 cannot run multi-rank')


if __name__ == '__main__':
    sys.exit(run(globals()))
