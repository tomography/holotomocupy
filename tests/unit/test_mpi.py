#!/usr/bin/env python
"""The MPI layer: the two redistributions, the reduction helpers, parallel HDF5.

    mpirun -np 2 python tests/unit/test_mpi.py

Runs on any rank count, including 1.  The redistribution is the piece that is
easy to get silently wrong -- it moves an array from an obj-slab decomposition
to a theta-slice one and back -- so it is checked against the global array
every rank can reconstruct for itself.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, run                                    # noqa: E402

from mpi4py import MPI                                            # noqa: E402

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
NZOBJ, NTHETA, NOBJ = 16, 12, 8


def _truth(dtype):
    """The same global array on every rank: value = z*1000 + t*10 + x."""
    z, t, x = np.meshgrid(np.arange(NZOBJ), np.arange(NTHETA), np.arange(NOBJ),
                          indexing='ij')
    return (z * 1000 + t * 10 + x).astype(dtype)


def test_redist_roundtrip():
    """theta-distributed -> z-distributed -> back is the identity, exactly."""
    from holotomocupy.mpi_functions import MPIClass
    for dtype in ('float32', 'complex64'):
        cl = MPIClass(COMM, NZOBJ, NTHETA, NOBJ, dtype)
        g = _truth(dtype)
        # forward layout: this rank's angles, all z
        src = np.ascontiguousarray(g[:, cl.st_theta:cl.end_theta].transpose(1, 0, 2))
        dst = np.empty((NTHETA, cl.end_obj - cl.st_obj, NOBJ), dtype=dtype)
        cl.redist(src, dst, direction='backward')
        want = g[cl.st_obj:cl.end_obj].transpose(1, 0, 2)
        check(f'redist backward lands the right values [{dtype}]',
              np.array_equal(dst, want),
              f'rank {RANK}: z[{cl.st_obj}:{cl.end_obj}] of {NZOBJ}')

        back = np.empty_like(src)
        cl.redist(dst, back, direction='forward')
        check(f'redist round-trip is the identity [{dtype}]',
              np.array_equal(back, src))
        cl.close()


def test_allreduce_helpers():
    from holotomocupy.mpi_functions import MPIClass
    cl = MPIClass(COMM, NZOBJ, NTHETA, NOBJ, 'float32')
    a = np.full(4, RANK + 1, dtype='float32')
    cl.allreduce(a)
    check('allreduce sums across ranks',
          np.allclose(a, SIZE * (SIZE + 1) / 2), f'{a[0]} over {SIZE} rank(s)')
    s = cl.allreduce_scalars(float(RANK), 2.0 * RANK)
    check('allreduce_scalars sums each entry',
          np.allclose(s, [SIZE * (SIZE - 1) / 2, SIZE * (SIZE - 1)]), f'{s}')
    cl.close()


def test_parallel_hdf5_write_read():
    """Every rank writes its own slab through driver='mpio' and reads it back."""
    import h5py
    if not h5py.get_config().mpi:
        check('h5py MPI write/read', False, 'h5py is serial; skipping is not passing')
        return
    # one path, agreed by rank 0; HTC_SCRATCH keeps it off a full /tmp
    d = COMM.bcast(os.environ.get('HTC_SCRATCH', tempfile.gettempdir()), root=0)
    path = os.path.join(d, f'htc_unit_mpi_{COMM.bcast(os.getpid(), root=0)}.h5')
    with h5py.File(path, 'w', driver='mpio', comm=COMM) as f:
        ds = f.create_dataset('x', shape=(SIZE, 4), dtype='float32')
        ds[RANK] = RANK
    COMM.Barrier()
    with h5py.File(path, 'r', driver='mpio', comm=COMM) as f:
        got = f['x'][:]
    check('parallel HDF5 write then read',
          np.array_equal(got, np.repeat(np.arange(SIZE, dtype='float32')[:, None], 4, 1)))
    COMM.Barrier()
    if RANK == 0:
        os.remove(path)


def test_gpu_per_rank():
    """Each rank must own a device; two ranks sharing one is legal but slow.

    A binder sets CUDA_VISIBLE_DEVICES per rank, after which every rank sees
    its own GPU as device 0 -- so the physical id has to come from the
    environment, not from getDevice().
    """
    import cupy as cp
    vis = os.environ.get('CUDA_VISIBLE_DEVICES')
    phys = vis if vis else str(cp.cuda.runtime.getDevice())
    devs = COMM.allgather(phys)
    free, _ = cp.cuda.runtime.memGetInfo()
    check('every rank has a usable CUDA device', free > 0,
          f'{SIZE} rank(s) on GPU(s) {devs}')
    if SIZE > 1 and len(set(devs)) < SIZE:
        print(f'  note: ranks share GPUs ({devs}); launch through demo/bind.sh '
              f'to spread them')


if __name__ == '__main__':
    code = run(globals())
    sys.exit(COMM.allreduce(code, op=MPI.MAX))
