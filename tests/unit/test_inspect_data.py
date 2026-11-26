#!/usr/bin/env python
"""RecNFP.inspect_data flags each injected defect and nothing else."""
import os
import sys

import numpy as np
import cupy as cp
import scipy.ndimage as ndimage
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, run                                    # noqa: E402

from mpi4py import MPI                                            # noqa: E402
from holotomocupy.rec_nfp_mpi import RecNFP                       # noqa: E402

N, NPOS = 64, 16
NOBJ = N + 64
HOT, DEAD = (20, 30), (40, 12)
rng = np.random.default_rng(7)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _smooth_cx(shape, sigma):
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    z = ndimage.gaussian_filter(z.real, sigma) + 1j * ndimage.gaussian_filter(z.imag, sigma)
    return (z / np.abs(z).max()).astype('complex64')


cl = RecNFP(SimpleNamespace(
    energy=17.1, detector_pixelsize=1.4760147601476e-6 * 4,
    focustodetectordistance=1.217, z1=5.110e-3,
    ntheta=NPOS, nz=N, n=N, nzobj=NOBJ, nobj=NOBJ,
    rho=[1, 2, 0.1], niter=1, nchunk=4, checkpoint_step=-1, error_step=-1,
    start_iter=0, psf_sigma=0.0, shift_type='cubic',
    path_out=os.environ.get('HTC_SCRATCH', '/tmp') + '/htc_unit_inspect',
    comm=comm))
prb = 1.0 + 0.3 * _smooth_cx((N, N), 6)
proj = 0.2 * _smooth_cx((NOBJ, NOBJ), 4)
pos = (2 * (rng.random((NPOS, 2)) - 0.5)).astype('float32')
cl.vars['prb'][:] = cp.asarray(prb)
cl.vars['proj'][:] = cp.asarray(proj)
cl.vars['pos'][:] = cp.asarray(pos[cl.st_theta:cl.end_theta])
cl.gen_data(cl.vars, cl.data)
CLEAN = cl.data.copy()


def _frame(g):
    return cl.data[g - cl.st_theta] if cl.st_theta <= g < cl.end_theta else None


def _inspect():
    return comm.bcast(cl.inspect_data(), root=0)


def _reset():
    cl.data[:] = CLEAN


def _ids(a):
    return [int(v) for v in np.asarray(a).ravel()]


def _px(a):
    return [tuple(int(v) for v in p) for p in np.asarray(a).reshape(-1, 2)]


def test_clean_data_flags_nothing():
    _reset()
    r = _inspect()
    for k in ('nonfinite_frames', 'nonpos_frames', 'outlier_frames'):
        check(f'clean: no {k}', len(r[k]) == 0, f'{_ids(r[k])}')
    for k in ('hot_pixels', 'dead_pixels'):
        check(f'clean: no {k}', len(r[k]) == 0, f'{_px(r[k])}')
    check('clean: normalized', not r['not_normalized'])
    check('clean: data left untouched', np.array_equal(cl.data, CLEAN))


def test_nan_pixel():
    _reset()
    f = _frame(3)
    if f is not None:
        f[10, 10] = np.nan
    r = _inspect()
    check('NaN: nonfinite_frames == [3]', _ids(r['nonfinite_frames']) == [3],
          f"{_ids(r['nonfinite_frames'])}")
    check('NaN: not read as a dead pixel', len(r['dead_pixels']) == 0,
          f"{_px(r['dead_pixels'])}")
    check('NaN: no outlier frame', len(r['outlier_frames']) == 0,
          f"{_ids(r['outlier_frames'])}")


def test_bright_frame():
    _reset()
    f = _frame(7)
    if f is not None:
        f *= 2
    r = _inspect()
    check('x2 frame: outlier_frames == [7]', _ids(r['outlier_frames']) == [7],
          f"{_ids(r['outlier_frames'])}")
    check('x2 frame: no nonfinite', len(r['nonfinite_frames']) == 0)


def test_hot_pixel():
    _reset()
    cl.data[:, HOT[0], HOT[1]] *= 5
    r = _inspect()
    check(f'hot: hot_pixels == [{HOT}]', _px(r['hot_pixels']) == [HOT],
          f"{_px(r['hot_pixels'])}")
    check('hot: no dead pixel', len(r['dead_pixels']) == 0, f"{_px(r['dead_pixels'])}")
    check('hot: no outlier frame', len(r['outlier_frames']) == 0,
          f"{_ids(r['outlier_frames'])}")


def test_dead_pixel():
    _reset()
    cl.data[:, DEAD[0], DEAD[1]] = 0
    r = _inspect()
    check(f'dead: dead_pixels == [{DEAD}]', _px(r['dead_pixels']) == [DEAD],
          f"{_px(r['dead_pixels'])}")
    check('dead: no hot pixel', len(r['hot_pixels']) == 0, f"{_px(r['hot_pixels'])}")
    check('dead: every frame has a non-positive pixel',
          _ids(r['nonpos_frames']) == list(range(NPOS)), f"{_ids(r['nonpos_frames'])}")



def test_raw_counts():
    _reset()
    cl.data[:] *= 1000
    r = _inspect()
    check('x1000: not_normalized', r['not_normalized'])
    check('x1000: no outlier frame', len(r['outlier_frames']) == 0,
          f"{_ids(r['outlier_frames'])}")
    check('x1000: no hot/dead pixel', len(r['hot_pixels']) + len(r['dead_pixels']) == 0)

if __name__ == '__main__':
    sys.exit(run(globals()))
