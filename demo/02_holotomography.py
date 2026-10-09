#!/usr/bin/env python
"""Demo step 6 — holotomographic reconstruction of a synthetic 3-D phantom.

    ./run.sh 02_holotomography.py              # 1 GPU
    ./run_mpi.sh 4 02_holotomography.py        # 4 GPUs

Forward-simulates `ndist` propagation distances through a 3-D phantom with the
measured ID16A probe, then recovers object, probe and positions jointly with
the Bilinear-Hessian solver.  The positions start deliberately wrong by ~0.5 px
so that the refinement has something to do.
"""
import argparse
import os
import sys

import numpy as np
import cupy as cp
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.paganin import multi_paganin
from holotomocupy.rec_mpi import Rec
from holotomocupy.writer import Writer
from holotomocupy.logger_config import logger, set_log_level

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phantoms import id16a_probe, phantom3d                       # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=128, help='detector size')
    p.add_argument('--ntheta', type=int, default=180, help='projection angles')
    p.add_argument('--ndist', type=int, default=4, help='propagation distances')
    p.add_argument('--niter', type=int, default=100, help='BH iterations')
    p.add_argument('--paganin', type=float, default=100.0,
                   help='delta/beta for the Paganin start (0 = start from zero)')
    p.add_argument('--nchunk', type=int, default=16, help='angles per GPU pass')
    p.add_argument('--out', default='./demo_out/rec', help='output directory')
    return p.parse_args()


SHIFT = 'fft'          # 'fft' (the package default) or 'cubic'
ENERGY, FOCUS_DET = 17.1, 1.217
PIXEL = 1.4760147601476e-6 * 2 * 8
Z1 = np.array([5.110, 5.464, 6.879, 9.817]) * 1e-3


def paganin_volume(cl, data, nobj, n, delta_beta):
    """Step 5's initial guess: normalise, resample, Paganin, then FBP.

    Three things step 5 does and all three are needed:
      * divide by the reference -- Paganin wants the FLAT-FIELD NORMALISED
        intensity; feeding it the raw frames leaves the probe's own structure
        in log(data) and the result is nonsense;
      * resample each plane onto the object grid, since the four are
        differently demagnified;
      * then one multi-distance inversion and one FBP.

    MPI: the angles are split across ranks but FBP needs all of them, so each
    rank inverts its own angles and then redistributes to (all angles, this
    rank's z-slab).  Returns that slab, not the whole volume.
    """
    mags = cp.asarray(1.0 / cl.norm_magnifications, dtype='float32')
    ref = cp.asarray(cl.ref) ** 2                  # cl.ref is an AMPLITUDE
    psi = np.empty((cl.local_ntheta, nobj, nobj), dtype='complex64')
    for i in range(cl.local_ntheta):
        big = cp.empty((cl.ndist, nobj, nobj), dtype='float32')
        for k in range(cl.ndist):
            r = cp.asarray(cl.vars['pos'][k][i:i + 1])
            m = cp.broadcast_to(mags[k], (1, 2)).copy()
            rdata = cp.asarray(data[k, i:i + 1]) / (ref[k] + 1e-5)
            big[k] = cp.exp(cl.cl_shift.curlySback(
                cp.log(rdata.astype('complex64')), r, m)[0].real)
        ph = multi_paganin(big, cl.cl_prop.distance, cl.cl_prop.wavelength,
                           cl.cl_prop.voxelsize, delta_beta)
        psi[i] = cp.asnumpy(ph + 1j * ph / delta_beta)
    allpsi = np.empty((cl.ntheta, cl.local_nzobj, nobj), dtype='complex64')
    cl.redist(psi, allpsi, 'backward')
    # cl_tomo's buffers are sized for nchunk z-slices, so FBP goes in slabs
    gpsi = cp.asarray(allpsi)
    out = cp.empty((cl.local_nzobj, nobj, nobj), dtype='complex64')
    for z in range(0, cl.local_nzobj, cl.nchunk):
        z1 = min(z + cl.nchunk, cl.local_nzobj)
        out[z:z1] = cl.cl_tomo.fbp(cp.ascontiguousarray(gpsi[:, z:z1]), 'ramp')
    return out


def main():
    a = parse()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    set_log_level('INFO')
    nobj = 3 * a.n // 2                     # room for the +-15 px displacement

    obj = phantom3d(nobj)
    prb = id16a_probe(a.n, a.ndist)
    rng = np.random.default_rng(10)
    pos = (30 * (rng.random((a.ntheta, a.ndist, 2)) - 0.5)).astype('float32')
    pos_err = (rng.random((a.ntheta, a.ndist, 2)) - 0.5).astype('float32')
    theta = np.linspace(0, np.pi, a.ntheta, dtype='float32')

    cl = Rec(SimpleNamespace(
        energy=ENERGY, detector_pixelsize=PIXEL,
        focustodetectordistance=FOCUS_DET, z1=Z1[:a.ndist], theta=theta,
        ndist=a.ndist, ntheta=a.ntheta, nz=a.n, n=a.n, nzobj=nobj, nobj=nobj,
        mask=0.9, lam_prbfit=2e-3, lam_laplacian=0, rho=[1, 0.05, 0.02, 0],
        shift_type=SHIFT,
        niter=a.niter, nchunk=a.nchunk,
        checkpoint_step=max(a.niter // 4, 1), error_step=max(a.niter // 8, 1),
        start_iter=0, comm=comm))

    writer = Writer(path_out=a.out, comm=comm, st_obj=cl.st_obj, end_obj=cl.end_obj,
                    nzobj=nobj, nobj=nobj, st_theta=cl.st_theta,
                    end_theta=cl.end_theta, ntheta=a.ntheta, ndist=a.ndist,
                    nz=a.n, n=a.n)

    # forward: each rank owns a slab of obj and a slice of the angles
    cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
    cl.vars['prb'][:] = prb
    cl.vars['pos'][:] = pos[cl.st_theta:cl.end_theta].transpose(1, 0, 2)
    cl.gen_data(cl.vars, cl.data)
    cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)

    # inverse: the Paganin+FBP start a real run gets from step 5, a flat
    # probe, and positions off by pos_err
    if a.paganin > 0:
        # vars['obj'] is pinned host memory, so the volume comes back explicitly
        # paganin_volume already returns this rank's z-slab
        cl.vars['obj'][:] = cp.asnumpy(
            paganin_volume(cl, cl.data, nobj, a.n, a.paganin))
    else:
        cl.vars['obj'][:] = 0
    cl.vars['prb'][:] = 1
    cl.vars['pos'][:] = (pos + pos_err)[cl.st_theta:cl.end_theta].transpose(1, 0, 2)
    cl.BH(writer=writer)

    # the honest number: distance to the truth, not to the starting guess
    got = cp.asnumpy(cl.vars['pos']) if isinstance(cl.vars['pos'], cp.ndarray) \
        else np.asarray(cl.vars['pos'])
    e1 = np.abs(got - pos[cl.st_theta:cl.end_theta].transpose(1, 0, 2)).mean()
    e0 = np.abs(pos_err[cl.st_theta:cl.end_theta]).mean()
    e1 = comm.allreduce(float(e1), MPI.SUM) / comm.Get_size()
    e0 = comm.allreduce(float(e0), MPI.SUM) / comm.Get_size()
    if rank == 0:
        logger.info(f'position error {e1:.4f} px (started at {e0:.4f})')

    if rank == 0:
        logger.info(f'wrote {os.path.abspath(a.out)}  '
                    f'(checkpoints/ and checkpoints_tiff/)')


if __name__ == '__main__':
    main()
