#!/usr/bin/env python
"""Demo step 0 — probe retrieval by near-field ptychography, on synthetic data.

    ./run.sh 01_nfp_probe.py              # 1 GPU
    ./run_mpi.sh 4 01_nfp_probe.py        # 4 GPUs

A Siemens star is scanned at `npos` random sub-pixel positions through the
measured ID16A probe; `RecNFP` then recovers the probe, the projection and the
positions from the intensities alone.  The probe it writes is what step 6 of a
real pipeline would be handed as `prb_file`.
"""
import argparse
import os
import sys

import numpy as np
import cupy as cp
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.paganin import multi_paganin
from holotomocupy.rec_nfp_mpi import RecNFP
from holotomocupy.logger_config import logger, set_log_level

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phantoms import id16a_probe, star_projection                 # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=512, help='detector size')
    p.add_argument('--npos', type=int, default=16, help='scan positions')
    p.add_argument('--niter', type=int, default=257, help='BH iterations')
    p.add_argument('--nchunk', type=int, default=8, help='positions per GPU pass')
    p.add_argument('--paganin', type=float, default=29.0,
                   help="delta/beta for the Paganin start (0 = start from zero)")
    p.add_argument('--out', default='./demo_out/nfp', help='output directory')
    return p.parse_args()


# Geometry: one distance, the ID16A cone-beam numbers at a coarse binning.
ENERGY, PIXEL, FOCUS_DET, Z1 = 17.1, 1.4760147601476e-6 * 4, 1.217, 5.110e-3


def build(a, rank):
    """Ground truth: probe, projection, positions, and the object grid."""
    rng = np.random.default_rng(10)
    pos_gt = (30 / 512 * a.n * (rng.random((a.npos, 2)) - 0.5)).astype('float32')
    pos_err = 4 * (rng.random((a.npos, 2)) - 0.5).astype('float32')
    # the object must contain every shifted projection
    reach = int(np.ceil(np.abs(pos_gt + pos_err).max())) + 8
    nobj = int(np.ceil((a.n + 2 * reach) / 32)) * 32
    if rank == 0:
        logger.info(f'positions +-{reach} px -> nobj {nobj} for n {a.n}')
    return id16a_probe(a.n)[0], star_projection(nobj), pos_gt, pos_err, nobj


def paganin_projection(cl, data, ref, pos, nobj, delta_beta):
    """Step 5's trick in 2-D: normalise by the flat, stitch, invert, once.

    The scan positions overlap, so the frames are mapped onto the object grid
    and averaged there.  Two details decide whether this is any use at all:

      * where no frame reached, the transmission is 1 -- no sample.  Dividing
        by a near-zero weight instead leaves a garbage rim, and the Fourier
        filter below spreads that rim over the whole frame (correlation with
        the truth 0.43 instead of 0.81).
      * the phase comes out of multi_paganin with the sign the object has, so
        it goes in as +ph, matching -delta + i*beta.
    """
    ones = cp.ones((1,) + data.shape[-2:], dtype='complex64')
    m = cp.ones((1, 2), dtype='float32')
    acc = cp.zeros((nobj, nobj), dtype='complex64')
    wt = cp.zeros((nobj, nobj), dtype='float32')
    for i in range(data.shape[0]):
        r = cp.asarray(pos[i:i + 1])
        rd = cp.asarray(data[i:i + 1]) / (ref + 1e-5)
        big = cl.cl_shift.curlySback(cp.log(rd.astype('complex64')), r, m)[0]
        one = cl.cl_shift.curlySback(ones, r, m)[0].real
        acc += cp.exp(big) * one
        wt += one
    # .real, not .astype: acc is complex and astype would drop the
    # imaginary part silently (ComplexWarning)
    rdata = cp.where(wt > 0.05, acc / cp.maximum(wt, 1e-3), 1.0).real
    ph = multi_paganin(rdata[None], cl.cl_prop.distance,
                       cl.cl_prop.wavelength, cl.cl_prop.voxelsize, delta_beta)
    return (ph + 1j * ph / delta_beta).astype('complex64')


def main():
    a = parse()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    set_log_level('INFO')
    prb_gt, proj_gt, pos_gt, pos_err, nobj = build(a, rank)

    cl = RecNFP(SimpleNamespace(
        energy=ENERGY, detector_pixelsize=PIXEL,
        focustodetectordistance=FOCUS_DET, z1=Z1,
        ntheta=a.npos,            # RecNFP calls the scan axis ntheta
        nz=a.n, n=a.n, nzobj=nobj, nobj=nobj,
        rho=[1, 2, 0.1], niter=a.niter, nchunk=a.nchunk,
        checkpoint_step=max(a.niter // 4, 1), error_step=max(a.niter // 8, 1),
        start_iter=0, psf_sigma=0.0, shift_type='fft',
        path_out=a.out, comm=comm))

    # forward: the truth in, the intensities out
    cl.vars['proj'][:] = cp.asarray(proj_gt)
    cl.vars['prb'][:] = cp.asarray(prb_gt)
    cl.vars['pos'][:] = cp.asarray(pos_gt[cl.st_theta:cl.end_theta])
    cl.gen_data(cl.vars, cl.data)

    # inverse: the Paganin start a real scan can build from its own flat
    # field, a flat probe, and a wrong position guess
    pos0 = pos_gt + pos_err
    if a.paganin > 0:
        u = cl.cl_prop.D(cp.asarray(prb_gt)[None], 0)
        ref = (cp.abs(u) ** 2)[0]          # the measured flat: probe, no sample
        cl.vars['proj'][:] = paganin_projection(cl, cl.data, ref, pos0, nobj,
                                                a.paganin)
    else:
        cl.vars['proj'][:] = 0
    cl.vars['prb'][:] = 1
    cl.vars['pos'][:] = cp.asarray(pos0[cl.st_theta:cl.end_theta])
    cl.BH()

    err = float(cp.abs(cl.vars['pos'] - cp.asarray(
        pos_gt[cl.st_theta:cl.end_theta])).mean())
    err = comm.allreduce(err, op=MPI.SUM) / comm.Get_size()
    if rank == 0:
        logger.info(f'position error {err:.4f} px (started at '
                    f'{np.abs(pos_err).mean():.4f})')
        logger.info(f'wrote {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
