#!/usr/bin/env python
"""
NFP — Synthetic Self-Test (step0-style: path_out + periodic checkpoints + tiffs).

End-to-end verification of `RecNFP` on fully synthetic data:
  1. Build a 2-D Siemens-star phantom (`proj`) and a Gaussian-smoothed probe (`prb`).
  2. Forward-simulate diffraction patterns via `gen_data`.
  3. Run iterative reconstruction (BH) — RecNFP saves tiffs every
     `checkpoint_step` iterations to `<path_out>/checkpoints_tiff/`.
  4. After BH, rank 0 gathers final proj / prb / pos errors and writes a
     final HDF5 to `<path_out>/result.h5` (same layout as step0.py).

Launch with:
    mpirun -n <N> python test.py
"""

import os
import sys
import subprocess
import numpy as np
import cupy as cp
import h5py
import scipy.ndimage as ndimage
from scipy.fft import fft2, ifft2, fftshift
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_nfp_mpi import RecNFP
from holotomocupy.utils import read_tiff, logger

import logging
logger.setLevel(logging.INFO)


# ── Acquisition parameters ───────────────────────────────────────────────────
n      = 1024          # detector / probe size (pixels)
ntheta = 16           # number of scan positions

energy                  = 17.1                       # keV
detector_pixelsize      = 1.4760147601476e-6 * 4     # m (binned)
focustodetectordistance = 1.217                      # m
z1                      = 5.110e-3                   # m  (sample–focus distance)

# ── Run config (step0-style) ──────────────────────────────────────────────────
path_out        = '/data2/vnikitin/tmp/test_nfp_results'
niter           = 513
nchunk          = 8
checkpoint_step = 32          # save tiff + (optional) h5 every N iters
error_step      = 32          # log error every N iters
rho             = [1, 2, 0.1] # gradient step-size scales for [proj, prb, pos]
photons         = None         # mean photons per pixel for Poisson noise; None to disable
# Detector PSF as one Gaussian on the detector intensity,
# sigma in detector pixels of this (binned) grid.  gen_data applies the SAME
# blur it reconstructs with, so a non-zero value here tests the blurred model
# against itself; to test robustness to a mis-specified PSF, blur the data at
# one sigma and reconstruct at another.  0 = no blur.
psf_sigma       = 0.0
# Shift interpolation: 'cubic' (B-spline, mirrored edges) or 'fft' (Fourier
# shift theorem, exact but PERIODIC -- nobj must clear n + 2*max|pos|, which
# the nobj below does).  'fft' is what the position refinement wants: the
# B-spline stencil's kernel error biases the position gradient.
shift_type      = 'cubic'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# ── Phantom object — Siemens star ────────────────────────────────────────────
delta_beta = 29   # δ/β ratio

def siemens_star(nobj, step_deg=15):
    """Return a (nobj, nobj) float32 Siemens-star mask."""
    def rotate_pts(pts, ang, center):
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (pts - center) @ R.T + center

    def tri_mask(X, Y, tri):
        (x1, y1), (x2, y2), (x3, y3) = tri
        d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if d == 0:
            return np.zeros_like(X, dtype=bool)
        a = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / d
        b = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / d
        return (a >= 0) & (b >= 0) & (1 - a - b >= 0)

    tri0 = np.array([
        (1.5 * nobj // 16, nobj // 2 - nobj // 32),
        (1.5 * nobj // 16, nobj // 2 + nobj // 32),
        (nobj // 2 - nobj // 128, nobj // 2),
    ], dtype=np.float32)
    yy, xx = np.mgrid[0:nobj, 0:nobj]
    center = np.array([nobj / 2, nobj / 2], dtype=np.float32)
    star   = np.zeros((nobj, nobj), dtype=np.float32)
    for deg in range(0, 360, step_deg):
        star += tri_mask(xx, yy, rotate_pts(tri0, np.deg2rad(deg), center)).astype(np.float32)
    star /= star.max() or 1.0
    return star


def gen_proj(nobj, delta_beta):
    star = cp.array(siemens_star(nobj))
    # smooth with Gaussian in Fourier space
    v  = cp.arange(-nobj // 2, nobj // 2, dtype='float32') / nobj
    vx, vy = cp.meshgrid(v, v)
    g  = cp.fft.fftshift(cp.exp(-8 * (vx ** 2 + vy ** 2)))
    star = cp.fft.ifft2(cp.fft.fft2(star) * g).real
    # scale delta to phase range [0, π/4]; beta = delta/delta_beta
    star = star / star.max() * (np.pi / 4)/2
    # complex: real = delta projection, imag = beta projection
    return (-star + 1j * star / delta_beta).astype('complex64')




# ── Probe — loaded from ID16A tiff files (cached locally) ────────────────────
_prb_dir = '../../demo/data/prb_id16a'
_urls = [
    'https://g-110014.fd635.8443.data.globus.org/holotomocupy/examples_synthetic/data/prb_id16a/prb_abs_2048.tiff',
    'https://g-110014.fd635.8443.data.globus.org/holotomocupy/examples_synthetic/data/prb_id16a/prb_phase_2048.tiff',
]
if rank == 0:
    os.makedirs(_prb_dir, exist_ok=True)
    for url in _urls:
        dest = os.path.join(_prb_dir, os.path.basename(url))
        if not os.path.exists(dest):
            subprocess.run(['wget', '-q', '-O', dest, url], check=True)
comm.Barrier()

prb_abs   = read_tiff(os.path.join(_prb_dir, 'prb_abs_2048.tiff'))[:1]
prb_phase = read_tiff(os.path.join(_prb_dir, 'prb_phase_2048.tiff'))[:1]
prb = (prb_abs * np.exp(1j * prb_phase)).astype('complex64')

# crop to n×n centred patch
prb = prb[:, prb.shape[1] // 2 - n // 2 : prb.shape[1] // 2 + n // 2,
             prb.shape[2] // 2 - n // 2 : prb.shape[2] // 2 + n // 2]

# mild Gaussian filter in Fourier space
v = np.arange(-n // 2, n // 2, dtype='float32') / n
vx, vy = np.meshgrid(v, v, indexing='ij')
filt = fftshift(np.exp(-4.0 * (vx ** 2 + vy ** 2)).astype('float32'))
prb  = ifft2(fft2(prb) * filt).astype('complex64')

# normalise amplitude to 1 (single distance, squeeze to 2D)
prb = prb[0]
prb /= np.mean(np.abs(prb))
prb_gt = cp.array(prb)


# ── Positions — random sub-pixel shifts ──────────────────────────────────────
rng     = np.random.default_rng(10)
pos_gt  = (30/512*n * (rng.random((ntheta, 2)) - 0.5)).astype('float32')   # true positions
pos_err = 4*(      rng.random((ntheta, 2)) - 0.5 ).astype('float32')   # initial guess error

# Object grid sized to fully contain shifted projections (step0 formula)
pos_range = int(np.ceil(np.abs(pos_gt + pos_err).max())) + 8
nobj      = int(np.ceil((n + 2 * pos_range) / 32)) * 32
if rank == 0:
    logger.info(f'pos_range = ±{pos_range} pix → nobj = {nobj} (n = {n})')

proj_gt = gen_proj(nobj, delta_beta)

# ── Initialise RecNFP ─────────────────────────────────────────────────────────
rec_args = SimpleNamespace(
    energy                  = energy,
    detector_pixelsize      = detector_pixelsize,
    focustodetectordistance = focustodetectordistance,
    z1                      = z1,
    ntheta                  = ntheta,
    nz                      = n,
    n                       = n,
    nzobj                   = nobj,
    nobj                    = nobj,
    rho                     = rho,
    niter                   = niter,
    nchunk                  = nchunk,
    checkpoint_step         = checkpoint_step,
    error_step              = error_step,
    start_iter              = 0,
    psf_sigma               = psf_sigma,
    shift_type              = shift_type,
    path_out                = path_out,
    comm                    = comm,
)

cl = RecNFP(rec_args)

if rank == 0:
    logger.info(f'nobj={nobj}, n={n}, ntheta={ntheta}, niter={niter}')
    logger.info(f'path_out = {path_out}')


# ── Generate synthetic data ──────────────────────────────────────────────────
cl.vars['proj'][:] = proj_gt
cl.vars['prb'][:]  = prb_gt
cl.vars['pos'][:]  = cp.array(pos_gt[cl.st_theta:cl.end_theta])

cl.gen_data(cl.vars, cl.data)

# ── Add Poisson noise on the intensity ──────────────────────────────────────
# Per-theta RNG keyed by GLOBAL theta index so each theta's noise realisation
# is identical regardless of how many MPI ranks split the job (reproducibility).
# cl.data IS the intensity now (F0 is intensity-based), so the Poisson draw is
# direct -- no square/sqrt round trip.
if photons is not None:
    seeds = np.random.SeedSequence(20251119).spawn(ntheta)
    for j_local in range(cl.end_theta - cl.st_theta):
        rng = np.random.default_rng(seeds[cl.st_theta + j_local])
        I   = cl.data[j_local].astype('float32')
        cl.data[j_local] = rng.poisson(I * photons).astype('float32') / photons
    if rank == 0:
        logger.info(f'Poisson noise: {photons} photons/pixel  '
                    f'(intensity std ≈ 1/sqrt(photons) = {1.0 / np.sqrt(photons):.4f})')


# ── Reconstruction: reset to initial guess, then BH (periodic tiff saves) ────
cl.vars['proj'][:] = 0
cl.vars['prb'][:]  = 1
cl.vars['pos'][:]  = cp.array((pos_gt + pos_err)[cl.st_theta:cl.end_theta])

cl.BH()


# ── Collect & write final HDF5 (step0 pattern) ───────────────────────────────
pos_final_local = cl.vars['pos'].get()
pos_init_local  = cl.pos_init.get()
pos_drift_local = pos_final_local - pos_init_local        # change from init guess

all_pos_final = comm.gather(pos_final_local, root=0)
all_pos_drift = comm.gather(pos_drift_local, root=0)
if rank == 0:
    pos_final  = np.concatenate(all_pos_final, axis=0)    # (ntheta, 2)
    pos_drift  = np.concatenate(all_pos_drift, axis=0)    # (ntheta, 2)
    pos_recov_err = pos_final - pos_gt                    # vs ground truth

    def _stats(label, e):
        logger.info(f'{label} y (pix): max={np.abs(e[:,0]).max():.4f}  '
                    f'mean={np.abs(e[:,0]).mean():.4f}  std={e[:,0].std():.4f}')
        logger.info(f'{label} x (pix): max={np.abs(e[:,1]).max():.4f}  '
                    f'mean={np.abs(e[:,1]).mean():.4f}  std={e[:,1].std():.4f}')

    _stats('init guess error  (pos_gt+pos_err - pos_gt)',  pos_err)
    _stats('recovered  vs GT  (pos_final     - pos_gt)',   pos_recov_err)
    _stats('drift from init   (pos_final - pos_init)',     pos_drift)

    prb_np  = cl.vars['prb'].get()
    proj_np = cl.vars['proj'].get()

    h5_out = os.path.join(path_out, 'result.h5')
    os.makedirs(path_out, exist_ok=True)
    with h5py.File(h5_out, 'w') as f:
        f.create_dataset('prb_amp',       data=np.abs(prb_np)[None])
        f.create_dataset('prb_phase',     data=np.angle(prb_np)[None])
        f.create_dataset('proj_delta',    data=proj_np.real[None])
        f.create_dataset('proj_beta',     data=proj_np.imag[None])
        f.create_dataset('pos_final',     data=pos_final)
        f.create_dataset('pos_gt',        data=pos_gt)
        f.create_dataset('pos_recov_err', data=pos_recov_err)    # vs ground truth
        f.create_dataset('pos_drift',     data=pos_drift)        # vs initial guess
        # ground truth fields for offline comparison
        f.create_dataset('proj_delta_gt', data=cp.asnumpy(proj_gt.real)[None])
        f.create_dataset('proj_beta_gt',  data=cp.asnumpy(proj_gt.imag)[None])
    logger.info(f'Saved final result to {h5_out}')
    logger.info(f'Periodic tiffs in       {path_out}/checkpoints_tiff/')
