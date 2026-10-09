#!/usr/bin/env python
"""
NFP reconstruction of the WT_H cell01 NFP2D scan, from its ESRF NXtomo (.nx).

Launch with:
    mpirun -n <N> python step0.py config_step0.conf

This folder is step 0 and nothing else.  WT_H_cell01_NFP2D_025nm_0001 is a
standalone 2-D near-field ptychography scan -- 50 frames on a 50-point spiral
at a single rotation angle -- so it has no tomogram to feed and no step 6.
What comes out is the probe and the single 2-D projection of the cell.

WHY THIS IS A COPY OF ../AtomiumS1_FT_RD300/step0.py.  Both read an NXtomo whose
`instrument/detector/data` is an HDF5 VIRTUAL dataset pointing at RAW_DATA
through four `..` written against the full beamline tree.  Our copy of that
tree is flatter -- PROCESSED_DATA holds projections/ directly and RAW_DATA has
no <sample> level -- so the recorded paths resolve to nothing and HDF5 returns
the FILL VALUE: zeros, silently.  nx_frames.NxFrames takes the mapping apart
and rebases each source onto the tree we actually have; the `fr.missing` check
below is what stops a run that would otherwise succeed and hand back a probe
retrieved from zeros.  Diff against ../AtomiumS1_FT_RD300/step0.py to see the whole
difference: geometry logging, the model= knob, and one scan instead of a tomo
scan's companion.

The three sources are scans 2 (20 darks), 3 (the 50 NFP frames) and 4 (20
flats) of RAW_DATA/WT_H_cell01_NFP2D_025nm_0001/.  To check they resolve
without starting a job,

    python nx_frames.py <path to the .nx>

prints the source file behind each block of frames, marks any that are missing,
says where each should be copied, and exits non-zero.  As of 2026-10-05 all
three resolve.

THE FLATS ARE READ AND NOT USED by the reconstruction.  In NFP the probe IS the
unknown, so there is nothing to flat-field by; only the darks enter.  They were
used offline, though, to show that this scan's 1.9% intensity drift continues
past the last data frame and into the flats -- i.e. it is instrumental, not the
sample, which is why the data is normalised frame by frame below.  See
README.md.
"""

import sys
import os
import numpy as np
import cupy as cp
import h5py
from types import SimpleNamespace
from mpi4py import MPI
from holotomocupy.rec_nfp_mpi import RecNFP
from holotomocupy.config import parse_args_step0_nx
from holotomocupy.reader import read_nxtomo_meta
from holotomocupy.utils import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nx_frames import NxFrames

args = parse_args_step0_nx(sys.argv[1])
logger.setLevel(args.log_level)

nx_file  = args.nx_file
h5_out   = args.h5_out
n        = args.n
niter    = args.niter
nchunk   = args.nchunk
checkpoint_step = args.checkpoint_step
error_step = args.error_step
rho      = args.rho

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ---------------------------------------------------------------------------
# Geometry (read on all ranks — file is small)
# ---------------------------------------------------------------------------
meta = read_nxtomo_meta(nx_file)

energy     = args.energy if args.energy is not None else meta['energy']
z1         = meta['z1']
z_total    = meta['z_total']
pixel_size = meta['pixel_size']
voxelsize  = meta['voxelsize']
magnification = meta['magnification']

data_ids = meta['data_ids']
dark_ids = meta['dark_ids']
flat_ids = meta['flat_ids']
ntheta   = len(data_ids)

ny, nx_det = meta['ny'], meta['nx']
sty = (ny - n) // 2
stx = (nx_det - n) // 2

# Positions in object-plane pixels: spy = x_trans (y motor), spz = y_trans (z
# motor).  The .json beside the .nx records the mapping nxtomomill used:
# x_translation = sy[mm] + spyp[um], y_translation = sz[mm] + spzp[um].
spy = meta['x_trans'] * 1e-3   # mm → m
spz = meta['y_trans'] * 1e-3   # mm → m
spy -= spy.mean()
spz -= spz.mean()
pos = np.stack([-spz / voxelsize, spy / voxelsize], axis=-1).astype('float32')

if rank == 0:
    logger.info(f'nx_file                 = {nx_file}')
    logger.info(f'entry                   = {meta["entry"]}')
    _esrc = 'config override' if args.energy is not None else f'{nx_file} ({meta["energy"]} keV)'
    logger.info(f'energy                  = {energy} keV   [{_esrc}]')
    logger.info(f'z1                      = {z1*1e3:.4f} mm')
    logger.info(f'focustodetectordistance = {z_total*1e3:.2f} mm')
    logger.info(f'magnification           = {magnification:.2f}')
    logger.info(f'voxelsize               = {voxelsize*1e9:.2f} nm')
    logger.info(f'detector pixel          = {pixel_size*1e6:.4f} um')
    logger.info(f'ntheta                  = {ntheta}  (dark={len(dark_ids)} flat={len(flat_ids)})')
    logger.info(f'misfit model            = {args.model}   psf_sigma = {args.psf_sigma} det.px')
    logger.info(f'positions y (pix): [{pos[:,0].min():.2f}, {pos[:,0].max():.2f}]')
    logger.info(f'positions x (pix): [{pos[:,1].min():.2f}, {pos[:,1].max():.2f}]')

pos_range = int(np.ceil(np.abs(pos).max())) + 8
nobj      = int(np.ceil((n + 2 * pos_range) / 32)) * 32
if rank == 0:
    logger.info(f'n={n}  nobj={nobj}  pos_range=±{pos_range} pix')

# ---------------------------------------------------------------------------
# Init RecNFP
# ---------------------------------------------------------------------------
_path_out = args.path_out if args.path_out else None
rec_args = SimpleNamespace(
    energy                  = energy,
    detector_pixelsize      = pixel_size,
    focustodetectordistance = z_total,
    z1                      = z1,
    ntheta                  = ntheta,
    nz                      = n,
    n                       = n,
    nzobj                   = nobj,
    nobj                    = nobj,
    rho                     = rho,
    niter                   = niter,
    nchunk                  = nchunk,
    checkpoint_step = checkpoint_step,
    error_step = error_step,
    start_iter              = 0,
    psf_sigma               = args.psf_sigma,
    # Forwarded so `model=` in the conf actually reaches RecNFP; the older
    # step0.py copies parse it and drop it, leaving RecNFP on its 'intensity'
    # default whatever the conf says.
    model                   = args.model,
    estimate_rho            = args.estimate_rho,
    rho_estimate_niter      = args.rho_estimate_niter,
    rho_trial_error_step    = args.rho_trial_error_step,
    path_out                = _path_out,
    shift_type              = args.shift_type,
    comm                    = comm,
)

cl = RecNFP(rec_args)

# ---------------------------------------------------------------------------
# Load data: dark-subtract, then normalise EACH FRAME to mean 1.
#
# cl.data is the INTENSITY, not its square root: RecNFP.F0 compares K|psi|^2
# with d under either misfit model.  Per-frame normalisation puts every frame
# at ~1, which is the scale rho and the PSF were tuned at, and removes the
# 1.9% flux drift across this scan that one global mean leaves in.
# ---------------------------------------------------------------------------
fr = NxFrames(nx_file)
if fr.missing:
    # Name the destination, not the recorded path.  fr.missing holds what the
    # NXtomo says, which is where ESRF had the file -- copying there is wrong
    # and the mismatch is confusing.  suggest_destination walks the same
    # ancestors _rebase searches and picks the one whose RAW_DATA/ already
    # exists.
    raise SystemExit(
        f'{len(fr.missing)} of the {len(fr.sources)} virtual sources of '
        f'{nx_file}\ndo not exist; reading through them would silently return '
        'zeros.  Copy the raw\nbalor detector files to:\n'
        + ''.join(f'    {fr.suggest_destination(m)}\n' for m in fr.missing)
        + 'These are scans 2 (darks), 3 (the 50 NFP frames) and 4 (flats) of '
        'RAW_DATA/WT_H_cell01_NFP2D_025nm_0001/.\n'
        'nx_frames.rebase_candidates() lists every path _rebase accepts.')

# Dark field (read once on each rank -- small).  Subtracted BEFORE the
# normalisation below: the dark is an additive offset (98.9 of ~8070 ADU) and
# the drift is a multiplicative gain, so the order is not interchangeable.
if len(dark_ids) > 0:
    dark_stack = fr.frames(dark_ids)[:, sty:sty+n, stx:stx+n]
    dark       = dark_stack.mean(axis=0)
else:
    dark_stack = None
    dark       = np.zeros((n, n), dtype='float32')

# Local slice of data frames for this rank
local_data_ids = data_ids[cl.st_theta:cl.end_theta]
raw = fr.frames(local_data_ids)[:, sty:sty+n, stx:stx+n] - dark[None]
np.maximum(raw, 0, out=raw)
fr.close()

# float64: the drift being removed is ~2%, the same order as the dark offset.
frame_mean = raw.mean(axis=(1, 2), dtype='float64', keepdims=True)

# get_local_chunk is a contiguous block partition in rank order, so gathering
# in rank order puts the means back in frame order.
_all_fm = comm.gather(frame_mean.ravel(), root=0)
if rank == 0:
    fm = np.concatenate(_all_fm)
    logger.info(f'frame means (ADU): mean={fm.mean():.1f} first={fm[0]:.1f} '
                f'last={fm[-1]:.1f}  p-p={100*np.ptp(fm)/fm.mean():.2f}%')
    # The spiral runs centre-outward, so a drift in frame index is RADIAL.
    logger.info(f'  corr(frame index, mean) = '
                f'{np.corrcoef(np.arange(len(fm)), fm)[0, 1]:+.3f}')
    if dark_stack is not None:
        _dm = dark_stack.mean(axis=(1, 2))
        logger.info(f'dark: mean={_dm.mean():.2f} ADU  p-p={np.ptp(_dm):.2f} ADU')
else:
    fm = None

# Per-frame flux normalisation -- what ../AtomiumS1_FT_RD300/steps15.py:419-421 already
# does on the tomo path.  Target is 1.0 on every frame, so the stack keeps the
# ~1 scale rho and the PSF were tuned at, and no collective is needed.
# NOT ported from steps15: its median-filter outlier removal.  This scan has no
# zingers (0 px above 10x the mean, max/mean 8.1) and a radius-9 median would
# risk smearing a 1.5%-contrast object.  See README.md.
frame_mean[frame_mean <= 0] = 1.0
raw /= frame_mean.astype('float32')

cl.data[:]          = raw
cl.vars['proj'][:]  = 0
cl.vars['prb'][:]   = 1
cl.vars['pos'][:]   = cp.array(pos[cl.st_theta:cl.end_theta])

# ---------------------------------------------------------------------------
# Reconstruct
# ---------------------------------------------------------------------------
cl.BH()

# ---------------------------------------------------------------------------
# Collect results and write HDF5
# ---------------------------------------------------------------------------
pos_final_local = cl.vars['pos'].get()
pos_init_local  = cl.pos_init.get()
pos_err_local   = pos_final_local - pos_init_local

all_pos_err = comm.gather(pos_err_local, root=0)
if rank == 0:
    pos_err = np.concatenate(all_pos_err, axis=0)
    logger.info(f'position errors y (pix): max={np.abs(pos_err[:,0]).max():.4f}  mean={np.abs(pos_err[:,0]).mean():.4f}  std={pos_err[:,0].std():.4f}')
    logger.info(f'position errors x (pix): max={np.abs(pos_err[:,1]).max():.4f}  mean={np.abs(pos_err[:,1]).mean():.4f}  std={pos_err[:,1].std():.4f}')

    prb_np  = cl.vars['prb'].get()
    proj_np = cl.vars['proj'].get()

    os.makedirs(os.path.dirname(h5_out) or '.', exist_ok=True)
    with h5py.File(h5_out, 'w') as f:
        f.create_dataset('prb_amp',    data=np.abs(prb_np)[None])
        f.create_dataset('prb_phase',  data=np.angle(prb_np)[None])
        f.create_dataset('proj_delta', data=proj_np.real[None])
        f.create_dataset('proj_beta',  data=proj_np.imag[None])
        f.create_dataset('pos_err',    data=pos_err)
        f.create_dataset('pos_init',   data=pos)
        # The per-frame flux, so the drift and the correction applied stay
        # recoverable without re-reading 419 MB of raw frames.
        f.create_dataset('frame_mean', data=fm)
        f.attrs['voxelsize']     = voxelsize
        f.attrs['energy']        = energy
        f.attrs['z1']            = z1
        f.attrs['z_total']       = z_total
        f.attrs['magnification'] = magnification
    logger.info(f'Saved to {h5_out}')
