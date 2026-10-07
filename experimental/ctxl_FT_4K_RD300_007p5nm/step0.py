#!/usr/bin/env python
"""
NFP probe retrieval from the ESRF NXtomo (.nx) companion of this scan.

Launch with:
    mpirun -n <N> python step0.py config_step0.conf

WHY THIS COPY EXISTS.  ../Siemens/step0.py already reads an NXtomo, and every
line of geometry and position handling below is its work unchanged.  What it
cannot do is read THIS scan's pixels: our copy of the 2026 tree drops the
PROCESSED_DATA level the NXtomo's virtual-source paths are written against, so
`f[entry + "/instrument/detector/data"][ids]` returns the fill value -- zeros,
with no error at all.  Every frame read therefore goes through
esrf_layout.NxFrames, which takes the virtual mapping apart, rebases each
source onto the real RAW_DATA file and reads from there; see that module's
docstring.  Nothing else differs, so diff against ../Siemens/step0.py to see
exactly that substitution and nothing more.

AND WHY IT IS WORTH RUNNING HERE.  The 4-distance scan next door cannot run
this at all -- its NFP companions point at RAW_DATA that was never transferred
-- so ../ctxl_HT_4K_RD300_007p5nm/config_step6_bin2.conf starts step 6 from a
flat probe.  This scan's raw NFP frames ARE on eagle
(RAW_DATA/ctxl/<pfile>/scan0004 = 20 darks, scan0005 = 50 NFP frames), and a
measured probe matters more here than it would there: with a single distance,
the probe and the object are separated by the +-300 px displacement diversity
alone, so handing step 6 a measured probe removes one of the two things it
would otherwise have to invent.  Feed the result to step 6 with prb_file= in
config_step6_bin2.conf -- see the note there.
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
from esrf_layout import NxFrames

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

# Positions in object-plane pixels: spy = x_trans (y motor), spz = y_trans (z motor)
spy = meta['x_trans'] * 1e-3   # mm → m
spz = meta['y_trans'] * 1e-3   # mm → m
spy -= spy.mean()
spz -= spz.mean()
pos = np.stack([-spz / voxelsize, spy / voxelsize], axis=-1).astype('float32')

if rank == 0:
    logger.info(f'nx_file                 = {nx_file}')
    logger.info(f'entry                   = {meta["entry"]}')
    logger.info(f'energy                  = {energy} keV')
    logger.info(f'z1                      = {z1*1e3:.4f} mm')
    logger.info(f'focustodetectordistance = {z_total*1e3:.2f} mm')
    logger.info(f'magnification           = {magnification:.2f}')
    logger.info(f'voxelsize               = {voxelsize*1e9:.2f} nm')
    logger.info(f'ntheta                  = {ntheta}  (dark={len(dark_ids)} flat={len(flat_ids)})')
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
    shift_type              = args.shift_type,
    estimate_rho            = args.estimate_rho,
    rho_estimate_niter      = args.rho_estimate_niter,
    rho_trial_error_step    = args.rho_trial_error_step,
    path_out                = _path_out,
    comm                    = comm,
)

cl = RecNFP(rec_args)

# ---------------------------------------------------------------------------
# Load data: dark-subtract, normalise by global mean.
#
# cl.data is the INTENSITY, not its square root: RecNFP.F0 compares K|psi|^2
# with d (see the F0 block in rec_nfp_mpi).  It used to be sqrt(I) because the
# misfit was on amplitudes.  Normalising by the global mean still puts it at
# ~1, which is the scale rho and the PSF were tuned at.
# ---------------------------------------------------------------------------
fr = NxFrames(nx_file)
if fr.missing:
    raise SystemExit(
        f'{len(fr.missing)} of the virtual sources of {nx_file} do not exist, '
        f'e.g.\n    {fr.missing[0]}\n'
        'Reading through them would silently return zeros.  The NFP frames '
        'live in RAW_DATA/ctxl/<pfile>/scan000[45]/balor_0000.h5; copy that '
        'tree next to the projections tree, as esrf_layout._rebase expects.')

# Dark field (read once on each rank -- small)
if len(dark_ids) > 0:
    dark = fr.frames(dark_ids)[:, sty:sty+n, stx:stx+n].mean(axis=0)
else:
    dark = np.zeros((n, n), dtype='float32')

# Local slice of data frames for this rank
local_data_ids = data_ids[cl.st_theta:cl.end_theta]
raw = fr.frames(local_data_ids)[:, sty:sty+n, stx:stx+n] - dark[None]
np.maximum(raw, 0, out=raw)
fr.close()

local_sum  = float(raw.sum())
global_sum = comm.allreduce(local_sum, op=MPI.SUM)
global_mean = global_sum / (ntheta * n * n)

cl.data[:]          = raw / (global_mean + 1e-5)
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
    logger.info(f'Saved to {h5_out}')
