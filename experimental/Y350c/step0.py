#!/usr/bin/env python
"""
NFP reconstruction for all 4 datasets.

Launch with:
    mpirun -n <N> python step0.py config_step0.conf
"""

import sys
import json
import os
import numpy as np
import cupy as cp
import h5py
from types import SimpleNamespace
from mpi4py import MPI
from holotomocupy.rec_nfp_mpi import RecNFP
from holotomocupy.config import parse_args_step0
from holotomocupy.utils import *

args = parse_args_step0(sys.argv[1])
logger.setLevel(args.log_level)

h5_out      = args.h5_out
dataset_ids = args.dataset_ids
_ref_id     = str(dataset_ids[0])   # index used in the conf-file paths
n        = args.n
niter    = args.niter
nchunk   = args.nchunk
checkpoint_step = args.checkpoint_step
error_step = args.error_step
rho      = args.rho

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _read_h5_field(h5path, suffix):
    result = {}
    def _visit(name, obj):
        if not result and isinstance(obj, h5py.Dataset) and name.endswith(suffix):
            result['val'] = obj[()]
    with h5py.File(h5path, 'r') as f:
        f.visititems(_visit)
    if not result:
        raise KeyError(f'{suffix!r} not found in {h5path}')
    return result['val']

def read_energy(h5path):
    return float(_read_h5_field(h5path, 'TOMO/energy'))

def read_sx0(h5path):
    return float(_read_h5_field(h5path, 'TOMO/sx0')) * 1e-3

def read_sx(h5path):
    names  = _read_h5_field(h5path, 'sample/positioners/name').decode().split()
    values = _read_h5_field(h5path, 'sample/positioners/value').decode().split()
    return float(values[names.index('sx')]) * 1e-3

def read_detector_pixelsize(h5path):
    par = json.loads(_read_h5_field(h5path, 'TOMO/FTOMO_PAR').decode())
    return float(par['image_pixel_size']) * 1e-6

def read_focustodetectordistance(h5path):
    return float(_read_h5_field(h5path, 'PTYCHO/focusToDetectorDistance')) * 1e-3

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

prb_amps    = []
prb_phases  = []
proj_deltas = []
proj_betas  = []
pos_errs    = []

for dataset_id in dataset_ids:
    _sid      = str(dataset_id)
    scan_file = args.scan_file.replace(f'_{_ref_id}_', f'_{_sid}_')
    meta_file = args.meta_file.replace(f'_{_ref_id}_', f'_{_sid}_')

    if rank == 0:
        logger.info(f'=== Dataset {dataset_id} ===')

    # --- Geometry ---
    energy                  = args.energy if args.energy is not None else read_energy(meta_file)
    z1_ref                  = read_sx0(meta_file)
    z1                      = read_sx(meta_file) - z1_ref
    detector_pixelsize      = read_detector_pixelsize(meta_file)
    focustodetectordistance = read_focustodetectordistance(meta_file)

    magnification = focustodetectordistance / z1
    voxelsize     = detector_pixelsize / magnification

    if rank == 0:
        logger.info(f'energy                  = {energy} keV')
        logger.info(f'z1                      = {z1*1e3:.4f} mm')
        logger.info(f'focustodetectordistance = {focustodetectordistance*1e3:.2f} mm')
        logger.info(f'magnification           = {magnification:.2f}')
        logger.info(f'voxelsize               = {voxelsize*1e9:.2f} nm')

    # --- Positions ---
    with h5py.File(scan_file, 'r') as f:
        dset    = f['entry_0000/instrument/PCIe/data']
        ntheta, ny, nx = dset.shape
        spy_str = f['entry_0000/instrument/PCIe/header/spy'][()]
        spz_str = f['entry_0000/instrument/PCIe/header/spz'][()]

    sty = (ny - n) // 2
    stx = (nx - n) // 2

    spy = np.array(spy_str.decode().split(), dtype='float32') * 1e-6
    spz = np.array(spz_str.decode().split(), dtype='float32') * 1e-6
    pos = np.stack([-spz / voxelsize, spy / voxelsize], axis=-1).astype('float32')
    if rank == 0:
        logger.info(f'positions (pix): y in [{pos[:,0].min():.2f}, {pos[:,0].max():.2f}],  z in [{pos[:,1].min():.2f}, {pos[:,1].max():.2f}]')

    pos_range = int(np.ceil(np.abs(pos).max())) + 8
    nobj      = int(np.ceil((n + 2 * pos_range) / 32)) * 32
    if rank == 0:
        logger.info(f'n = {n},  nobj = {nobj},  pos_range = ±{pos_range} pix')

    # --- Init RecNFP ---
    _path_out = os.path.join(args.path_out, f'nfp_dist{dataset_id}') if args.path_out else None
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
        checkpoint_step = checkpoint_step,
        error_step = error_step,
        start_iter              = 0,
        psf_sigma               = args.psf_sigma,
        shift_type              = args.shift_type,
        path_out                = _path_out,
        comm                    = comm,
    )

    cl = RecNFP(rec_args)

    # --- Load data ---
    with h5py.File(scan_file, 'r') as f:
        raw_slice = f['entry_0000/instrument/PCIe/data'][cl.st_theta:cl.end_theta, sty:sty+n, stx:stx+n].astype('float32')
    global_mean = comm.allreduce(raw_slice.sum(), op=MPI.SUM) / (ntheta * n * n)
    cl.data[:] = np.abs(raw_slice / (global_mean + 1e-5))

    cl.vars['proj'][:] = 0
    cl.vars['prb'][:] = 1
    cl.vars['pos'][:] = cp.array(pos[cl.st_theta:cl.end_theta])

    # --- Reconstruct ---
    cl.BH()

    # --- Position errors ---
    pos_final_local = cl.vars['pos'].get()
    pos_init_local  = cl.pos_init.get()
    pos_err_local   = pos_final_local - pos_init_local

    all_pos_err = comm.gather(pos_err_local, root=0)
    if rank == 0:
        pos_err = np.concatenate(all_pos_err, axis=0)
        logger.info(f'position errors y (pix): max={np.abs(pos_err[:,0]).max():.4f}, mean={np.abs(pos_err[:,0]).mean():.4f}, std={pos_err[:,0].std():.4f}')
        logger.info(f'position errors x (pix): max={np.abs(pos_err[:,1]).max():.4f}, mean={np.abs(pos_err[:,1]).mean():.4f}, std={pos_err[:,1].std():.4f}')

        prb_np  = cl.vars['prb'].get()
        proj_np = cl.vars['proj'].get()
        prb_amps.append(np.abs(prb_np))
        prb_phases.append(np.angle(prb_np))
        proj_deltas.append(proj_np.real)
        proj_betas.append(proj_np.imag)
        pos_errs.append(pos_err)

    del cl
    cp.get_default_memory_pool().free_all_blocks()

# ---------------------------------------------------------------------------
# Write combined h5
# ---------------------------------------------------------------------------
if rank == 0:
    with h5py.File(h5_out, 'w') as f:
        f.create_dataset('prb_amp',    data=np.stack(prb_amps))
        f.create_dataset('prb_phase',  data=np.stack(prb_phases))
        f.create_dataset('proj_delta', data=np.stack(proj_deltas))
        f.create_dataset('proj_beta',  data=np.stack(proj_betas))
        f.create_dataset('pos_err',    data=np.concatenate(pos_errs, axis=0))
    logger.info(f'Saved all datasets to {h5_out}')
