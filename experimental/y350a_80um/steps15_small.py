#!/usr/bin/env python
"""Stitch bin=3 projections (no Paganin) and save as TIFFs."""

import sys
import h5py
import glob
import json
import os
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import tifffile
from mpi4py import MPI
from holotomocupy.shift import Shift
from holotomocupy.logger_config import logger, set_log_level
from holotomocupy.config import parse_args_steps15
from holotomocupy.utils import *

args = parse_args_steps15(sys.argv[1])
rotation_center_shift = args.rotation_center_shift
set_log_level(args.log_level)

path  = args.path + '/'
pfile = args.pfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ngpus = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(rank % ngpus).use()


# ---------------------------------------------------------------------------
# Helpers — read geometry from HDF5 scan files
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
    if 'sx' not in names:
        raise ValueError(f"'sx' not found in positioners for {h5path}.\nAvailable: {names}")
    return float(values[names.index('sx')]) * 1e-3

def read_detector_pixelsize(h5path):
    par = json.loads(_read_h5_field(h5path, 'TOMO/FTOMO_PAR').decode())
    return float(par['image_pixel_size']) * 1e-6

def read_focustodetectordistance(h5path):
    return float(_read_h5_field(h5path, 'PTYCHO/focusToDetectorDistance')) * 1e-3


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

path_out = args.path_out if args.path_out else path.rstrip('/') + '_rec'
file_out = f'{pfile}.h5'
fpath    = f'{path_out}/{file_out}'

dirs    = sorted(glob.glob(f'{path}/{pfile}_[0-9]_/'))
h5files = [sorted(glob.glob(f'{d}/*.h5'))[0] for d in dirs]
ndist   = len(h5files)

energy                  = read_energy(h5files[0])
detector_pixelsize      = read_detector_pixelsize(h5files[0])
focustodetectordistance = read_focustodetectordistance(h5files[0])
sx0                     = read_sx0(h5files[0])
z1                      = np.array([read_sx(f) for f in h5files]) - sx0

z2                  = focustodetectordistance - z1
magnifications      = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]
voxelsize           = abs(detector_pixelsize / magnifications[0])

with h5py.File(fpath) as fid:
    ntheta = fid['/exchange/theta'].shape[0]
    n      = fid['/exchange/pdata0'].shape[1]

from holotomocupy.reader import load_shrink_from_mats
shrink_nd = load_shrink_from_mats(path, pfile, ndist, ntheta)
eff_magnifications = norm_magnifications[:, None] / (1 + shrink_nd[0])   # [ndist, 2] (y, x)

nobj = args.nobj if args.nobj is not None else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64

bin      = 3
n_bin    = n    // (2**bin)
nobj_bin = nobj // (2**bin)

tiff_dir  = f'{path_out}/srdata_bin{bin}'
rdata_dir = f'{path_out}/rdata_bin{bin}'
data_dir  = f'{path_out}/data_bin{bin}'
if rank == 0:
    logger.info(f'path={path}  pfile={pfile}  ntheta={ntheta}  ndist={ndist}  n={n}  nobj={nobj}')
    logger.info(f'bin={bin}  n_bin={n_bin}  nobj_bin={nobj_bin}')
    logger.info(f'shrink[0]               = {np.round(np.asarray(shrink_nd[0], dtype='float64'), 6).tolist()}')
    os.makedirs(tiff_dir,  exist_ok=True)
    os.makedirs(rdata_dir, exist_ok=True)
    os.makedirs(data_dir,  exist_ok=True)
comm.Barrier()

ids_per_rank = np.array_split(np.arange(ntheta)[::50], size)
local_ids    = ids_per_rank[rank]
logger.info(f'theta-range [{int(local_ids[0])}:{int(local_ids[-1])+1}), local_ntheta={len(local_ids)}')


# ---------------------------------------------------------------------------
# Read cshifts
# ---------------------------------------------------------------------------

if rank == 0:
    with h5py.File(fpath) as fid:
        cshifts = fid['/exchange/cshifts_final'][:].astype('float32')
else:
    cshifts = np.empty([ntheta, ndist, 2], dtype='float32')
comm.Bcast(cshifts, root=0)

scale = 1.0 / 2**bin
r = (cshifts * scale).astype('float32')
r[..., 1] += rotation_center_shift * scale
r_gpu = cp.array(r)

if rank == 0:
    with h5py.File(fpath) as fid:
        ref     = fid[f'/exchange/pref_{bin}'][:ndist].astype('float32')
        ref_end = fid[f'/exchange/pref_end_{bin}'][:ndist].astype('float32') if f'/exchange/pref_end_{bin}' in fid else ref.copy()
else:
    ref     = np.empty([ndist, n_bin, n_bin], dtype='float32')
    ref_end = np.empty([ndist, n_bin, n_bin], dtype='float32')
comm.Bcast(ref,     root=0)
comm.Bcast(ref_end, root=0)

cref     = cp.array(ref)
cref_end = cp.array(ref_end)
fwhm_ref    = 17.0 * (n_bin / 2048)
sigma_ref   = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
cref_smooth     = cp.stack([ndimage.gaussian_filter(cref[k],     sigma_ref) for k in range(ndist)])
cref_end_smooth = cp.stack([ndimage.gaussian_filter(cref_end[k], sigma_ref) for k in range(ndist)])
t_scale = max(ntheta - 1, 1)
cl_shift = Shift(n_bin, nobj_bin, n_bin, nobj_bin)
npad_bin = n_bin // 16
v_bin    = cp.linspace(0, 1, npad_bin, endpoint=False)
v_bin    = v_bin**5 * (126 - 420*v_bin + 540*v_bin**2 - 315*v_bin**3 + 70*v_bin**4)


# ---------------------------------------------------------------------------
# Stitch and save TIFFs
# ---------------------------------------------------------------------------

def _stitch(fid, srdata, j):
    data_j = cp.empty([ndist, n_bin, n_bin], dtype='float32')
    for k in range(ndist):
        data_j[k] = cp.array(fid[f'/exchange/pdata{k}_{bin}'][j].astype('float32'))
    t = float(j) / t_scale
    cref_chunk_smooth = (1 - t) * cref_smooth + t * cref_end_smooth
    data_j_smooth = cp.stack([ndimage.gaussian_filter(data_j[k], sigma_ref) for k in range(ndist)])
    rdata = data_j_smooth / (cref_chunk_smooth + 1e-5)
    srdata.fill(0)
    for k in range(ndist - 1, -1, -1):
        shrink_jk  = shrink_nd[j, k]                      # (2,) y, x
        eff_mag_jk = float(norm_magnifications[k]) / (1 + shrink_jk)   # (2,)
        if j%100==0:
            print(j,k,eff_mag_jk)
        mag        = cp.array(1.0 / eff_mag_jk, dtype='float32')[None]
        tmp = rdata[k].astype('complex64')
        tmp = cl_shift.curlySback(
            cp.log(tmp[None]).astype('complex64'), r_gpu[j:j+1, k], mag
        )[0].real
        tmp = cp.exp(tmp)
        padx0 = int((nobj_bin - n_bin / eff_mag_jk[1]) / 2) - int(r[j, k, 1])
        pady0 = int((nobj_bin - n_bin / eff_mag_jk[0]) / 2) - int(r[j, k, 0])
        padx1 = int((nobj_bin - n_bin / eff_mag_jk[1]) / 2) + int(r[j, k, 1])
        pady1 = int((nobj_bin - n_bin / eff_mag_jk[0]) / 2) + int(r[j, k, 0])
        padx0 = min(nobj_bin, max(0, padx0)) + 5
        pady0 = min(nobj_bin, max(0, pady0)) + 5
        padx1 = min(nobj_bin, max(0, padx1)) + 5
        pady1 = min(nobj_bin, max(0, pady1)) + 5
        tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)), 'edge')
        tmp = cp.pad(tmp[:, padx0:-padx1], ((0, 0), (padx0, padx1)),
                     'linear_ramp', end_values=((1, 1), (1, 1)))
        if k < ndist - 1:
            denom = tmp[pady0:-pady1, padx0:-padx1].mean() + 1e-10
            mmm   = float(srdata[k+1][pady0:-pady1, padx0:-padx1].mean() / denom)
            tmp  *= mmm
            cs   = min(nobj_bin // 16, (nobj_bin - pady0 - pady1) // 2, (nobj_bin - padx0 - padx1) // 2)
            ch   = cs // 2
            midy = nobj_bin // 2
            midx = nobj_bin // 2
            ys   = [pady0,        midy - ch,        nobj_bin - pady1 - cs]
            xs   = [padx0,        midx - ch,        nobj_bin - padx1 - cs]
            ref  = srdata[k + 1]
            R = cp.array([[float(ref[y:y+cs, x:x+cs].mean() / (tmp[y:y+cs, x:x+cs].mean() + 1e-10))
                            for x in xs] for y in ys], dtype='float32')
            ratio_map = ndimage.zoom(R, nobj_bin / 3, order=1)
            tmp *= ratio_map[:nobj_bin, :nobj_bin]
            wx = cp.ones(nobj_bin, dtype='float32')
            wy = cp.ones(nobj_bin, dtype='float32')
            wx[:padx0]                 = 0
            wx[padx0:padx0+npad_bin]   = v_bin
            wx[-padx1-npad_bin:-padx1] = 1 - v_bin
            wx[-padx1:]                = 0
            wy[:pady0]                 = 0
            wy[pady0:pady0+npad_bin]   = v_bin
            wy[-pady1-npad_bin:-pady1] = 1 - v_bin
            wy[-pady1:]                = 0
            tmp = tmp * cp.outer(wy, wx) + srdata[k+1] * (1 - cp.outer(wy, wx))
        srdata[k] = tmp
    return rdata, data_j

srdata = cp.zeros([ndist, nobj_bin, nobj_bin], dtype='float32')

with h5py.File(fpath) as fid:
    for i, j in enumerate(local_ids):
        rdata, data_j = _stitch(fid, srdata, j)
        for k in range(ndist):
            tifffile.imwrite(f'{tiff_dir}/ang{j:04d}_dist{k}.tiff',  srdata[k].get())
            tifffile.imwrite(f'{rdata_dir}/ang{j:04d}_dist{k}.tiff', rdata[k].get())
            tifffile.imwrite(f'{data_dir}/ang{j:04d}_dist{k}.tiff',  data_j[k].get())
        if i % 100 == 0:
            logger.info(f'stitching proj {int(j):4d}/{ntheta}')

comm.Barrier()
if rank == 0:
    logger.info('Done.')
