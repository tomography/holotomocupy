#!/usr/bin/env python
"""
Steps 1–3 — Convert EDF→HDF5, preprocess, and combine shifts (MPI + GPU parallel).

Step 1: read raw EDF projections → parallel HDF5
Step 2: outlier removal + intensity normalisation (GPU)
Step 3: combine encoder / RHAPP / motion / 3-D-correction shifts → cshifts_final

Launch with:
    mpirun -n <N> python steps_15.py steps15.conf
"""

import sys
import logging
import h5py
import fabio
logging.getLogger('fabio').setLevel(logging.ERROR)
import glob
import json
import os
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from holotomocupy.shift import Shift
from holotomocupy.tomo import Tomo
from holotomocupy.chunking import Chunking
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.logger_config import logger, set_log_level
from holotomocupy.config import parse_args_steps15
from holotomocupy.reader import load_octave_text_mat, load_shrink_from_mats
from holotomocupy.utils import *

args = parse_args_steps15(sys.argv[1])
start_step            = args.start_step
rotation_center_shift = args.rotation_center_shift
nlevels               = args.nlevels
start_level_rec       = args.start_level_rec
paganin               = args.paganin
nchunk                = args.nchunk
ref_dist              = args.ref_dist
set_log_level(args.log_level)

path  = args.path + '/'
pfile = args.pfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Assign one GPU per rank (round-robin if fewer GPUs than ranks)
ngpus = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(rank % ngpus).use()


# ---------------------------------------------------------------------------
# Helpers — read geometry from HDF5 scan files
# ---------------------------------------------------------------------------

def _read_h5_field(h5path, suffix):
    """Return the value of the first dataset whose path ends with `suffix`."""
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
        raise ValueError(f"'sx' not found in positioners for {h5path}.\n"
                         f"Available names: {names}")
    return float(values[names.index('sx')]) * 1e-3

def read_detector_pixelsize(h5path):
    par = json.loads(_read_h5_field(h5path, 'TOMO/FTOMO_PAR').decode())
    return float(par['image_pixel_size']) * 1e-6

def read_focustodetectordistance(h5path):
    return float(_read_h5_field(h5path, 'PTYCHO/focusToDetectorDistance')) * 1e-3

def find_angle(fname):
    with open(fname, 'rb') as f:
        buf = b''
        while b'motor_pos' not in buf:
            chunk = f.read(512)
            if not chunk:
                break
            buf += chunk
    for line in buf.decode('latin-1').split('\n'):
        if 'motor_pos' in line:
            return float(line.split()[3])




# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

path_out = args.path_out if args.path_out else path.rstrip('/') + '_rec'
file_out = f'{pfile}.h5'

# Auto-detect ntheta from ref filenames
dname0 = f'{path}/{pfile}_1_'
ntheta = max(int(f.split('_')[-1].split('.')[0])
             for f in glob.glob(f'{dname0}/ref0000_*.edf'))

# Auto-extract geometry — one H5 file per distance directory
dirs    = sorted(glob.glob(f'{path}/{pfile}_[0-9]_/'))
h5files = [sorted(glob.glob(f'{d}/*.h5'))[0] for d in dirs]
ndist   = len(h5files)

energy                  = read_energy(h5files[0])
detector_pixelsize      = read_detector_pixelsize(h5files[0])
focustodetectordistance = read_focustodetectordistance(h5files[0])
sx0                     = read_sx0(h5files[0])
z1                      = np.array([read_sx(f) for f in h5files]) - sx0

wavelength          = 1.24e-09 / energy
z2                  = focustodetectordistance - z1
magnifications      = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]
distances           = (z1 * z2) / focustodetectordistance * norm_magnifications**2
voxelsizes          = np.abs(detector_pixelsize / magnifications)
voxelsize           = voxelsizes[0]

shrink_nd = load_shrink_from_mats(path, pfile, ndist, ntheta)  # [ntheta, ndist, 2] (y, x)
shrink = shrink_nd[0]  # first-angle cumulative values, used for module-level eff_magnifications
eff_magnifications = norm_magnifications[:, None] / (1 + shrink)   # [ndist, 2] (y, x)

# n from actual EDF file size (images are n×n), overrideable via --n
n0, n1 = fabio.open(f'{dname0}/ref0000_0000.edf').data.shape
n = args.n if args.n is not None else n0
sty, endy = n0 // 2 - n // 2, n0 // 2 + n // 2
stx, endx = n1 // 2 - n // 2, n1 // 2 + n // 2

# Auto-detect number of flat / dark frames by counting files at angle 0
nref  = len(glob.glob(f'{dname0}/ref*_0000.*'))
ndark = len(glob.glob(f'{dname0}/darkend*.*'))

# Stitched object size (same at all steps that use it: 4, 5), overrideable via --nobj
nobj = args.nobj if args.nobj is not None else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64

if rank == 0:
    logger.info(f'path                    = {path}')
    logger.info(f'pfile                   = {pfile}')
    logger.info(f'ntheta                  = {ntheta}')
    logger.info(f'energy                  = {energy} keV')
    logger.info(f'detector_pixelsize      = {detector_pixelsize} m')
    logger.info(f'focustodetectordistance = {focustodetectordistance} m')
    logger.info(f'sx0                     = {sx0} m')
    logger.info(f'z1                      = {z1} m')
    logger.info(f'ndist={ndist}  n={n}  nobj={nobj}  nref={nref}  ndark={ndark}')
    logger.info(f'shrink cumulative       = {np.round(np.asarray(shrink, dtype='float64'), 6).tolist()}')
    logger.debug(f'wavelength              = {wavelength} m')
    logger.debug(f'magnifications          = {magnifications}')
    logger.debug(f'voxelsizes              = {voxelsizes} m')
    os.makedirs(path_out, exist_ok=True)
comm.Barrier()

# Distribute ntheta projections across ranks
ids_per_rank = np.array_split(np.arange(ntheta), size)
local_ids    = ids_per_rank[rank]
local_start  = int(local_ids[0])
local_end    = int(local_ids[-1]) + 1
logger.info(f'theta-range [{local_start}:{local_end}), local_ntheta={local_end - local_start}')


# ===========================================================================
# STEP 1: Convert EDF → HDF5
# ===========================================================================

fpath = f'{path_out}/{file_out}'

if start_step > 1:
    logger.info('Step 1: skipped.')
    comm.Barrier()
else:
    logger.info('Step 1: converting EDF files to HDF5...')

    # Angles: each rank reads its own subset in parallel, then rank 0 gathers
    local_fnames = [f'{dname0}/{pfile}_1_{id:04}.edf' for id in local_ids]
    with ThreadPoolExecutor() as pool:
        local_theta = np.array(list(pool.map(find_angle, local_fnames)), dtype='float32')

    all_theta_parts = comm.gather(local_theta, root=0)
    if rank == 0:
        theta_vals = np.concatenate(all_theta_parts)

    with h5py.File(fpath, 'w', driver='mpio', comm=comm) as fid:

        # Collective: all ranks create every dataset
        data_ds   = [fid.create_dataset(f'/exchange/data{k}',             shape=(ntheta, n, n), dtype='uint16') for k in range(ndist)]
        white0_ds = [fid.create_dataset(f'/exchange/data_white_start{k}', shape=(nref,  n, n),  dtype='uint16') for k in range(ndist)]
        white1_ds = [fid.create_dataset(f'/exchange/data_white_end{k}',   shape=(nref,  n, n),  dtype='uint16') for k in range(ndist)]
        dark_ds   = [fid.create_dataset(f'/exchange/data_dark{k}',        shape=(ndark, n, n),  dtype='uint16') for k in range(ndist)]
        theta_ds  = fid.create_dataset('/exchange/theta',  shape=(ntheta, ndist), dtype='float32')
        shifts_ds = fid.create_dataset('/exchange/shifts', shape=(ntheta, ndist, 2), dtype='float32')
        attrs_ds  = fid.create_dataset('/exchange/attrs',  shape=(ntheta, ndist, 3), dtype='float32')
        vs_ds     = fid.create_dataset('/exchange/voxelsize',             shape=voxelsizes.shape, dtype='float32')
        z1_ds     = fid.create_dataset('/exchange/z1',                    shape=z1.shape,         dtype='float32')
        dpx_ds    = fid.create_dataset('/exchange/detector_pixelsize',    shape=(1,),             dtype='float32')
        en_ds     = fid.create_dataset('/exchange/energy',                shape=(1,),             dtype='float32')
        fdd_ds    = fid.create_dataset('/exchange/focusdetectordistance', shape=(1,),             dtype='float32')

        if rank == 0:
            vs_ds[:]    = voxelsizes
            z1_ds[:]    = z1
            dpx_ds[:]   = [detector_pixelsize]
            en_ds[:]    = [energy]
            fdd_ds[:]   = [focustodetectordistance]
            theta_ds[:] = theta_vals[:, None]

        for k in range(ndist):
            dname = f'{path}/{pfile}_{k + 1}_'

            shifts_all = np.loadtxt(f'{dname}/correct.txt',    dtype='float32')[:ntheta]
            attrs_all  = np.loadtxt(f'{dname}/attributes.txt', dtype='float32')[:ntheta, :3]
            shifts_ds[local_start:local_end, k] = shifts_all[local_start:local_end]
            attrs_ds[local_start:local_end,  k] = attrs_all[local_start:local_end]

            if rank == 0:
                for id in range(nref):
                    white0_ds[k][id] = fabio.open(f'{dname}/ref{id:04}_0000.edf').data[sty:endy, stx:endx]
                    white1_ds[k][id] = fabio.open(f'{dname}/ref{id:04}_{ntheta:04}.edf').data[sty:endy, stx:endx]
                for id in range(ndark):
                    dark_ds[k][id]   = fabio.open(f'{dname}/darkend{id:04}.edf').data[sty:endy, stx:endx]

            norms = np.empty(len(local_ids), dtype='float64')
            for ii, id in enumerate(local_ids):
                fname = f'{dname}/{pfile}_{k + 1}_{id:04}.edf'
                frame = fabio.open(fname).data[sty:endy, stx:endx]
                data_ds[k][id] = frame
                norms[ii] = np.linalg.norm(frame)
                if ii%100==0:
                    logger.info(f'step1: proj {int(id):4d}/{ntheta}, dist {k+1}/{ndist}, norm={norms[ii]:.3e}')

            ref_norm = np.median(norms)
            for ii, id in enumerate(local_ids):
                if norms[ii] < ref_norm / 10:
                    logger.warning(f'step1: broken frame proj={int(id)} dist={k+1}  norm={norms[ii]:.3e}  median={ref_norm:.3e}')
                    prev_id = local_ids[ii - 1] if ii > 0 else None
                    next_id = local_ids[ii + 1] if ii < len(local_ids) - 1 else None
                    if prev_id is not None and next_id is not None:
                        rep = 0.5 * (data_ds[k][prev_id].astype('float32') +
                                     data_ds[k][next_id].astype('float32'))
                    elif prev_id is not None:
                        rep = data_ds[k][prev_id].astype('float32')
                    else:
                        rep = data_ds[k][next_id].astype('float32')
                    data_ds[k][id] = np.round(rep).astype(data_ds[k].dtype)

    comm.Barrier()
    logger.info('Step 1: done.')


# ===========================================================================
# STEP 2: Preprocessing (outlier removal + intensity normalisation)
# ===========================================================================

if start_step > 2:
    logger.info('Step 2: skipped.')
    comm.Barrier()
else:
    logger.info('Step 2: preprocessing...')

    radius     = 9
    threshold  = 0.9
    chunk_size = 16

    def remove_outliers(data, radius, threshold):
        fdata = ndimage.median_filter(data, size=(1, radius, radius))
        mask  = cp.abs(data - fdata) > fdata * threshold
        return cp.where(mask, fdata, data)

    # --- Rank 0 reads flat/dark fields, computes ref -----------------------
    if rank == 0:
        ref0 = np.empty([nref,  ndist, n, n], dtype='float32')
        dark_arr = np.empty([ndark, ndist, n, n], dtype='float32')
        with h5py.File(fpath) as fid:
            for k in range(ndist):
                ref0[:, k]     = fid[f'/exchange/data_white_start{k}'][:, :n, :n]
                dark_arr[:, k] = fid[f'/exchange/data_dark{k}'][:, :n, :n]

        dark = np.mean(dark_arr, axis=0).astype('float32')   # [ndist, n, n]
        ref  = np.mean(ref0,     axis=0).astype('float32')   # [ndist, n, n]

        ref_gpu  = cp.array(ref) - cp.array(dark)
        ref_gpu[ref_gpu < 0] = 1e-3
        # ref_gpu[:, 1402:1430, 844:872] = ref_gpu.mean(axis=(1, 2), keepdims=True)
        ref_gpu[:] = remove_outliers(ref_gpu, radius, threshold)
        ref = ref_gpu.get()
    else:
        ref  = np.empty([ndist, n, n], dtype='float32')
        dark = np.empty([ndist, n, n], dtype='float32')

    comm.Bcast(ref,  root=0)
    comm.Bcast(dark, root=0)

    dark_gpu = cp.array(dark)

    # --- Compute mean_data_ref on rank 0, broadcast ------------------------
    if rank == 0:
        mean_data_ref = np.zeros(ndist, dtype='float32')
        with h5py.File(fpath) as fid:
            for k in range(ndist):
                data = cp.array(fid[f'/exchange/data{k}'][0, :n, :n].astype('float32'))
                data -= dark_gpu[k]
                data[data < 0] = 0
                data = remove_outliers(data[None], radius, threshold)[0]
                mean_data_ref[k] = float(data.mean())

        mmr = np.mean(ref, axis=(1, 2))
        mean_data_ref *= mmr[0] / mmr[:]
        ref           *= mmr[0] / mmr[:, None, None]
        mean_data_ref /= mmr[0]
        ref           /= mmr[0]
    else:
        mean_data_ref = np.zeros(ndist, dtype='float32')

    comm.Bcast(mean_data_ref, root=0)
    comm.Bcast(ref,           root=0)   # ref was rescaled above

    # --- Rank 0: write pref, delete any existing pdata ---------------------
    if rank == 0:
        with h5py.File(fpath, 'a') as fid:
            if '/exchange/pref' in fid:
                del fid['/exchange/pref']
            fid.create_dataset('/exchange/pref', data=ref)
            for k in range(ndist):
                if f'/exchange/pdata{k}' in fid:
                    del fid[f'/exchange/pdata{k}']
    comm.Barrier()

    # --- All ranks write pdata in parallel ---------------------------------
    ref_gpu = cp.array(ref)
    with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
        pdata_ds = [fid.create_dataset(f'/exchange/pdata{k}', shape=(ntheta, n, n), dtype='float32')
                    for k in range(ndist)]

        for k in range(ndist):
            for j in range(local_start, local_end, chunk_size):
                end = min(j + chunk_size, local_end)

                data = cp.array(fid[f'/exchange/data{k}'][j:end, :n, :n].astype('float32'))
                data -= dark_gpu[k]
                data[data < 0] = 0
                # data[:, 1402:1430, 844:872] = data.mean(axis=(1, 2), keepdims=True) ## broken region on Ximea detector
                data[:] = remove_outliers(data, radius, threshold)
                _mean = data.mean(axis=(1, 2), keepdims=True)
                _mean[_mean == 0] = 1  # dark/blocked projections: keep as-is, avoid NaN
                data *= mean_data_ref[k] / _mean
                data[~cp.isfinite(data)] = 1

                pdata_ds[k][j:end] = data.get()

                if j % 100 == 0:
                    logger.info(f'step2: proj {j:4d}/{ntheta}, dist {k+1}/{ndist}, mean={float(data[0].mean()):.4f}')

    # Print per-rank norm of pdata (accumulated over all distances and local projections)
    with h5py.File(fpath, 'r', driver='mpio', comm=comm) as fid:
        _norm_sq = 0.0
        _rbatch = max(1, (1 << 28) // (n * n))
        for k in range(ndist):
            ds = fid[f'/exchange/pdata{k}']
            for _i0 in range(local_start, local_end, _rbatch):
                _i1 = min(_i0 + _rbatch, local_end)
                _chunk = cp.array(ds[_i0:_i1])
                _norm_sq += float(cp.linalg.norm(_chunk)**2)
    logger.info(f'step2: rank {rank:4d}  pdata norm = {_norm_sq**0.5:.6e}')

    logger.info('Step 2: done.')


# ===========================================================================
# STEP 3: Combine shifts
# ===========================================================================

# All work is tiny numpy — rank 0 does it, writes result, others wait.
if rank == 0:
    if start_step > 3:
        logger.info('Step 3: skipped.')
    else:
        logger.info('Step 3: combining shifts...')

        logger.info(f'Step 3: reading shifts      from {fpath}  [/exchange/shifts]')
        with h5py.File(fpath) as fid:
            shifts = fid['/exchange/shifts'][:]   # [ntheta, ndist, 2]

        # --- Encoder (random) shifts → object-plane pixels ---
        # axis 2 is (y, x) in detector pixels; swap to (row, col) and convert
        random_shifts = np.empty([ntheta, ndist, 2], dtype='float32')
        random_shifts[..., 0] = shifts[..., 1] / norm_magnifications
        random_shifts[..., 1] = shifts[..., 0] / norm_magnifications

        # --- RHAPP inter-plane shifts (from Peter's MATLAB pipeline) ---
        _rhapp_path = f'{path}/{pfile}_/rhapp.mat'
        if not os.path.exists(_rhapp_path):
            logger.warning(f'Step 3: rhapp.mat not found, using zeros: {_rhapp_path}')
            rhapp_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
        else:
            logger.info(f'Step 3: reading rhapp       from {_rhapp_path}')
            rhapp_raw = load_octave_text_mat(_rhapp_path, 'rhapp')
            rhapp_reordered = rhapp_raw.swapaxes(0, 2)[:ntheta]
            avg_plane_zero = rhapp_reordered[:, 0].mean(axis=0)   # [2]
            rhapp_reordered -= avg_plane_zero[np.newaxis, np.newaxis, :]
            logger.info(f'Step 3: avg_plane_zero  y={avg_plane_zero[0]:.4f} px   x={avg_plane_zero[1]:.4f} px')
            rhapp_shifts = (-rhapp_reordered).astype('float32')

        # --- Motion shifts (slow drift of reference plane) ---
        _motion_dname = f'{path}/{pfile}_{ref_dist+1}_'
        _motion_path = f'{_motion_dname}/correct_motion.txt'
        if not os.path.exists(_motion_path):
            logger.warning(f'Step 3: correct_motion.txt not found, using zeros: {_motion_path}')
            motion_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
        else:
            logger.info(f'Step 3: reading motion      from {_motion_path}')
            raw_motion = np.loadtxt(_motion_path)[:ntheta, ::-1].astype('float32')
            motion_base   = raw_motion / eff_magnifications[ref_dist] - random_shifts[:, ref_dist]
            motion_shifts = np.tile(motion_base[:, np.newaxis], (1, ndist, 1))

        # --- 3-D tomographic correction shifts ---
        _c3d_path = f'{path}/{pfile}_/correct_correct3D.txt'
        if os.path.exists(_c3d_path):
            logger.info(f'Step 3: reading correct3D   from {_c3d_path}')
            raw_3d = np.loadtxt(_c3d_path)[:ntheta, ::-1].astype('float32')
            correct3d_shifts = np.tile(raw_3d[:, np.newaxis], (1, ndist, 1))
        else:
            logger.info(f'Step 3: correct3D file not found, using zeros: {_c3d_path}')
            correct3d_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')

        # --- Sum all sources and save ---
        shifts_final = random_shifts + rhapp_shifts + motion_shifts + correct3d_shifts

        with h5py.File(fpath, 'a') as fid:
            if '/exchange/cshifts_final' in fid:
                del fid['/exchange/cshifts_final']
            fid.create_dataset('/exchange/cshifts_final', data=shifts_final)
            if '/exchange/shrink' in fid:
                del fid['/exchange/shrink']
            fid.create_dataset('/exchange/shrink', data=shrink_nd)

        logger.info('Step 3: done.')

comm.Barrier()


# ===========================================================================
# STEP 4: Make binned data (multi-distance alignment + amplitude correction)
# ===========================================================================

if start_step > 4:
    logger.info('Step 4: skipped.')
    comm.Barrier()
else:
    logger.info('Step 4: making binned data...')

    npad    = n // 16

    # --- Rank 0 reads ref and full shift array; broadcast to all ranks ----
    if rank == 0:
        with h5py.File(fpath) as fid:
            ref = fid['/exchange/pref'][:, :n, :n].astype('float32')   # [ndist, n, n]
            r   = fid['/exchange/cshifts_final'][:].astype('float32')
        r[..., 1] += rotation_center_shift
    else:
        ref = np.empty([ndist, n, n], dtype='float32')
        r   = np.empty([ntheta, ndist, 2], dtype='float32')

    comm.Bcast(ref, root=0)
    comm.Bcast(r,   root=0)

    # --- Rank 0 writes binned refs ----------------------------------------
    if rank == 0:
        ref0 = ref.copy()
        with h5py.File(fpath, 'a') as fid:
            for bin in range(nlevels):
                if f'/exchange/pref_{bin}' in fid:
                    del fid[f'/exchange/pref_{bin}']
                fid.create_dataset(f'/exchange/pref_{bin}', data=ref0)
                ref0 = 0.5 * (ref0[..., ::2] + ref0[..., 1::2])
                ref0 = 0.5 * (ref0[..., ::2, :] + ref0[..., 1::2, :])

            # Delete existing pdata{k}_{bin} datasets
            for bin in range(nlevels):
                for k in range(ndist):
                    if f'/exchange/pdata{k}_{bin}' in fid:
                        del fid[f'/exchange/pdata{k}_{bin}']
    comm.Barrier()

    # --- All ranks create output datasets collectively + process -----------
    cl_shift = Shift(n, nobj, n, nobj)
    cref     = cp.array(ref)

    # Smooth reference with Gaussian (Peter's approach): divide by blurred ref
    # to correct for large-scale illumination variations without removing
    # fine structure from the flat-field. FWHM = 17 * (n/2048) pixels.
    fwhm_ref  = 17.0 * (n / 2048)
    sigma_ref = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
    cref_smooth = cp.stack([ndimage.gaussian_filter(cref[k], sigma_ref) for k in range(ndist)])

    with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
        data_out = [[fid.create_dataset(f'/exchange/pdata{k}_{bin}',
                                        shape=(ntheta, n // 2**bin, n // 2**bin),
                                        dtype='float32')
                     for k in range(ndist)]
                    for bin in range(nlevels)]

        srdata = cp.zeros([ndist, nobj, nobj], dtype='float32')

        v = cp.linspace(0, 1, npad, endpoint=False)
        v = v**5 * (126 - 420*v + 540*v**2 - 315*v**3 + 70*v**4)

        for j in local_ids:
            data = cp.empty([ndist, n, n], dtype='float32')
            for k in range(ndist):
                data[k] = cp.array(fid[f'/exchange/pdata{k}'][j, :n, :n].astype('float32'))

            data_smooth = cp.stack([ndimage.gaussian_filter(data[k], sigma_ref) for k in range(ndist)])
            rdata = data_smooth / (cref_smooth + 1e-5)

            for k in range(ndist - 1, -1, -1):
                shrink_jk  = shrink_nd[j, k]                      # (2,) y, x
                eff_mag_jk = float(norm_magnifications[k]) / (1 + shrink_jk)   # (2,)
                mag = cp.array(1.0 / eff_mag_jk, dtype='float32')[None]
                tmp = rdata[k].astype('complex64')
                tmp = cl_shift.curlySback(
                    cp.log(tmp[None]).astype('complex64'),
                    cp.array(r[j:j+1, k]), mag
                )[0].real
                tmp = cp.exp(tmp)

                padx0 = int((nobj - n / eff_mag_jk[1]) / 2) - int(r[j, k, 1])
                pady0 = int((nobj - n / eff_mag_jk[0]) / 2) - int(r[j, k, 0])
                padx1 = int((nobj - n / eff_mag_jk[1]) / 2) + int(r[j, k, 1])
                pady1 = int((nobj - n / eff_mag_jk[0]) / 2) + int(r[j, k, 0])
                padx0 = min(nobj, max(0, padx0)) + 5
                pady0 = min(nobj, max(0, pady0)) + 5
                padx1 = min(nobj, max(0, padx1)) + 5
                pady1 = min(nobj, max(0, pady1)) + 5

                tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)), 'edge')
                tmp = cp.pad(tmp[:, padx0:-padx1], ((0, 0), (padx0, padx1)),
                             'linear_ramp', end_values=((1, 1), (1, 1)))

                if k < ndist - 1:
                    mmm = float(srdata[k + 1][pady0:-pady1, padx0:-padx1].mean() /
                                tmp[pady0:-pady1, padx0:-padx1].mean())
                    tmp     *= mmm
                    data[k] *= mmm
                    wx = cp.ones(nobj, dtype='float32')
                    wy = cp.ones(nobj, dtype='float32')
                    wx[:padx0]               = 0
                    wx[padx0:padx0 + npad]   = v
                    wx[-padx1 - npad:-padx1] = 1 - v
                    wx[-padx1:]              = 0
                    wy[:pady0]               = 0
                    wy[pady0:pady0 + npad]   = v
                    wy[-pady1 - npad:-pady1] = 1 - v
                    wy[-pady1:]              = 0
                    w   = cp.outer(wy, wx)
                    tmp = tmp * w + srdata[k + 1] * (1 - w)
                srdata[k] = tmp

            if j % 100 == 0:
                logger.info(f'step4: proj {int(j):4d}/{ntheta}')

            for k in range(ndist):
                datak = data[k]
                for bin in range(nlevels):
                    data_out[bin][k][j] = datak.get()
                    datak = 0.5 * (datak[::2, :] + datak[1::2, :])
                    datak = 0.5 * (datak[:, ::2]  + datak[:, 1::2])

    comm.Barrier()
    logger.info('Step 4: done.')


# ===========================================================================
# STEP 5: Paganin phase retrieval + FBP initial reconstruction (all bin levels)
# ===========================================================================

if start_step > 5:
    if rank == 0:
        logger.info('Step 5: skipped.')
    comm.Barrier()
else:
    if rank == 0:
        logger.info('Step 5: Paganin + FBP...')

    # Read theta and cshifts once (rank 0 → Bcast)
    if rank == 0:
        with h5py.File(fpath) as fid:
            theta_raw = fid['/exchange/theta'][:, 0].astype('float32')
            cshifts   = fid['/exchange/cshifts_final'][:].astype('float32')
    else:
        theta_raw = np.empty(ntheta, dtype='float32')
        cshifts   = np.empty([ntheta, ndist, 2], dtype='float32')
    comm.Bcast(theta_raw, root=0)
    comm.Bcast(cshifts,   root=0)
    theta = (-theta_raw / 180 * np.pi).astype('float32')

    def multiPaganin(data, distances, wavelength, voxelsize, delta_beta, alpha):
        """Multi-distance Paganin phase retrieval on GPU. data: [ndist, ny, nx]."""
        fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
        fy = cp.fft.fftfreq(data.shape[-2], d=voxelsize).astype('float32')
        fx, fy = cp.meshgrid(fx, fy)
        numerator   = 0
        denominator = 0
        for j in range(data.shape[0]):
            rad_freq   = cp.fft.fft2(data[j].astype('complex64'))
            taylorExp  = 1 + wavelength * distances[j] * cp.pi * delta_beta * (fx**2 + fy**2)
            numerator  += taylorExp * rad_freq
            denominator += taylorExp**2
        numerator   /= len(distances)
        denominator  = denominator / len(distances) + alpha
        phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
        phase *= delta_beta * 0.5
        return phase

    fpath_obj = fpath.replace('.h5', '_obj.h5')
    if rank == 0 and os.path.exists(fpath_obj):
        os.remove(fpath_obj)
    comm.Barrier()

    fpath_srdata = fpath.replace('.h5', '_srdata.h5')
    if rank == 0:
        if os.path.exists(fpath_srdata):
            os.remove(fpath_srdata)
        with h5py.File(fpath_srdata, 'w') as _f:
            pass
    comm.Barrier()

    for bin in range(start_level_rec, nlevels):
        n_bin         = n // (2**bin)
        nobj_bin      = nobj // (2**bin)
        voxelsize_bin = voxelsize * (2**bin)
        if rank == 0:
            logger.info(f'Step 5: bin={bin}  n_bin={n_bin}  nobj_bin={nobj_bin}  voxelsize={voxelsize_bin*1e9:.3f} nm')

        scale = 1.0 / 2**bin
        r     = (cshifts * scale).astype('float32')
        r[..., 1] += rotation_center_shift * scale
        r_gpu = cp.array(r)

        # Ref for this bin level (rank 0 → Bcast)
        if rank == 0:
            with h5py.File(fpath) as fid:
                ref = fid[f'/exchange/pref_{bin}'][:ndist].astype('float32')
        else:
            ref = np.empty([ndist, n_bin, n_bin], dtype='float32')
        comm.Bcast(ref, root=0)

        cref        = cp.array(ref)
        fwhm_ref    = 17.0 * (n_bin / 2048)
        sigma_ref   = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
        cref_smooth = cp.stack([ndimage.gaussian_filter(cref[k], sigma_ref) for k in range(ndist)])
        cl_shift = Shift(n_bin, nobj_bin, n_bin, nobj_bin)
        npad_bin = n_bin // 16
        v_bin    = cp.linspace(0, 1, npad_bin, endpoint=False)
        v_bin    = v_bin**5 * (126 - 420*v_bin + 540*v_bin**2 - 315*v_bin**3 + 70*v_bin**4)

        # --- Each rank stitches + applies Paganin for its local projections ---
        local_ntheta = len(local_ids)
        local_recPag = np.empty([local_ntheta, nobj_bin, nobj_bin], dtype='float32')

        def _stitch(fid, srdata, j):
            data_j = cp.empty([ndist, n_bin, n_bin], dtype='float32')
            for k in range(ndist):
                data_j[k] = cp.array(fid[f'/exchange/pdata{k}_{bin}'][j].astype('float32'))
            data_j_smooth = cp.stack([ndimage.gaussian_filter(data_j[k], sigma_ref) for k in range(ndist)])
            rdata = data_j_smooth / (cref_smooth + 1e-5)
            srdata.fill(0)
            for k in range(ndist - 1, -1, -1):
                shrink_jk  = shrink_nd[j, k]                      # (2,) y, x
                eff_mag_jk = float(norm_magnifications[k]) / (1 + shrink_jk)   # (2,)
                mag = cp.array(1.0 / eff_mag_jk, dtype='float32')[None]
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
                    wx = cp.ones(nobj_bin, dtype='float32')
                    wy = cp.ones(nobj_bin, dtype='float32')
                    wx[:padx0]                    = 0
                    wx[padx0:padx0+npad_bin]      = v_bin
                    wx[-padx1-npad_bin:-padx1]    = 1 - v_bin
                    wx[-padx1:]                   = 0
                    wy[:pady0]                    = 0
                    wy[pady0:pady0+npad_bin]      = v_bin
                    wy[-pady1-npad_bin:-pady1]    = 1 - v_bin
                    wy[-pady1:]                   = 0
                    w   = cp.outer(wy, wx)
                    tmp = tmp * w + srdata[k+1] * (1 - w)
                srdata[k] = tmp

        srdata = cp.zeros([ndist, nobj_bin, nobj_bin], dtype='float32')

        # --- Estimate mm and global_bg from projection 0 on rank 0, broadcast ---
        calib = np.zeros(2, dtype='float32')
        if rank == 0:
            with h5py.File(fpath) as fid:
                _stitch(fid, srdata, 0)
            pj0       = cp.array(srdata)
            calib[0]  = float(pj0[:, :32 * n_bin // 512, :32 * n_bin // 512].mean())
            pad8      = nobj_bin // 8
            pj0       = cp.pad(pj0, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
            ph0       = multiPaganin(pj0, distances * (1 + shrink_nd[0].mean(axis=-1))**2 / norm_magnifications**2, wavelength, voxelsize_bin, paganin, 0.01)
            ph0_crop  = ph0[pad8:pad8+nobj_bin, pad8:pad8+nobj_bin]
            calib[1]  = float(cp.median(ph0_crop[:16 * n_bin // 512, :16 * n_bin // 512]))
        comm.Bcast(calib, root=0)
        mm_fixed, global_bg = float(calib[0]), float(calib[1])
        if rank == 0:
            logger.info(f'step5 bin={bin}: mm={mm_fixed:.6f}  global_bg={global_bg:.6f}')

        pad8 = nobj_bin // 8
        with h5py.File(fpath_srdata, 'a', driver='mpio', comm=comm) as fid_srdata:
            srdata_ds = fid_srdata.create_dataset(
                f'/exchange/srdata_bin{bin}',
                shape=(ntheta, ndist, nobj_bin, nobj_bin),
                dtype='float32',
            )
            with h5py.File(fpath) as fid:
                for i, j in enumerate(local_ids):
                    _stitch(fid, srdata, j)
                    srdata_ds[j] = srdata.get()
                    pj  = cp.array(srdata)
                    pj  = cp.pad(pj, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
                    phase = multiPaganin(pj, distances * (1 + shrink_nd[j].mean(axis=-1))**2 / norm_magnifications**2, wavelength, voxelsize_bin, paganin, 0.01)
                    local_recPag[i] = phase[pad8:pad8+nobj_bin, pad8:pad8+nobj_bin].get()

                    if i % 100 == 0:
                        logger.info(f'step5 bin={bin}: proj {int(j):4d}/{ntheta}')

        local_recPag -= global_bg
        logger.info(f'step5 bin={bin}: rank {rank:4d}  paganin norm = {np.linalg.norm(local_recPag):.6e}')

        # --- Save Paganin projections (every 200th frame) to separate file ---
        _proj_key  = f'/exchange/proj_bin{bin}'
        fpath_proj = fpath.replace('.h5', '_proj.h5')
        n_proj_10 = len(range(0, ntheta, 10))
        if rank == 0:
            if not os.path.exists(fpath_proj):
                with h5py.File(fpath_proj, 'w') as _f:
                    pass
            else:
                with h5py.File(fpath_proj, 'a') as fid:
                    if _proj_key in fid:
                        del fid[_proj_key]
        comm.Barrier()
        with h5py.File(fpath_proj, 'a', driver='mpio', comm=comm) as fid:
            proj_ds = fid.create_dataset(_proj_key,
                                         shape=(n_proj_10, nobj_bin, nobj_bin), dtype='float32')
            for i, j in enumerate(local_ids):
                if j % 10 == 0:
                    proj_ds[j // 10] = local_recPag[i]
        logger.debug(f'step5 bin={bin}: saved {_proj_key} → {fpath_proj}')

        # --- Redistribute: theta-distributed → z-distributed via MPIClass.redist ---
        # backward: (local_ntheta, nzobj, nobj) → (ntheta, local_nzobj, nobj)
        cl_mpi5 = MPIClass(comm, nobj_bin, ntheta, nobj_bin, 'float32')
        local_nz = cl_mpi5.local_nzobj
        z_start  = cl_mpi5.st_obj
        z_end    = cl_mpi5.end_obj
        logger.debug(f'step5 bin={bin}: z-range [{z_start}:{z_end}), local_nz={local_nz}')

        psi_z = np.empty((ntheta, local_nz, nobj_bin), dtype='float32')
        cl_mpi5.redist(local_recPag, psi_z, direction='backward')
        del local_recPag

        # --- Build complex psi and run FBP on each rank for its z-range ---
        psi_z_c = np.empty((ntheta, local_nz, nobj_bin), dtype='complex64')
        psi_z_c.real[:] = psi_z
        psi_z_c.imag[:] = psi_z / paganin
        del psi_z

        rec_loc = np.zeros((local_nz, nobj_bin, nobj_bin), dtype='complex64')

        cl_tomo = Tomo(nobj_bin, nchunk, theta, mask_r=0.9)
        nbytes  = 2 * (ntheta * nchunk * nobj_bin + nchunk * nobj_bin**2) * np.dtype('complex64').itemsize
        cl      = Chunking(nbytes, nchunk)

        @cl.gpu_batch(axis_out=0, axis_inp=1, nout=1)
        def _fbp(_, rec_loc, psi_z_c):
            rec_loc[:] = cl_tomo.fbp(psi_z_c, 'ramp')

        logger.info(f'step5 bin={bin}: FBP start, local_nz={local_nz}, nobj_bin={nobj_bin}')
        _fbp(cl, rec_loc, psi_z_c)
        logger.info(f'step5 bin={bin}: FBP done')
        logger.info(f'step5 bin={bin}: rank {rank:4d}  fbp norm = {np.linalg.norm(rec_loc):.6e}')
        del psi_z_c

        paganin_tag = int(paganin) if paganin == int(paganin) else paganin
        if rank == 0 and not os.path.exists(fpath_obj):
            with h5py.File(fpath_obj, 'w') as _f:
                pass
        comm.Barrier()

        # Batch writes to stay under the 2^31-byte MPI-IO transfer limit
        _wbatch = max(1, (1 << 28) // (nobj_bin * nobj_bin * 4))
        with h5py.File(fpath_obj, 'a', driver='mpio', comm=comm) as fid:
            re_ds = fid.create_dataset(f'/exchange/obj_init_re{paganin_tag}_{bin}',
                                       shape=(nobj_bin, nobj_bin, nobj_bin), dtype='float32')
            im_ds = fid.create_dataset(f'/exchange/obj_init_imag{paganin_tag}_{bin}',
                                       shape=(nobj_bin, nobj_bin, nobj_bin), dtype='float32')
            for _i0 in range(0, local_nz, _wbatch):
                _i1 = min(_i0 + _wbatch, local_nz)
                re_ds[z_start + _i0 : z_start + _i1] = rec_loc[_i0:_i1].real
                im_ds[z_start + _i0 : z_start + _i1] = rec_loc[_i0:_i1].imag
        del rec_loc

        if rank == 0:
            logger.info(f'Step 5: bin={bin} done.')
        comm.Barrier()

    if rank == 0:
        logger.info('Step 5: done.')

comm.Barrier()
