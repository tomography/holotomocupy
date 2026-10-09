#!/usr/bin/env python
"""
Steps 1–3 — Convert EDF→HDF5, preprocess, and combine shifts (MPI + GPU parallel).

Step 1: read raw EDF projections → parallel HDF5
Step 2: outlier removal + intensity normalisation (GPU)
Step 3: combine encoder / RHAPP / motion / 3-D-correction shifts → cshifts_final

Launch with:
    mpirun -n <N> python steps_15.py steps15_Y350a.conf
"""

import sys
import re
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
from holotomocupy import esrf_meta
from holotomocupy.utils import *

# Filenames and geometry live in esrf_layout.py next to this script: the 2026
# "ewoks" scans this folder processes name their flat fields, darks and shift
# files differently from the 2025 "bliss" scans the rest of experimental/ was
# written for, and keep their geometry in an NXtomo rather than a bliss HDF5.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esrf_layout import Layout

args = parse_args_steps15(sys.argv[1])
start_step            = args.start_step
rotation_center_shift = args.rotation_center_shift
nlevels               = args.nlevels
start_level_rec       = args.start_level_rec
paganin               = args.paganin
nchunk                = args.nchunk
ref_dist              = args.ref_dist
rhapp_bin_cfg         = args.rhapp_bin
c3d_bin               = args.correct3d_bin
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

# Geometry readers moved to esrf_layout.py (read_energy, read_sx, ... for the
# 2025 bliss HDF5; read_nx_geometry for the 2026 NXtomo).  Layout.geometry()
# picks the right one and returns the same numbers either way.


# EDF headers are ASCII, open with '{', close with '}' and are padded to a
# multiple of 512 B (2048 B for this detector).  Read one generous slab rather
# than growing a bytes object 512 B at a time: the old loop scanned the whole
# 32 MB frame with a quadratic `buf += chunk` whenever motor_pos was absent,
# which turns one malformed file into a multi-minute stall inside the thread
# pool, and then returned None so the failure surfaced far from its cause.
_EDF_HEADER_MAX = 1 << 16

def find_angle(fname):
    """Rotation angle (deg) of one projection, from its EDF header.

    Looked up by MOTOR NAME rather than by column index.  The 2025 headers
    carry `motor_mne = dummy somega sx sy sz focus ...` so somega happened to
    be field 3 of the motor_pos line; the 2026 headers carry `motor_mne =
    somega` alone, where field 3 does not exist and the old code raised
    IndexError on the very first projection.
    """
    with open(fname, 'rb') as f:
        buf = f.read(_EDF_HEADER_MAX)
    end = buf.find(b'}')
    header = (buf[:end] if end >= 0 else buf).decode('latin-1')
    mne = pos = None
    for line in header.split('\n'):
        key, _, val = line.partition('=')
        key = key.strip()
        val = val.strip().rstrip(';').strip()
        if key == 'motor_mne':
            mne = val.split()
        elif key == 'motor_pos':
            pos = val.split()
    if mne is None or pos is None or 'somega' not in mne:
        raise ValueError(
            f'no somega in the EDF header of {fname} '
            f'(motor_mne={mne}, motor_pos={pos}, header '
            f'{"terminated at " + str(end) + " B" if end >= 0 else "unterminated"}, '
            f'read {len(buf)} B)')
    return float(pos[mne.index('somega')])


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

path_out = args.path_out if args.path_out else path.rstrip('/') + '_rec'
file_out = f'{pfile}.h5'

# Filenames, ntheta and geometry — one object, both directory flavours.
lay    = Layout(path.rstrip('/'), pfile)
dname0 = lay.dname(0)
ntheta = lay.ntheta
ndist  = lay.ndist

_geo                    = lay.geometry()
energy                  = _geo['energy']
detector_pixelsize      = _geo['detector_pixelsize']
focustodetectordistance = _geo['focustodetectordistance']
sx0                     = _geo['sx0']
z1                      = _geo['z1']
z2                      = _geo['z2']
magnifications          = _geo['magnifications']
norm_magnifications     = _geo['norm_magnifications']
distances               = _geo['distances']
voxelsizes              = _geo['voxelsizes']
voxelsize               = _geo['voxelsize']
wavelength              = 1.24e-09 / energy

shrink_nd          = load_shrink_from_mats(path, pfile, ndist, ntheta)  # [ntheta, ndist, 2] (y, x)
shrink             = shrink_nd[0]
eff_magnifications = norm_magnifications[:, None] / (1 + shrink)   # [ndist, 2] (y, x)

# n from actual EDF file size (images are n×n), overrideable via --n
n0, n1 = fabio.open(lay.refs(0, 0)[0]).data.shape
n = args.n if args.n is not None else n0
sty, endy = n0 // 2 - n // 2, n0 // 2 + n // 2
stx, endx = n1 // 2 - n // 2, n1 // 2 + n // 2

# Number of flat / dark frames per batch.  Both patterns are anchored on a
# digit so partially-processed dirs holding refHST*/dark.edf averages are not
# counted as extra batches; which of the two ref conventions applies is decided
# once, in Layout, from the directory flavour.
nref  = lay.nref
ndark = lay.ndark

# Stitched object size (same at all steps that use it: 4, 5), overrideable via --nobj
nobj = args.nobj if args.nobj is not None else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64

# --- Rotation axis, as ESRF reconstructed it ------------------------------
# <pfile>_/naburec/*.conf records rotation_axis_position on the <pfile>_rec_
# grid, which is the axis Peter's own reconstruction focused on -- a stated
# value, not an estimate, and the arbiter that settled AtomiumS1 after four
# focus metrics spread over 12 px.  Reading it back also means it cannot go
# stale when he re-drops the directory.
#
# It is never applied automatically: the axis is degenerate with the x column of
# cshifts_final, so it has to be typed into rotation_center_shift here AND into
# all three config_step6_bin*.conf together, and a switch that changed only one
# of them would silently desynchronise step 5 from step 6.  So this reports and
# warns, nothing more.  Note the grid is pi/2-padded here (3216, not 4096), so
# the width must be read out of <pfile>_rec_.info, never assumed.
_nabu = esrf_meta.nabu_axis(path, pfile, voxelsize)
_pyhst = esrf_meta.pyhst_axis(path, pfile, voxelsize)
if _nabu is not None and _nabu.get('rcs') is not None:
    if rank == 0:
        logger.info(f'rotation axis: nabu {os.path.basename(_nabu["source"])} -> '
                    f'{_nabu["rcs"]:+.4f} raw px  {_nabu["note"]}')
        for _c in _nabu['candidates']:
            logger.info(f'    candidate {_c["file"]:28s} axis_pos {_c["axis_pos"]:.6f}'
                        + (f'  shifts {_c["shifts"]}' if _c['shifts'] else ''))
        if _pyhst is not None and _pyhst.get('rcs') is not None:
            logger.info(f'    PyHST cross-check (1-based centre) -> {_pyhst["rcs"]:+.4f} raw px')
    if abs(_nabu['rcs'] - rotation_center_shift) > 0.5 and rank == 0:
        logger.warning(f'rotation_center_shift {rotation_center_shift:+.4f} disagrees with '
                       f'nabu {_nabu["rcs"]:+.4f} by '
                       f'{abs(_nabu["rcs"] - rotation_center_shift):.4f} raw px; retype it '
                       f'here and in all three config_step6_bin*.conf if nabu is right')
elif rank == 0:
    logger.info('rotation axis: no usable naburec/*.conf, using the configured '
                f'rotation_center_shift = {rotation_center_shift:+.4f}')

if rank == 0:
    logger.info(f'path                    = {path}')
    logger.info(f'pfile                   = {pfile}')
    logger.info(f'layout flavour          = {lay.flavour}')
    logger.info(f'ntheta                  = {ntheta}')
    logger.info(f'energy                  = {energy} keV')
    logger.info(f'detector_pixelsize      = {detector_pixelsize} m')
    logger.info(f'focustodetectordistance = {focustodetectordistance} m')
    logger.info(f'sx0                     = {sx0} m')
    logger.info(f'z1                      = {z1} m')
    logger.info(f'ndist={ndist}  n={n}  nobj={nobj}  nref={nref}  ndark={ndark}')
    logger.info(f'shrink                  = {shrink}')
    logger.debug(f'wavelength              = {wavelength} m')
    logger.debug(f'magnifications          = {magnifications}')
    logger.debug(f'voxelsizes              = {voxelsizes} m')
    for _m in lay.info_check(_geo):
        logger.warning(f'geometry vs .info: {_m}')
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
    local_fnames = [lay.proj(0, int(id)) for id in local_ids]
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
            if rank == 0:
                for id in range(nref):
                    white0_ds[k][id] = fabio.open(lay.ref(k, id, 0)).data[sty:endy, stx:endx]
                    white1_ds[k][id] = fabio.open(lay.ref(k, id, ntheta)).data[sty:endy, stx:endx]
                for id, dfile in enumerate(lay.darks(k, ndark)):
                    dark_ds[k][id]   = fabio.open(dfile).data[sty:endy, stx:endx]

            norms = np.empty(len(local_ids), dtype='float64')
            for ii, id in enumerate(local_ids):
                fname = lay.proj(k, int(id))
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

    # --- Rank 0 reads flat/dark fields, computes ref (start-of-scan whites) --
    # Only the start-of-scan flats are used. The old pipeline blended
    # ref_start -> ref_end linearly in angle; that blend tracked ring/beam drift
    # but also dragged a slow intensity ramp into the normalised projections,
    # which then fought with the per-distance amplitude matching in steps 4/5.
    if rank == 0:
        ref0_arr  = np.empty([nref,  ndist, n, n], dtype='float32')
        dark_arr  = np.empty([ndark, ndist, n, n], dtype='float32')
        with h5py.File(fpath) as fid:
            for k in range(ndist):
                ref0_arr[:, k]  = fid[f'/exchange/data_white_start{k}'][:, :n, :n]
                dark_arr[:, k]  = fid[f'/exchange/data_dark{k}'][:, :n, :n]

        dark = np.mean(dark_arr, axis=0).astype('float32')   # [ndist, n, n]

        ref = np.mean(ref0_arr, axis=0).astype('float32')    # [ndist, n, n]
        ref_gpu = cp.array(ref) - cp.array(dark)
        ref_gpu[ref_gpu < 0] = 1e-3
        ref_gpu[:] = remove_outliers(ref_gpu, radius, threshold)
        ref = ref_gpu.get()
    else:
        ref  = np.empty([ndist, n, n], dtype='float32')
        dark = np.empty([ndist, n, n], dtype='float32')

    comm.Bcast(ref,  root=0)
    comm.Bcast(dark, root=0)

    dark_gpu = cp.array(dark)

    # --- Per-projection target intensity: one fixed scalar per distance ------
    # Rank 0 measures the mean of the *first* projection at each distance and
    # broadcasts it; every projection at that distance is then rescaled to that
    # same value. Constant-in-angle, unlike the old start->end interpolation, so
    # projection-to-projection flux jitter is removed without adding a ramp.
    if rank == 0:
        mean_data_ref = np.zeros(ndist, dtype='float32')
        with h5py.File(fpath) as fid:
            for k in range(ndist):
                data = cp.array(fid[f'/exchange/data{k}'][0, :n, :n].astype('float32'))
                data -= dark_gpu[k]
                data[data < 0] = 0
                data = remove_outliers(data[None], radius, threshold)[0]
                mean_data_ref[k] = float(data.mean())

        # Cross-distance normalisation: scale everything to the distance-0 flat
        # mean, then make that mean exactly 1.
        mmr = np.mean(ref, axis=(1, 2))              # [ndist]
        mean_data_ref *= mmr[0] / mmr[:]
        ref           *= mmr[0] / mmr[:, None, None]
        mean_data_ref /= mmr[0]
        ref           /= mmr[0]
        logger.info(f'step2: mean_data_ref = {mean_data_ref}')
    else:
        mean_data_ref = np.zeros(ndist, dtype='float32')

    comm.Bcast(mean_data_ref, root=0)
    comm.Bcast(ref,           root=0)

    # --- Rank 0: write pref, delete any existing pdata ----------------------
    if rank == 0:
        with h5py.File(fpath, 'a') as fid:
            if '/exchange/pref' in fid:
                del fid['/exchange/pref']
            fid.create_dataset('/exchange/pref', data=ref)
            # Drop any legacy pref_end left by the old start/end-blend pipeline.
            if '/exchange/pref_end' in fid:
                del fid['/exchange/pref_end']
            for k in range(ndist):
                if f'/exchange/pdata{k}' in fid:
                    del fid[f'/exchange/pdata{k}']
    comm.Barrier()

    # --- All ranks write pdata in parallel ---------------------------------
    # Create output datasets first so all metadata is committed before data I/O
    with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
        for k in range(ndist):
            fid.create_dataset(f'/exchange/pdata{k}', shape=(ntheta, n, n), dtype='float32')
    comm.Barrier()

    with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
        pdata_ds = [fid[f'/exchange/pdata{k}'] for k in range(ndist)]

        for k in range(ndist):
            for j in range(local_start, local_end, chunk_size):
                end = min(j + chunk_size, local_end)

                data = cp.array(fid[f'/exchange/data{k}'][j:end, :n, :n].astype('float32'))
                data -= dark_gpu[k]
                data[data < 0] = 0
                data[:] = remove_outliers(data, radius, threshold)

                _mean = data.mean(axis=(1, 2), keepdims=True)
                _mean[_mean == 0] = 1
                data *= float(mean_data_ref[k]) / _mean
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


def driver_bin_factor(path, pfile):
    """Peter's bin_factor for this scan, read out of the octave driver.

    rhapp.mat records the inter-plane registration in the detector pixels of
    the grid HIS pipeline worked on, which is the raw grid divided by
    bin_factor -- so a rhapp entry has to be multiplied by bin_factor to become
    a raw detector pixel.  holotomo_slave.m defaults bin_factor to 1 when the
    driver does not set it, and every scan in this tree except
    ctxl_HT_4K_RD300_007p5nm leaves it unset; that is why this only started
    mattering here.

    Returns (bin_factor, one-line explanation of where it came from).
    """
    mfile = f'{path}/{pfile}_/ht_{pfile}.m'
    if not os.path.exists(mfile):
        return 1, f'no driver at {mfile}, assuming 1'
    with open(mfile, 'r', errors='replace') as f:
        hits = re.findall(r'^[^%#\n]*?\bbin_factor\s*=\s*([0-9]+)\s*;',
                          f.read(), re.M)
    if not hits:
        return 1, f'no bin_factor in {mfile} (holotomo_slave.m defaults to 1)'
    return int(hits[-1]), f'bin_factor={hits[-1]} in {mfile}'


# ===========================================================================
# STEP 3: Combine shifts
# ===========================================================================

# All work is tiny numpy — rank 0 does it, writes result, others wait.
if rank == 0:
    if start_step > 3:
        logger.info('Step 3: skipped.')
    else:
        logger.info('Step 3: combining shifts...')

        # Encoder displacements come straight from the raw scan dirs (one
        # correct.txt per distance, [ntheta, 2]) rather than from a copy stashed
        # in the HDF5 at step 1 — so step 3 can be re-run after the shift files
        # are updated, without redoing the EDF conversion.
        shifts = np.empty([ntheta, ndist, 2], dtype='float32')
        for k in range(ndist):
            # 2025 keeps this as <pfile>_k_/correct.txt, 2026 as
            # <pfile>/projections/<pfile>_000k.txt.  Same content either way:
            # the commanded random displacement in detector pixels, ntheta rows
            # (plus the retakes, which are dropped here).
            _sfile = lay.shift_source(k)
            logger.info(f'Step 3: reading shifts      from {_sfile}')
            shifts[:, k] = np.loadtxt(_sfile, dtype='float32')[:ntheta]

        # --- Encoder (random) shifts → object-plane pixels ---
        # axis 2 is (y, x) in detector pixels; swap to (row, col) and convert
        random_shifts = np.empty([ntheta, ndist, 2], dtype='float32')
        
        #NOTE: here we use norm_magnifiaciton folowing Peters code, but strictly it should be eff_magnification
        #all further corrections are found using these initial coordinates
        random_shifts[..., 0] = shifts[..., 1] / norm_magnifications
        random_shifts[..., 1] = shifts[..., 0] / norm_magnifications

        # --- What ESRF recorded about this scan, read back rather than retyped ---
        # rhapp.mat and reference_motion.mat both stamp the `pixelsize` of the
        # grid Peter's pipeline ran on, and <pfile>_rec_.info stamps the one
        # nabu reconstructed on.  Dividing either by our voxelsize gives the
        # bin factor his numbers need, with no guessing from the driver -- see
        # holotomocupy/esrf_meta.py.
        _vox = voxelsize
        _meta_rhapp_ps = esrf_meta.mat_pixelsize(f'{path}/{pfile}_/rhapp.mat')
        _meta_recinfo  = esrf_meta.rec_info(path, pfile)
        _meta_refm     = esrf_meta.reference_motion(path, pfile)

        # The reference plane is not ours to choose: rhapp is differenced
        # against it and correct_motion.txt is written for it, so ref_dist has
        # to be the plane ESRF used or every plane picks up a constant offset.
        if _meta_refm is not None and _meta_refm['ref_dist'] is not None:
            _rp = _meta_refm['ref_dist']
            if _rp == ref_dist:
                logger.info(f'Step 3: reference plane {_rp} (ESRF reference_plane='
                            f'{_meta_refm["reference_plane_1based"]}) matches ref_dist')
            else:
                logger.warning(f'Step 3: ESRF reference_plane='
                               f'{_meta_refm["reference_plane_1based"]} means ref_dist '
                               f'{_rp}, but the config says {ref_dist} -- rhapp and '
                               f'correct_motion.txt are both referenced to ESRF\'s plane, '
                               f'so this offsets every distance')
        else:
            logger.info('Step 3: no reference_motion.mat, reference plane unchecked')

        # --- RHAPP inter-plane shifts (from Peter's MATLAB pipeline) ---
        _rhapp_path = f'{path}/{pfile}_/rhapp.mat'
        if not os.path.exists(_rhapp_path):
            logger.warning(f'Step 3: rhapp.mat not found, using zeros: {_rhapp_path}')
            rhapp_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
        else:
            logger.info(f'Step 3: reading rhapp       from {_rhapp_path}')
            rhapp_raw = load_octave_text_mat(_rhapp_path, 'rhapp')
            rhapp_reordered = rhapp_raw.swapaxes(0, 2)[:ntheta]
            rhapp_reordered -= rhapp_reordered[:,ref_dist:ref_dist+1]
            avg_plane_zero = rhapp_reordered[:, 0].mean(axis=0)
            rhapp_reordered -= avg_plane_zero[np.newaxis, np.newaxis, :]
            logger.info(f'Step 3: avg_plane_zero  y={avg_plane_zero[0]:.4f} px   x={avg_plane_zero[1]:.4f} px')

            # rhapp is in the detector pixels of the BINNED grid Peter's
            # pipeline ran on, not the raw grid -- see driver_bin_factor.
            # Everything else in this function is already in raw detector
            # pixels (random_shifts comes from the .txt the beamline wrote,
            # correct_motion.txt likewise), so rhapp has to be scaled up to
            # match before the four sources are summed.
            #
            # MEASURED, not assumed.  Resampling the four planes of one
            # projection onto a common object grid and cross-correlating
            # adjacent pairs gives the true inter-plane offset directly, with
            # no shift file involved; over 24 angles the residual left after
            # removing the known random displacement was 1.79 / 1.83 / 1.98
            # times the rhapp increment for pairs 1-2 / 2-3 / 3-4 (robust
            # median, MAD 0.22 on the two well-conditioned pairs), and the
            # correlation peak sat at the x2 prediction rather than the x1
            # prediction in 69 of 72 angle-pairs.  Independently, step 6 with
            # rho[pos] free walks the plane-2 positions to 2.07x the input
            # spacing and parks there.  Both say bin_factor, which the driver
            # for this scan sets to 2.
            #
            # Left unscaled this was invisible in every 2025 folder, where the
            # driver leaves bin_factor at 1 and the rhapp offsets are under
            # 20 px anyway; this scan's reach 182 px.
            if rhapp_bin_cfg > 0:
                rhapp_bin, _why = rhapp_bin_cfg, 'rhapp_bin in the config'
            else:
                # rhapp.mat says outright what grid it is on; the driver's
                # bin_factor is only the fallback, because it is often unset.
                rhapp_bin, _why = esrf_meta.bin_from_pixelsize(_meta_rhapp_ps, _vox)
                if rhapp_bin is None:
                    rhapp_bin, _why = driver_bin_factor(path.rstrip('/'), pfile)
                else:
                    _why = f'rhapp.mat pixelsize: {_why}'
            if _meta_rhapp_ps:
                _b_mat, _n_mat = esrf_meta.bin_from_pixelsize(_meta_rhapp_ps, _vox)
                if _b_mat is not None and _b_mat != rhapp_bin:
                    logger.warning(f'Step 3: rhapp_bin={rhapp_bin} but rhapp.mat itself '
                                   f'implies {_b_mat} ({_n_mat})')
            logger.info(f'Step 3: rhapp bin factor = {rhapp_bin}  ({_why})')
            rhapp_shifts = (-rhapp_reordered * rhapp_bin).astype('float32')
            _rh_mean = rhapp_shifts.mean(axis=0)
            logger.info(f'Step 3: rhapp per-plane mean, raw detector px:  '
                        f'y={np.round(_rh_mean[:, 0], 2)}  x={np.round(_rh_mean[:, 1], 2)}')

        # --- Motion shifts (slow drift of reference plane) ---
        _motion_dname = lay.dname(ref_dist)
        _motion_path = f'{_motion_dname}/correct_motion.txt'
        if not os.path.exists(_motion_path):
            logger.warning(f'Step 3: correct_motion.txt not found, using zeros: {_motion_path}')
            motion_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
        else:
            logger.info(f'Step 3: reading motion      from {_motion_path}')
            raw_motion = np.loadtxt(_motion_path)[:ntheta, ::-1].astype('float32')
            # norm_magnifications (not eff_*) to stay consistent with the encoder
            # shifts above: both are the initial coordinates the later
            # corrections are measured against.
            motion_base   = raw_motion / norm_magnifications[ref_dist] - random_shifts[:, ref_dist]
            motion_shifts = np.tile(motion_base[:, np.newaxis], (1, ndist, 1))

            # reference_motion.mat holds ESRF's own ref_v / ref_h: the drift of
            # the reference plane in object px, i.e. exactly what motion_base
            # just reconstructed by subtracting the random displacement.  It is
            # a check, not an input -- if these disagree, the subtraction or the
            # magnification is wrong and every later shift inherits it.
            if _meta_refm is not None and _meta_refm.get('ref_v') is not None:
                _rv = np.asarray(_meta_refm['ref_v']).ravel()[:ntheta]
                _rh = np.asarray(_meta_refm['ref_h']).ravel()[:ntheta]
                if _rv.size == ntheta:
                    for _lbl, _ours, _theirs in (('vertical', motion_base[:, 0], _rv),
                                                 ('horizontal', motion_base[:, 1], _rh)):
                        _d = min(np.abs(_ours - _theirs).max(),
                                 np.abs(_ours + _theirs).max())
                        _f = 'WARN' if _d > 0.05 else 'ok'
                        logger.info(f'Step 3: motion {_lbl:10s} vs ESRF reference_motion.mat: '
                                    f'max|diff| {_d:.6f} object px  [{_f}]')
                        if _d > 0.05:
                            logger.warning(f'Step 3: motion {_lbl} disagrees with '
                                           f'reference_motion.mat by {_d:.4f} object px')

        # --- 3-D tomographic correction shifts ---
        # find_drop_file looks one level deeper when the outer path is empty:
        # a drop landing in <pfile>_/<pfile>_/ would otherwise be read as zeros.
        _c3d_path, _c3d_note = esrf_meta.find_drop_file(path, pfile, 'correct_correct3D.txt')
        if _c3d_note:
            logger.warning(f'Step 3: {_c3d_note}')
        if _c3d_path is not None:
            logger.info(f'Step 3: reading correct3D   from {_c3d_path}')
            _raw_c3d = np.loadtxt(_c3d_path)
            # Peter's files have ntheta+1 rows: his angle grid runs 0..180
            # INCLUSIVE (ANGLE_BETWEEN_PROJECTIONS in the PyHST .par times
            # TOMO_N is exactly 180 deg), so the last row is the 180 deg repeat
            # and dropping it is right.  Anything else belongs in the log.
            if _raw_c3d.shape[0] != ntheta:
                logger.warning(f'Step 3: correct3D has {_raw_c3d.shape[0]} rows for '
                               f'{ntheta} angles; using the first {ntheta}')
            raw_3d = _raw_c3d[:ntheta, ::-1].astype('float32')
            # Same unit trap as rhapp: ESRF fits correct3D with nabu on the
            # <pfile>_rec_.nx projections, whose PixelSize in <pfile>_rec_.info
            # is bin_factor times this scan's own voxel, so the file is in
            # binned px and scales.  1 (default) leaves older scans alone.
            if c3d_bin > 0:
                _c3d_bin, _c3d_why = c3d_bin, 'correct3d_bin in the config'
            else:
                _c3d_bin, _c3d_why = esrf_meta.bin_from_pixelsize(
                    _meta_recinfo['pixelsize_m'] if _meta_recinfo else None, _vox)
                if _c3d_bin is None:
                    _c3d_bin, _c3d_why = 1, 'no <pfile>_rec_.info, assuming 1'
                else:
                    _c3d_why = f'<pfile>_rec_.info: {_c3d_why}'
            if _meta_recinfo:
                _b_info, _n_info = esrf_meta.bin_from_pixelsize(_meta_recinfo['pixelsize_m'], _vox)
                if _b_info is not None and _b_info != _c3d_bin:
                    logger.warning(f'Step 3: correct3d_bin={_c3d_bin} but '
                                   f'<pfile>_rec_.info implies {_b_info} ({_n_info})')
            raw_3d *= _c3d_bin
            logger.info(f'Step 3: correct3D bin factor = {_c3d_bin}  ({_c3d_why})')
            logger.info(f'Step 3: correct3D, raw detector px:  '
                        f'y ptp {np.ptp(raw_3d[:, 0]):.3f}  mean {raw_3d[:, 0].mean():+.4f}   '
                        f'x ptp {np.ptp(raw_3d[:, 1]):.3f}  mean {raw_3d[:, 1].mean():+.4f}')
            correct3d_shifts = np.tile(raw_3d[:, np.newaxis], (1, ndist, 1))
        else:
            logger.info(f'Step 3: correct3D file not found, using zeros')
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
            ref = fid['/exchange/pref'][:, :n, :n].astype('float32')     # [ndist, n, n]
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
                # Drop legacy pref_end_{bin} from the old start/end-blend pipeline
                # (reader.read_ref averages it in when present).
                if f'/exchange/pref_end_{bin}' in fid:
                    del fid[f'/exchange/pref_end_{bin}']
                ref0 = 0.5 * (ref0[..., ::2]    + ref0[..., 1::2])
                ref0 = 0.5 * (ref0[..., ::2, :] + ref0[..., 1::2, :])
    comm.Barrier()

    # --- All ranks create output datasets collectively + process -----------
    cl_shift = Shift(n, nobj, n, nobj)
    cref     = cp.array(ref)

    # Inter-distance amplitude matching is a low-frequency operation: matching
    # the raw ratio images couples it to speckle and edge detail, which differ
    # between distances by construction. Smooth both data and flat with the same
    # Gaussian first so the ratio carries only the slow illumination trend.
    fwhm_ref    = 17.0 * (n / 2048)
    sigma_ref   = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
    cref_smooth = cp.stack([ndimage.gaussian_filter(cref[k], sigma_ref) for k in range(ndist)])

    with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
        # require_dataset: step 4 can be re-run over an existing file without
        # first deleting every pdata{k}_{bin} (a collective delete of TB-sized
        # datasets is slow and leaves the file fragmented).
        data_out = [[fid.require_dataset(f'/exchange/pdata{k}_{bin}',
                                         shape=(ntheta, n // 2**bin, n // 2**bin),
                                         dtype='float32', exact=True)
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
                    if k == 0:
                        # A single scalar mmm cannot absorb a spatially varying
                        # mismatch between the innermost distance and the rest.
                        # Measure the ratio on a 3x3 grid of patches (corners of
                        # the valid area + centre), bilinearly upsample it to the
                        # object grid and apply it to both the stitched frame and
                        # the detector-space data that step 6 will fit.
                        cs   = min(nobj // 16, (nobj - pady0 - pady1) // 2, (nobj - padx0 - padx1) // 2)
                        ch   = cs // 2
                        midy = nobj // 2
                        midx = nobj // 2
                        ys   = [pady0, midy - ch, nobj - pady1 - cs]
                        xs   = [padx0, midx - ch, nobj - padx1 - cs]
                        prev = srdata[k + 1]
                        R = cp.array([[float(prev[y:y+cs, x:x+cs].mean() /
                                             (tmp[y:y+cs, x:x+cs].mean() + 1e-10))
                                       for x in xs] for y in ys], dtype='float32')
                        ratio_map = ndimage.zoom(R, nobj / 3, order=1)
                        tmp *= ratio_map[:nobj, :nobj]
                        ratio_crop = ratio_map[pady0:nobj-pady1, padx0:nobj-padx1]
                        data[k] *= ndimage.zoom(ratio_crop,
                                                (n / ratio_crop.shape[0],
                                                 n / ratio_crop.shape[1]),
                                                order=1)[:n, :n]
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
        r = (cshifts * scale).astype('float32')
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
        # Same low-pass as step 4, scaled with the bin level.
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
                    if k == 0:
                        # 3x3 spatially varying ratio correction, as in step 4.
                        cs   = min(nobj_bin // 16,
                                   (nobj_bin - pady0 - pady1) // 2,
                                   (nobj_bin - padx0 - padx1) // 2)
                        ch   = cs // 2
                        midy = nobj_bin // 2
                        midx = nobj_bin // 2
                        ys   = [pady0, midy - ch, nobj_bin - pady1 - cs]
                        xs   = [padx0, midx - ch, nobj_bin - padx1 - cs]
                        prev = srdata[k + 1]
                        R = cp.array([[float(prev[y:y+cs, x:x+cs].mean() /
                                             (tmp[y:y+cs, x:x+cs].mean() + 1e-10))
                                       for x in xs] for y in ys], dtype='float32')
                        ratio_map = ndimage.zoom(R, nobj_bin / 3, order=1)
                        tmp *= ratio_map[:nobj_bin, :nobj_bin]
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
        # Save the stitched frames for the first n_srdata_save angles (not just
        # angle 0) so the inter-distance matching can be inspected over a range
        # of the scan. Layout is distance-major, then angle:
        #   [d0_a0 .. d0_a19, d1_a0 .. d1_a19, ...]
        n_srdata_save = min(20, ntheta)
        with h5py.File(fpath_srdata, 'a', driver='mpio', comm=comm) as fid_srdata:
            srdata_ds = fid_srdata.create_dataset(
                f'/exchange/srdata_bin{bin}',
                shape=(ndist * n_srdata_save, nobj_bin, nobj_bin),
                dtype='float32',
            )
            with h5py.File(fpath) as fid:
                for i, j in enumerate(local_ids):
                    _stitch(fid, srdata, j)
                    if j < n_srdata_save:
                        for k in range(ndist):
                            srdata_ds[k * n_srdata_save + j] = srdata[k].get()
                    pj  = cp.array(srdata)
                    pj  = cp.pad(pj, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
                    phase = multiPaganin(pj, distances * (1 + shrink_nd[j].mean(axis=-1))**2 / norm_magnifications**2, wavelength, voxelsize_bin, paganin, 0.01)
                    local_recPag[i] = phase[pad8:pad8+nobj_bin, pad8:pad8+nobj_bin].get()

                    if i % 100 == 0:
                        logger.info(f'step5 bin={bin}: proj {int(j):4d}/{ntheta}')

        local_recPag -= global_bg
        logger.info(f'step5 bin={bin}: rank {rank:4d}  paganin norm = {np.linalg.norm(local_recPag):.6e}')

        # --- Save Paganin projections (every 10th frame) to separate file ---
        _proj_key  = f'/exchange/proj_bin{bin}'
        fpath_proj = fpath.replace('.h5', '_proj.h5')
        n_proj_10  = len(range(0, ntheta, 10))
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
