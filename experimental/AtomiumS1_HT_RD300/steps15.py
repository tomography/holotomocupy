#!/usr/bin/env python
"""
Steps 1-5 — raw frames → HDF5, preprocess, shifts, binned data, Paganin+FBP.

Step 1: read the raw projections → parallel HDF5
Step 2: outlier removal + intensity normalisation (GPU)
Step 3: commanded displacement + measured RHAPP → cshifts_final
        (per-angle drift is NOT here: step 7 owns it, see its docstring)
Step 4: binned, stitched projections for every level in range(nlevels)
Step 4b: rotation axis, measured on bin-0 Paganin projections and added to the
        x column of cshifts_final (needs step 4's output, so it cannot be in
        step 3; needed by step 5, so it cannot be in step 5)
Step 5: multi-distance Paganin + FBP initial volume

Launch with:
    mpirun -n <N> python steps15.py config_steps15.conf

ONE FILE, ALL SEVEN AtomiumS1 DIRECTORIES -- the four FT (single-distance) and
the three HT (4-distance) scans run this byte-identical copy, and everything
that differs between them is in config_steps15.conf.  Keep it that way: edit
one and mirror it, do not fork.

IT READS FRAMES THROUGH THE LAYOUT, not through fabio, which is what lets one
file cover scans that arrived in different shapes.  esrf_layout.Layout handles
all three flavours -- plain EDF, EDF whose geometry comes from the `.info`
sidecar because the NXtomo had not been written yet, and an nxvds scan whose
pixels sit in raw balor HDF5 behind a virtual dataset.  Rotation angles come
from `read_angles()` below: the EDF `somega` header field on an EDF scan,
`sample/rotation_angle` on an nxvds one.  The two were checked against each
other on the 4-distance ctxl_HT scan and agree frame for frame.
"""

import sys
import logging
import h5py
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

# Filenames, FRAMES and geometry live in esrf_layout.py next to this script.
# The 2026 scans this folder processes keep their geometry in an NXtomo rather
# than a bliss HDF5, and this one keeps its pixels there too (virtual dataset
# -> RAW_DATA/<pfile>/scanNNNN/balor_*.h5); see that module's docstring.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esrf_layout import Layout

args = parse_args_steps15(sys.argv[1])
start_step            = args.start_step
rotation_center_shift = args.rotation_center_shift
nlevels               = args.nlevels
start_level_rec       = args.start_level_rec
paganin               = args.paganin

# Stride at which step 5 writes the Paganin-filtered projections to
# <pfile>_proj.h5:/exchange/proj_bin{bin}.  1 = every angle.
PROJ_SAVE_STEP        = 1
nchunk                = args.nchunk
ref_dist              = args.ref_dist
center_src            = args.center_src
set_log_level(args.log_level)

# Checked here and not inside a step, so it fires whatever start_step is: under
# center_src=measured the axis lives in cshifts_final, and steps 4, 5 and
# reader.py (step 6) each add the configured value on top of it.
if center_src == 'measured' and args.rotation_center_shift != 0.0:
    raise SystemExit(
        f'center_src=measured and rotation_center_shift='
        f'{args.rotation_center_shift:+.4f}.  Step 4b folds the measured axis '
        f'into cshifts_final, and steps 4/5/6 would add the configured value on '
        f'top, so the axis would be off by {args.rotation_center_shift:+.4f} px. '
        f'Set rotation_center_shift=0 in every config for this scan (steps15 '
        f'AND step6), or use center_src=config to type the number in instead.')

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

    Only reached on an EDF scan; the nxvds flavour has no per-frame headers and
    `read_angles` below takes the angles straight out of the NXtomo instead.

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


def read_angles(lay, ids):
    """Rotation angles (deg) of the projections `ids`, whatever the flavour.

    NXtomo `sample/rotation_angle` and the EDF `somega` header field were
    checked against each other on the 4-distance scan next door and agree
    frame for frame, so this is a drop-in replacement for the header read and
    not a second convention.
    """
    if lay.flavour == 'nxvds':
        return np.asarray(lay.angles(0), dtype='float32')[np.asarray(ids)]
    fnames = [lay.proj(0, int(i)) for i in ids]
    with ThreadPoolExecutor() as pool:
        return np.array(list(pool.map(find_angle, fnames)), dtype='float32')


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

# n from the actual detector frame size (images are n×n), overrideable via --n
n0, n1 = lay.frame_shape(0)
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
# warns, nothing more.  Since 2026-09-08 this scan HAS a <pfile>_/naburec, and
# it is the source used: rotation_axis_position = 1008.76 -> -29.48 raw px,
# which is what rotation_center_shift is set to.
#
# The PyHST <pfile>_rec_.par is logged only as a cross-check and MUST NOT be
# used here: it says 1046.598214 -> +44.20, 73.7 raw px away, because it was
# written before the alignment pass that produced naburec.  esrf_meta.nabu_axis
# is preferred over pyhst_axis for that reason, and the .par is only fallen back
# on when there is no naburec at all.  Note his grid may be pi/2-padded (the ctxl_HT scan's
# is 3216, not 4096), so the width is read from <pfile>_rec_.info, never
# assumed; here it is 2048 against our 4096, i.e. the 2x2 binning his driver
# asks for, and esrf_meta multiplies by that bin factor before reporting.
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
    # Under center_src=measured the config is 0 by design and step 4b folds the
    # measured axis into cshifts_final, so "retype it" is the wrong advice --
    # nabu becomes something to CHECK the measurement against, which step 4b
    # does once it has a number.  _esrf_rcs carries it there.
    if (abs(_nabu['rcs'] - rotation_center_shift) > 0.5 and rank == 0
            and center_src == 'config'):
        logger.warning(f'rotation_center_shift {rotation_center_shift:+.4f} disagrees with '
                       f'nabu {_nabu["rcs"]:+.4f} by '
                       f'{abs(_nabu["rcs"] - rotation_center_shift):.4f} raw px; retype it '
                       f'here and in all three config_step6_bin*.conf if nabu is right')
elif rank == 0:
    if _pyhst is not None and _pyhst.get('rcs') is not None:
        logger.info(f'rotation axis: no naburec/*.conf; PyHST '
                    f'{os.path.basename(_pyhst["source"])} axis_pos '
                    f'{_pyhst["axis_pos"]:.6f} on {_pyhst["dim1"]} px, bin '
                    f'{_pyhst["bin"]:g} -> {_pyhst["rcs"]:+.4f} raw px')
        if abs(_pyhst['rcs'] - rotation_center_shift) > 0.5 and center_src == 'config':
            logger.warning(f'rotation_center_shift {rotation_center_shift:+.4f} disagrees '
                           f'with PyHST {_pyhst["rcs"]:+.4f} by '
                           f'{abs(_pyhst["rcs"] - rotation_center_shift):.4f} raw px; '
                           'retype it here and in all three config_step6_bin*.conf '
                           'if PyHST is right')
    else:
        logger.info('rotation axis: no usable naburec/*.conf and no PyHST .par, using '
                    f'the configured rotation_center_shift = {rotation_center_shift:+.4f}')

# ESRF's own axis for this scan, or None.  Not applied -- step 4b compares the
# measured axis against it so a disagreement is visible in the log.
_esrf_rcs = None
for _src_d in (_nabu, _pyhst):
    if _src_d is not None and _src_d.get('rcs') is not None:
        _esrf_rcs = (float(_src_d['rcs']), os.path.basename(_src_d['source']))
        break


if rank == 0:
    logger.info(f'path                    = {path}')
    logger.info(f'pfile                   = {pfile}')
    logger.info(f'layout flavour          = {lay.flavour}')
    # A partial or aborted NXtomo drop demotes the flavour to `edfinfo`; say so
    # loudly, because the pixels then come from the EDF frames and the geometry
    # from the .info sidecar rather than from the NXtomo.
    # nx_relinked is {plane: path}; the name it did NOT point at is the
    # _000k convention, so rebuild it rather than carry it through.
    for _k, _now in getattr(lay, 'nx_relinked', {}).items():
        logger.info(f'plane {_k + 1} re-linked        : {os.path.basename(_now)} '
                    f'(the master .nx points here, not at '
                    f'{pfile}_{_k + 1:04d}.nx)')
    for _f in getattr(lay, 'nx_missing', []):
        logger.warning(f'NXtomo missing          : {os.path.basename(_f)}')
    for _f, _n in getattr(lay, 'nx_short', []):
        logger.warning(f'NXtomo too short        : {os.path.basename(_f)} has '
                       f'{_n} projections, TOMO_N is {lay.ntheta}')
    if getattr(lay, 'nx_missing', []) or getattr(lay, 'nx_short', []):
        logger.warning('-> reading pixels from the EDF frames instead '
                       '(flavour edfinfo); angles come from the somega header')
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
    local_theta = read_angles(lay, local_ids)

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
                white0 = lay.read_refs(k, 0, nref)
                white1 = lay.read_refs(k, ntheta, nref)
                if len(white1) == 0:
                    # An aborted scan stops before the end-of-scan flats.  Step
                    # 2 averages the two batches, so duplicating the start batch
                    # keeps that arithmetic well defined; it costs the drift the
                    # end flats would have carried, which is one more reason not
                    # to reconstruct an aborted scan.
                    logger.warning('step1: no end-of-scan flats at distance '
                                   f'{k+1}; reusing the start batch')
                    white1 = white0
                dark = lay.read_darks(k, ndark)
                for id in range(nref):
                    white0_ds[k][id] = white0[id][sty:endy, stx:endx]
                    white1_ds[k][id] = white1[id][sty:endy, stx:endx]
                for id in range(len(dark)):
                    dark_ds[k][id]   = dark[id][sty:endy, stx:endx]

            norms = np.empty(len(local_ids), dtype='float64')
            for ii, id in enumerate(local_ids):
                frame = lay.read_proj(k, int(id))[sty:endy, stx:endx]
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



# ===========================================================================
# STEP 3: Combine shifts
# ===========================================================================

# All work is tiny numpy — rank 0 does it, writes result, others wait.
def _shift_figure(path, theta, ref_dist, random_s, motion_s, rhapp_s, final_s):
    """Every term that went into cshifts_final, at the reference plane.

    Two columns, vertical and horizontal.  Top row is the terms laid over
    each other so their relative size is obvious; bottom row is the sum that
    was actually written.  The commanded displacement is plotted on its own
    twin axis in the top row because it is 300 px against the others' 1-10
    and would otherwise flatten everything into a line.

    rhapp is per-plane, so the reference plane is shown and the spread across
    the other planes is drawn as a band -- at ref_dist rhapp is zero by
    construction (it is differenced against this plane), and a non-zero band
    is the inter-plane residual it exists to carry.

    motion is per-plane too, but it is NOT a difference: on HT the four planes
    are four separate scans that drifted independently, so the band there is
    the real spread between them, not a residual.  A term that is identically
    zero is skipped, which is why the FT dirs plot motion and not rhapp.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    th = np.asarray(theta, dtype='float64')
    if th.size != final_s.shape[0]:
        th = np.arange(final_s.shape[0], dtype='float64')
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for col, comp in enumerate(('vertical (y)', 'horizontal (x)')):
        a = ax[0, col]
        tw = a.twinx()
        tw.plot(th, random_s[:, ref_dist, col], color='0.75', lw=0.8,
                label='commanded (right axis)')
        tw.set_ylabel('commanded, object px', color='0.55')
        tw.tick_params(axis='y', colors='0.55')

        for arr, lbl, c in ((rhapp_s, 'rhapp', 'tab:blue'),
                            (motion_s, 'motion', 'tab:red')):
            if not np.any(arr):
                continue
            a.plot(th, arr[:, ref_dist, col], c, lw=1.3, label=lbl)
            lo = arr[:, :, col].min(axis=1)
            hi = arr[:, :, col].max(axis=1)
            if np.ptp(hi - lo) > 0:
                a.fill_between(th, lo, hi, color=c, alpha=0.15, lw=0,
                               label=f'{lbl}, all planes')
        a.set_title(f'{comp}: terms at plane {ref_dist}')
        a.set_ylabel('object px')
        a.grid(alpha=0.3)
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2)

        # Everything EXCEPT the commanded sweep.  Plotting cshifts_final
        # itself is useless: +-300 px of commanded displacement swamps the
        # 1-30 px of correction and the panel is just noise.  The correction
        # is the part that was estimated and the part that can be wrong, so
        # that is what is drawn.  In x it sits on the axis offset, which
        # shows up as the constant the whole family is displaced by.
        b = ax[1, col]
        corr = final_s[:, :, col] - random_s[:, :, col]
        for k in range(final_s.shape[1]):
            b.plot(th, corr[:, k], lw=0.9, label=f'plane {k}')
        b.axhline(0, color='0.6', lw=0.8, ls=':')
        b.set_title(f'{comp}: cshifts_final - commanded  (all corrections)')
        b.set_xlabel('theta, deg')
        b.set_ylabel('object px')
        b.grid(alpha=0.3)
        b.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


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
            if not os.path.exists(_sfile):
                # No .txt AND no NXtomo means nxtomomill has not run yet: the
                # commanded sweep is unknown, not zero.  This is the first term
                # of shifts_final and nothing else in the tree derives it, so a
                # zero fill here would silently reconstruct the wrong geometry.
                if not os.path.exists(lay.nxfiles[k]):
                    logger.error(f'Step 3: commanded random displacement not '
                                 f'found, and neither is the NXtomo it is '
                                 f'written beside: {_sfile}')
                    logger.error('Step 3: nxtomomill has not run on this scan -- '
                                 'wait for it; there is no local substitute. '
                                 'Steps 1-2 can be run meanwhile with start_step=1.')
                    raise SystemExit(2)
                # The NXtomo is there and the .txt is not: zero displacement was
                # commanded, so ESRF wrote no file.  Only ever true of an RD000
                # scan -- anything else means a half-copied projections/ dir.
                logger.warning(f'Step 3: no commanded displacement at plane {k}, '
                               f'using zeros -- expected only on an RD000 scan, '
                               f'check the scan if not: {_sfile}')
                shifts[:, k] = 0
                continue
            logger.info(f'Step 3: reading shifts      from {_sfile}')
            shifts[:, k] = np.loadtxt(_sfile, dtype='float32')[:ntheta]

        # --- Encoder (random) shifts → object-plane pixels ---
        # axis 2 is (y, x) in detector pixels; swap to (row, col) and convert
        random_shifts = np.empty([ntheta, ndist, 2], dtype='float32')
        
        #NOTE: here we use norm_magnifiaciton folowing Peters code, but strictly it should be eff_magnification
        #all further corrections are found using these initial coordinates
        random_shifts[..., 0] = shifts[..., 1] / norm_magnifications
        random_shifts[..., 1] = shifts[..., 0] / norm_magnifications

        # --- MOTION: sample drift, measured from the post-scan retakes ---
        # Every scan writes ntheta+3 frames; the last three re-take omega
        # 180/90/0 right after the scan ends, so correlating each against its
        # in-scan twin says how far the sample drifted.  Three points, fitted
        # with a quadratic and mean removed -- see estimate_quali_motion.py.
        #
        # Unlike rhapp this is not a plane-to-plane difference, so it is real
        # at ndist=1 too: FT_RD300 drifts ~113 object px.
        #
        # Switched off by default (motion_src=none): the estimator is only
        # trustworthy where it has been checked against quali.mat, and with
        # 'none' step 7 owns the whole drift, as it did before this existed.
        _motion_meas = f'{path_out}/measured/motion_measured.npy'
        if args.motion_src == 'none':
            logger.info('Step 3: motion_src=none, no step-3 drift term -- '
                        'step 7 owns the drift')
            motion_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
        else:
            # Always re-measure: a cached file from a different config is
            # indistinguishable from a fresh one and silently wrong.
            import estimate_quali_motion
            logger.info('Step 3: measuring sample drift from the post-scan retakes')
            estimate_quali_motion.write_measured(lay, _motion_meas,
                                                 log=logger.info)
            if not os.path.exists(_motion_meas):
                raise SystemExit(f'Step 3: the motion estimator produced no '
                                 f'{_motion_meas}')
            logger.info(f'Step 3: reading motion      from {_motion_meas} (measured)')
            motion_shifts = np.load(_motion_meas).astype('float32')
            if motion_shifts.shape != (ntheta, ndist, 2):
                raise SystemExit(f'Step 3: {_motion_meas} has shape '
                                 f'{motion_shifts.shape}, expected {(ntheta, ndist, 2)}')
            _mo_ptp = np.ptp(motion_shifts, axis=0)
            logger.info(f'Step 3: motion per-plane ptp, object px:  '
                        f'y={np.round(_mo_ptp[:, 0], 2)}  x={np.round(_mo_ptp[:, 1], 2)}')

        # --- RHAPP: the inter-plane residual, measured from our own frames ---
        # rhapp is a DIFFERENCE between planes, so at ndist=1 it is identically
        # zero and there is nothing to measure.  Above that it is 182 px on this
        # sample and dropping it is not survivable, so there is no knob: the
        # estimator runs, or step 3 stops.
        #
        # ESRF's rhapp.mat is no longer read here.  It only exists for the scans
        # Peter's pipeline has been run on, it is on his binned grid, and the
        # estimator was validated against it at corr 0.875-0.998 -- that
        # comparison now lives in `estimate_rhapp.py --validate`, where it
        # belongs, instead of being a second code path in the pipeline.
        # Separate file per motion_src: rhapp is measured with the drift
        # already undone, so the two settings give different answers and must
        # not overwrite each other's saved copy.
        _rhapp_meas = (f'{path_out}/measured/rhapp_measured.npy'
                       if args.motion_src == 'none' else
                       f'{path_out}/measured/rhapp_measured_{args.motion_src}.npy')
        if ndist == 1:
            logger.info('Step 3: ndist=1, rhapp is zero by construction')
            rhapp_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
        else:
            # Rank 0 only -- it is one serial pass over a sparse sample of
            # angles.
            #
            # NO comm.Barrier() HERE.  Every line of step 3 is already inside
            # `if rank == 0:`, so a barrier in this branch is one collective
            # that only rank 0 ever calls: it pairs with the other ranks'
            # end-of-step-3 barrier, leaves rank 0 one call out of phase for
            # the rest of the run, and rank 0 then meets step 4's Bcast with a
            # Barrier.  The end-of-step-3 barrier at the bottom of this block
            # is the one that makes the others wait, and it is enough.
            # Always re-measure: rhapp is measured with the drift already
            # undone, so a cache from a different motion_src is silently wrong.
            import estimate_rhapp
            logger.info('Step 3: measuring inter-plane residual from the frames')
            estimate_rhapp.write_measured(lay, ref_dist, _rhapp_meas,
                                          motion=motion_shifts)
            # Hard fail, not zeros: silently dropping a 182 px term is exactly
            # what this branch exists to prevent.
            if not os.path.exists(_rhapp_meas):
                raise SystemExit(f'Step 3: the rhapp estimator produced no '
                                 f'{_rhapp_meas}')
            logger.info(f'Step 3: reading rhapp       from {_rhapp_meas} (measured)')
            rhapp_shifts = np.load(_rhapp_meas).astype('float32')
            if rhapp_shifts.shape != (ntheta, ndist, 2):
                raise SystemExit(f'Step 3: {_rhapp_meas} has shape '
                                 f'{rhapp_shifts.shape}, expected {(ntheta, ndist, 2)}')
            # The estimator correlates magnification-corrected frames, so this
            # is already object-grid px referenced to ref_dist -- the same grid
            # random_shifts lands on above.  No bin factor, no ref differencing.
            _rh_mean = rhapp_shifts.mean(axis=0)
            logger.info(f'Step 3: rhapp per-plane mean, object px:  '
                        f'y={np.round(_rh_mean[:, 0], 2)}  x={np.round(_rh_mean[:, 1], 2)}')

        # --- Sum all sources and save ---
        # Three terms, and that is the whole model -- two with motion_src=none,
        # where the drift term is identically zero.  ESRF's correct_motion.txt
        # and correct_correct3D.txt are both gone; the drift is measured here
        # from the retakes instead of read from his files, and step 7 still
        # owns whatever is left over -- pass 1 reconstructs, step 7 fits the
        # residual drift from that volume into correct_correct3D_extra.txt, and
        # step 6 adds it on pass 2.  The motion term is a mean-removed
        # quadratic, so it contributes no net translation and does not fight
        # the step-4b axis.
        shifts_final = random_shifts + motion_shifts + rhapp_shifts

        # THE ROTATION AXIS IS NOT A STEP-3 TERM.  It is measured in step 4b,
        # after step 4 and before step 5's Paganin, and added to the horizontal
        # column of cshifts_final there -- see estimate_axis_paganin.py.  It has
        # to wait: the axis is a mirror symmetry of the projected object, so it
        # can only be measured on PHASE, and phase needs step 4's
        # amplitude-corrected pdata{k}_0.  Step 3 therefore leaves the x column
        # axis-free, which is also what makes step 4b's measurement the whole
        # axis on a fresh run and a near-zero refinement on a re-run.

        with h5py.File(fpath, 'a') as fid:
            if '/exchange/cshifts_final' in fid:
                del fid['/exchange/cshifts_final']
            fid.create_dataset('/exchange/cshifts_final', data=shifts_final)
            if '/exchange/shrink' in fid:
                del fid['/exchange/shrink']
            fid.create_dataset('/exchange/shrink', data=shrink_nd)

        # --- What went into the sum, as a picture ---------------------------
        # The log prints a ptp per term, which says how big each one is but
        # not what it looks like: a drift that locked onto the static
        # background and a real one have similar ptp and completely different
        # shapes.  Written next to the config, not into path_out, so it sits
        # with the run that produced it.
        try:
            with h5py.File(fpath) as fid:
                _th = fid['/exchange/theta'][:, 0].astype('float32')
            _figpath = f'{os.path.dirname(os.path.abspath(sys.argv[1]))}/shifts.png'
            _shift_figure(_figpath, _th, ref_dist,
                          random_shifts, motion_shifts, rhapp_shifts,
                          shifts_final)
            logger.info(f'Step 3: wrote {_figpath}')
        except Exception as _e:                      # never lose a run to a plot
            logger.warning(f'Step 3: could not write shifts.png: {_e}')

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
# STEP 4b: Rotation axis, measured on bin-0 Paganin projections
# ===========================================================================
# Between step 4 and step 5 because it needs both halves: step 4's
# amplitude-corrected pdata{k}_0, and step 5's stitch + Paganin.  Measured on
# PHASE and never on frames -- the axis is a mirror symmetry of the projected
# object, and an ID16A hologram at 4.5 nm is mostly Fresnel fringes, which
# belong to the propagation and do not mirror.  See estimate_axis_paganin.py.
#
# The answer goes into the x column of cshifts_final, the one place the
# horizontal shift lives, so steps 5 and 6 pick it up by re-reading the file
# and rotation_center_shift stays 0 everywhere.
#
# NOTHING IS CACHED.  Each run measures what is LEFT OVER after the shifts
# cshifts_final currently holds, so a re-run measures ~0 and adds ~0.  Caching
# and re-adding would double-count; caching and skipping would miss a step-3
# re-run that wiped the axis out again.  Measuring the residual every time is
# the only form that is right in both cases.
#
# Rank 0 only: a handful of angles, and step 5 re-reads the file anyway.

if start_step > 5 or center_src != 'measured':
    if rank == 0:
        logger.info(f'Step 4b: axis measurement skipped (start_step={start_step}, '
                    f'center_src={center_src}, rotation_center_shift='
                    f'{rotation_center_shift:+.4f} applied downstream).')
else:
    if rank == 0:
        import estimate_axis_paganin
        logger.info('Step 4b: measuring rotation axis from opposed pairs of '
                    'bin-0 Paganin projections...')
        with h5py.File(fpath) as fid:
            _ref0 = fid['/exchange/pref_0'][:ndist].astype('float32')
            _sh   = fid['/exchange/cshifts_final'][:].astype('float32')
            # Real angles, so the "pair is not 180 deg apart" warning means
            # something on a scan that is not a clean 0-180.
            _thax = fid['/exchange/theta'][:, 0].astype('float64')

        def _measure_axis(shifts):
            return estimate_axis_paganin.measure_axis(
                estimate_axis_paganin.build_phase_fn(
                    fpath, _ref0, shifts, shrink_nd, n, nobj, ndist,
                    norm_magnifications, distances, wavelength, voxelsize,
                    paganin),
                ntheta, nobj, theta=_thax, log=logger.info)

        _rcs = _measure_axis(_sh)['center']
        _sh[..., 1] += _rcs
        with h5py.File(fpath, 'a') as fid:
            _ds = fid['/exchange/cshifts_final']
            _ds[...] = _sh
            # Running total, so a refinement run's "+0.03" can be read against
            # the full axis rather than mistaken for it.  Reset whenever step 3
            # recreates the dataset, which is correct: it wiped the axis too.
            _ds.attrs['rotation_axis'] = \
                float(_ds.attrs.get('rotation_axis', 0.0)) + _rcs
            _tot = float(_ds.attrs['rotation_axis'])
        os.makedirs(f'{path_out}/measured', exist_ok=True)
        np.save(f'{path_out}/measured/center_measured.npy',
                np.array([_tot], dtype='float32'))
        logger.info(f'Step 4b: rotation axis {_rcs:+.4f} px added to '
                    f'cshifts_final (running total {_tot:+.4f} px; config '
                    f'rotation_center_shift stays 0)')

        if _esrf_rcs is not None:
            # INFO, never a warning: the two numbers are not in the same frame.
            # ESRF reconstructed on his own reference plane, and the object
            # pixel size follows norm_magnifications[ref_dist], so the same
            # physical axis takes a different value per plane; he also folds a
            # motion correction into his geometry, which his axis absorbs part
            # of.  Several px of disagreement is expected and is not evidence
            # that either side is wrong.  The check that does mean something is
            # the verify residual below -- self-consistent, no outside number.
            logger.info(f'Step 4b: measured {_tot:+.4f} vs ESRF '
                        f'{_esrf_rcs[0]:+.4f} ({_esrf_rcs[1]}), differ by '
                        f'{abs(_esrf_rcs[0] - _tot):.4f} px -- different ref '
                        f'plane and motion convention, so magnitude check only')

        # Re-measure with the axis folded in.  The pairs should now overlay, so
        # a residual much above a pixel means what was taken out was not a rigid
        # offset and the number should not be trusted.
        if estimate_axis_paganin.VERIFY:
            _res = _measure_axis(_sh)['center']
            _m = (f'Step 4b: axis verify -- residual {_res:+.4f} px after '
                  f'folding in {_rcs:+.4f}')
            (logger.info if abs(_res) <= 1.0 else logger.warning)(_m)

comm.Barrier()


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

        # --- Save Paganin projections to a separate file --------------------
        # PROJ_SAVE_STEP = 1 keeps every angle (67 GB at bin 2, 269 GB at
        # bin 1, 1.1 TB at bin 0); raise it to thin the stack out.
        _proj_key  = f'/exchange/proj_bin{bin}'
        fpath_proj = fpath.replace('.h5', '_proj.h5')
        n_proj_save = len(range(0, ntheta, PROJ_SAVE_STEP))
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
                                         shape=(n_proj_save, nobj_bin, nobj_bin), dtype='float32')
            for i, j in enumerate(local_ids):
                if j % PROJ_SAVE_STEP == 0:
                    proj_ds[j // PROJ_SAVE_STEP] = local_recPag[i]
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
