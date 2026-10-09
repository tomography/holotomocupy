#!/usr/bin/env python
"""
Rotation-centre sweep for Y350a_largedisp -- step 5 only, one slice per
candidate centre.

`rotation_center_shift` enters steps15.py in exactly one place that matters for
the tomogram: step 5 adds it to the horizontal component of the stitching
shifts,

    r[..., 1] += rotation_center_shift * scale

and everything downstream (Paganin, FBP) follows.  Finding the right value is
therefore a one-parameter search, and it does not need a full volume per trial:
one horizontal slice through the reconstruction is enough to see the split
double-edges close up.  That is what this script produces -- for every centre in
the sweep, the SAME middle slice, as a separate TIFF.

Because only one z row survives, the cost per candidate is step 5's projection
stage without its FBP: stitch + multi-distance Paganin for all ntheta angles,
keeping row `--slice` of each.  The angle loop is split across MPI ranks exactly
as in steps15.py; the ntheta x nobj_bin sinograms are reduced to rank 0, which
back-projects all of them in one Tomo call (each candidate occupies one z slot,
and FBP treats z slices as independent) and writes the TIFFs.

The shift-independent half of the stitch -- read pdata, Gaussian-smooth, divide
by the smoothed flat -- is computed once and cached in host RAM, so a sweep of
N candidates costs roughly N x (the shift-dependent half) rather than N x (the
whole thing).  Cache size per rank is printed at startup; use --no-cache if it
does not fit.

    mpirun -n 4 ./set_affinity_gpu.sh python step5_center_sweep.py \
        config_steps15.conf --start -20 --stop 20 --step 1

There is no local_run.sh here -- this sample is set up for Polaris, and
polaris_run.sh's `python $SCRIPT $CONFIG` line passes nothing after the config,
so append the sweep flags to that line by hand, or run the mpirun above
interactively on a compute node.

Everything except the sweep itself comes from the same config file steps15.py
uses; the geometry is re-read from the step-1..4 output HDF5 rather than from
the raw EDF tree, so the script needs only `path_out` to be present.

Output (default `{path_out}/center_sweep_bin{bin}/`):

    center_120.000_r-20.000.tiff   ...   center_080.000_r+20.000.tiff
    center_sweep_bin2.tiff         all candidates in one stack, shift ascending
    center_sweep_bin2.txt          filename <-> shift table

The leading number is `--name-offset` minus the shift (100 by default, as
asked), zero-padded to a fixed width so an alphabetical image-sequence import
follows the sweep; the `r` field repeats the signed shift so a single frame is
never ambiguous.  Both numbers are in UNBINNED (bin-0) pixels, the same units as
`rotation_center_shift=` in the config -- the 1/2**bin rescaling happens inside.

Caveat for ndist > 1: step 4 writes `pdata{k}_{bin}` after an inter-distance
amplitude match whose patch means are taken at the config's centre, so those
arrays carry a weak, low-frequency dependence on it.  This sweep re-stitches
from them without redoing that match, which is why a candidate picked here is
worth confirming with one full steps15 step-5 run before the bin-0 job.  With
ndist == 1 (this dataset) step 4's matching never runs and the sweep is exact.
"""

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from mpi4py import MPI

from holotomocupy.shift import Shift
from holotomocupy.tomo import Tomo
from holotomocupy.chunking import Chunking
from holotomocupy.logger_config import logger, set_log_level
from holotomocupy.config import parse_args_steps15
from holotomocupy.utils import write_tiff


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('config', help='the same config_steps15.conf steps15.py reads')
p.add_argument('--start', type=float, default=-20.0, help='first rotation_center_shift [unbinned px]')
p.add_argument('--stop',  type=float, default= 20.0, help='last rotation_center_shift [unbinned px]')
p.add_argument('--step',  type=float, default=  1.0, help='sweep increment [unbinned px]')
p.add_argument('--bin',   type=int,   default=None,
               help='binning level to search at (default: the coarsest, nlevels-1). '
                    'Step 4 writes pdata{k}_{bin} for every bin < nlevels.')
p.add_argument('--slice', type=int,   default=None,
               help='z index of the slice, in the binned object grid (default: nobj_bin//2)')
p.add_argument('--out',   default=None, help='output directory (default: {path_out}/center_sweep_bin{bin})')
p.add_argument('--name-offset', type=float, default=100.0,
               help='the number in the filename is this minus the shift (default 100)')
p.add_argument('--nchunk', type=int, default=None, help='override nchunk from the config (FBP batching)')
p.add_argument('--no-cache', action='store_true',
               help='re-read and re-smooth pdata for every candidate instead of caching it in host RAM')
a = p.parse_args()

args = parse_args_steps15(a.config)
set_log_level(args.log_level)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

paganin = args.paganin
nchunk  = a.nchunk if a.nchunk is not None else args.nchunk
bin     = a.bin    if a.bin    is not None else args.nlevels - 1

path_out = args.path_out if args.path_out else args.path.rstrip('/') + '_rec'
fpath    = f'{path_out}/{args.pfile}.h5'
if not os.path.exists(fpath):
    raise SystemExit(f'{fpath} not found -- run steps 1-4 of steps15.py first')


# ---------------------------------------------------------------------------
# Geometry, straight out of the step-1..4 HDF5
# ---------------------------------------------------------------------------
# steps15.py rebuilds all of this from the raw EDF/H5 scan tree every run; step 1
# stored the same numbers, and step 3 stored shrink and cshifts_final, so reading
# them back keeps this script independent of `path` still being mounted.

with h5py.File(fpath, 'r') as fid:
    ndist = 0
    while f'/exchange/pdata{ndist}' in fid:
        ndist += 1
    if ndist == 0:
        raise SystemExit(f'{fpath} has no /exchange/pdata0 -- step 2 has not run')
    if f'/exchange/pdata0_{bin}' not in fid:
        raise SystemExit(f'{fpath} has no /exchange/pdata0_{bin} -- step 4 has not run '
                         f'for bin={bin} (nlevels={args.nlevels})')

    z1                      = fid['/exchange/z1'][:].astype('float64')
    energy                  = float(fid['/exchange/energy'][0])
    detector_pixelsize      = float(fid['/exchange/detector_pixelsize'][0])
    focustodetectordistance = float(fid['/exchange/focusdetectordistance'][0])
    theta_raw               = fid['/exchange/theta'][:, 0].astype('float32')
    cshifts                 = fid['/exchange/cshifts_final'][:].astype('float32')
    shrink_nd               = fid['/exchange/shrink'][:].astype('float32')
    n                       = int(fid['/exchange/pdata0'].shape[-1])

ntheta              = int(theta_raw.shape[0])
theta               = (-theta_raw / 180 * np.pi).astype('float32')
wavelength          = 1.24e-09 / energy
z2                  = focustodetectordistance - z1
magnifications      = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]
distances           = (z1 * z2) / focustodetectordistance * norm_magnifications**2
voxelsize           = float(np.abs(detector_pixelsize / magnifications[0]))
nobj = args.nobj if args.nobj is not None else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64

n_bin         = n // 2**bin
nobj_bin      = nobj // 2**bin
voxelsize_bin = voxelsize * 2**bin
scale         = 1.0 / 2**bin
zmid          = a.slice if a.slice is not None else nobj_bin // 2
if not 0 <= zmid < nobj_bin:
    raise SystemExit(f'--slice {zmid} outside [0, {nobj_bin})')

nsh    = int(round((a.stop - a.start) / a.step)) + 1
shifts = (a.start + a.step * np.arange(nsh)).astype('float64')

out_dir = a.out if a.out else f'{path_out}/center_sweep_bin{bin}'
tag     = f'center_sweep_bin{bin}'

# np.array_split hands out ascending contiguous blocks, so rank 0 owns
# projection 0 -- which is the angle step 5 measures its Paganin background on,
# and the one the `if j == 0` branch below keys off.
ids_per_rank = np.array_split(np.arange(ntheta), size)
local_ids    = ids_per_rank[rank]
local_ntheta = len(local_ids)

if rank == 0:
    os.makedirs(out_dir, exist_ok=True)
    logger.info('=' * 62)
    logger.info('  rotation-centre sweep, step 5 only')
    logger.info(f'  in                   : {fpath}')
    logger.info(f'  out                  : {out_dir}')
    logger.info(f'  bin                  : {bin}   n_bin={n_bin}  nobj_bin={nobj_bin}')
    logger.info(f'  voxel size           : {voxelsize_bin*1e9:.3f} nm')
    logger.info(f'  ndist / ntheta       : {ndist} / {ntheta}')
    logger.info(f'  paganin              : {paganin}')
    logger.info(f'  slice                : z = {zmid} of {nobj_bin}')
    logger.info(f'  sweep                : {a.start:g} .. {a.stop:g} step {a.step:g} '
                f'unbinned px  ({nsh} candidates)')
    logger.info(f'  config centre        : {args.rotation_center_shift:.6f} unbinned px'
                f'{"" if a.start <= args.rotation_center_shift <= a.stop else "   *** OUTSIDE THE SWEEP ***"}')
    logger.info(f'  n MPI ranks          : {size}')
    logger.info('=' * 62)
comm.Barrier()


# ---------------------------------------------------------------------------
# Step 5's own operators, copied so this script stays runnable on its own
# ---------------------------------------------------------------------------

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


# Same low-pass as steps 4 and 5, scaled with the bin level.
fwhm_ref  = 17.0 * (n_bin / 2048)
sigma_ref = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))

with h5py.File(fpath, 'r') as fid:
    ref = fid[f'/exchange/pref_{bin}'][:ndist].astype('float32')
cref_smooth = cp.stack([ndimage.gaussian_filter(cp.array(ref[k]), sigma_ref)
                        for k in range(ndist)])

cl_shift = Shift(n_bin, nobj_bin, n_bin, nobj_bin)
npad_bin = n_bin // 16
v_bin    = cp.linspace(0, 1, npad_bin, endpoint=False)
v_bin    = v_bin**5 * (126 - 420*v_bin + 540*v_bin**2 - 315*v_bin**3 + 70*v_bin**4)
pad8     = nobj_bin // 8


def _rdata(fid, j):
    """The shift-independent half of step 5's stitch: read, smooth, ratio."""
    data_j = cp.empty([ndist, n_bin, n_bin], dtype='float32')
    for k in range(ndist):
        data_j[k] = cp.array(fid[f'/exchange/pdata{k}_{bin}'][j].astype('float32'))
    data_j_smooth = cp.stack([ndimage.gaussian_filter(data_j[k], sigma_ref)
                              for k in range(ndist)])
    return data_j_smooth / (cref_smooth + 1e-5)


def _stitch(rdata, srdata, j, r, r_gpu):
    """The shift-dependent half: warp each distance onto the object grid and
    blend.  Line-for-line steps15.py's `_stitch` after the ratio."""
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
            wx[:padx0]                 = 0
            wx[padx0:padx0+npad_bin]   = v_bin
            wx[-padx1-npad_bin:-padx1] = 1 - v_bin
            wx[-padx1:]                = 0
            wy[:pady0]                 = 0
            wy[pady0:pady0+npad_bin]   = v_bin
            wy[-pady1-npad_bin:-pady1] = 1 - v_bin
            wy[-pady1:]                = 0
            w   = cp.outer(wy, wx)
            tmp = tmp * w + srdata[k+1] * (1 - w)
        srdata[k] = tmp


# ---------------------------------------------------------------------------
# Cache the shift-independent half
# ---------------------------------------------------------------------------

cache_bytes = local_ntheta * ndist * n_bin * n_bin * 4
cache = None
if not a.no_cache:
    logger.info(f'caching smoothed ratios: {cache_bytes/2**30:.2f} GiB per rank '
                f'({local_ntheta} angles x {ndist} dist x {n_bin}^2)')
    cache = np.empty([local_ntheta, ndist, n_bin, n_bin], dtype='float32')
    with h5py.File(fpath, 'r') as fid:
        for i, j in enumerate(local_ids):
            cache[i] = _rdata(fid, int(j)).get()
            if i % 200 == 0:
                logger.info(f'  cache {i}/{local_ntheta}')
comm.Barrier()


# ---------------------------------------------------------------------------
# Sweep: one sinogram row per candidate
# ---------------------------------------------------------------------------

sino   = np.zeros([nsh, ntheta, nobj_bin], dtype='float32')   # summed over ranks
bg     = np.zeros(nsh, dtype='float64')                       # ditto, one rank contributes
srdata = cp.zeros([ndist, nobj_bin, nobj_bin], dtype='float32')
# Paganin's distance argument is per-angle (shrink varies with theta), so build
# it once per angle inside the loop, exactly as step 5 does.
dist_base = distances / norm_magnifications**2

t_all = time.time()
with h5py.File(fpath, 'r') as fid:
    for isf, sh in enumerate(shifts):
        t0 = time.time()
        r = (cshifts * scale).astype('float32')
        r[..., 1] += sh * scale
        r_gpu = cp.array(r)

        for i, j in enumerate(local_ids):
            j = int(j)
            rdata = cp.array(cache[i]) if cache is not None else _rdata(fid, j)
            _stitch(rdata, srdata, j, r, r_gpu)
            pj    = cp.pad(srdata, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
            phase = multiPaganin(pj, dist_base * (1 + shrink_nd[j].mean(axis=-1))**2,
                                 wavelength, voxelsize_bin, paganin, 0.01)
            sino[isf, j] = phase[pad8 + zmid, pad8:pad8 + nobj_bin].get()
            if j == 0:
                # Step 5's global background: the median of a corner patch of
                # angle 0, subtracted from every projection.
                crop = phase[pad8:pad8 + nobj_bin, pad8:pad8 + nobj_bin]
                bg[isf] = float(cp.median(crop[:16 * n_bin // 512, :16 * n_bin // 512]))

        if rank == 0:
            logger.info(f'sweep {isf+1:3d}/{nsh}: shift={sh:+8.3f}  '
                        f'{time.time()-t0:6.1f} s')

comm.Allreduce(MPI.IN_PLACE, sino, op=MPI.SUM)
comm.Allreduce(MPI.IN_PLACE, bg,   op=MPI.SUM)
if rank == 0:
    logger.info(f'projections done in {time.time()-t_all:.1f} s')

del cache, srdata


# ---------------------------------------------------------------------------
# FBP -- every candidate is one independent z slot of a single Tomo call
# ---------------------------------------------------------------------------

if rank == 0:
    sino -= bg[:, None, None].astype('float32')

    psi = np.empty((ntheta, nsh, nobj_bin), dtype='complex64')
    psi.real[:] = np.ascontiguousarray(sino.transpose(1, 0, 2))
    psi.imag[:] = psi.real / paganin
    del sino

    rec     = np.zeros((nsh, nobj_bin, nobj_bin), dtype='complex64')
    cl_tomo = Tomo(nobj_bin, nchunk, theta, mask_r=0.9)
    nbytes  = 2 * (ntheta * nchunk * nobj_bin + nchunk * nobj_bin**2) * np.dtype('complex64').itemsize
    cl      = Chunking(nbytes, nchunk)

    @cl.gpu_batch(axis_out=0, axis_inp=1, nout=1)
    def _fbp(_, rec, psi):
        rec[:] = cl_tomo.fbp(psi, 'ramp')

    logger.info(f'FBP: {nsh} slices of {nobj_bin}^2')
    _fbp(cl, rec, psi)
    del psi

    # 'center_120.000_r-20.000' -- the leading key is --name-offset minus the
    # shift, at a fixed 7 characters so an alphabetical stack import follows the
    # sweep; the r field repeats the signed shift so one frame on its own is
    # still unambiguous.  Both in unbinned px.
    lines = []
    for isf, sh in enumerate(shifts):
        key   = a.name_offset - sh
        fname = f'center_{key:07.3f}_r{sh:+08.3f}'
        write_tiff(np.ascontiguousarray(rec[isf].real), f'{out_dir}/{fname}')
        lines.append(f'{fname}.tiff  rotation_center_shift={sh:+.4f}  '
                     f'{a.name_offset:g}-shift={key:.4f}')

    write_tiff(np.ascontiguousarray(rec.real), f'{out_dir}/{tag}')
    with open(f'{out_dir}/{tag}.txt', 'w') as f:
        f.write(f'# rotation-centre sweep from {fpath}\n')
        f.write(f'# bin={bin}  nobj_bin={nobj_bin}  slice z={zmid}  '
                f'voxel={voxelsize_bin*1e9:.3f} nm\n')
        f.write(f'# config rotation_center_shift = {args.rotation_center_shift:.6f}\n')
        f.write(f'# stack {tag}.tiff holds all {nsh} slices, shift ascending\n')
        f.write('\n'.join(lines) + '\n')

    logger.info(f'wrote {nsh} slices + {tag}.tiff to {out_dir}')
    logger.info(f'total {time.time()-t_all:.1f} s')

comm.Barrier()
