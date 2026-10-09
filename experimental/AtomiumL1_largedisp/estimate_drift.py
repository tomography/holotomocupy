#!/usr/bin/env python
"""
Estimate the residual sample drift during acquisition from the Paganin
projections, and fit it with polynomials in the rotation angle.

WHAT IS BEING MEASURED
----------------------
Step 3 builds `cshifts_final` from the encoders (`correct.txt`) plus rhapp, and
step 5 stitches every distance onto the object grid with those shifts before
Paganin.  Whatever misalignment survives that is what this script measures: a
slow, smooth drift of the sample relative to the grid over the ~hours of the
scan.  It is NOT the random +-300 px displacement -- that is already removed.

For AtomiumL1_largedisp there is no other handle on it.  HT has a
`correct_motion.txt` from the ESRF pipeline; this scan has no aux directory at
all, so `cshifts_final = random + rhapp` and nothing corrects the drift.

THE ESTIMATOR
-------------
Centre of mass, in x and in y, of every Paganin projection, with projection 0 as
the reference -- so both series start at 0 by construction.

The signal is NEGATIVE (Paganin returns Re(obj) = -delta, so the sample is a dip
in an otherwise flat empty-beam level), so the weights are `air - phase` clipped
at zero, with `air` the --air-pct percentile of the measured region at that
angle.  Taking the level per angle rather than globally matters: over this scan
the pedestal drifts by ~25% and the contrast changes several-fold, and a
centroid on a relative baseline is invariant to both an additive offset and a
multiplicative scale.

Three details are not optional, each fixing something that made the first
version of this script return noise:

  2-D, not two 1-D profiles.  The flat field leaves a smooth blotchy background
      over the whole frame.  In a marginal (a row- or column-sum) an air column
      still carries that background into the sum; in a 2-D weight map every air
      pixel falls below the baseline on its own and weighs nothing.  Measured:
      the 1-D version scattered twice as much on both axes.

  Fill with air, not with 'edge'.  steps15.py pads the unmeasured part of the
      object grid with `edge` VERTICALLY, which replicates the top and bottom
      rows of the measured window outwards.  When the dark sample mount crosses
      the top of that window it becomes a solid black bar over every filled row
      -- the heaviest object in the frame, moving with the random displacement.
      This script ramps to 1 instead, matching what steps15.py already does
      horizontally, so the fill reads as empty beam.

  Mask to the measured window.  Only the part of the object grid the detector
      actually saw at this angle is weighted, minus a --guard band for the
      Paganin kernel's smear inwards from the boundary.

  CAVEAT -- the sample does not fit.  The object spans about 75% of the detector
      and the stage then moves it by +-300 px, so at large displacements part of
      it is off the frame.  The script reports the fraction of sample mass
      outside the measured window at each angle; on this dataset it is 24% mean
      / 41% worst over the whole grid.  A centroid over a support that changes
      from angle to angle is biased by roughly that fraction times the object
      half-width, and the measured trend does shrink as the analysis box shrinks
      -- which a rigid drift could not do.  So read the amplitude as an order of
      magnitude, not a calibration.  --box is the lever: 60,400,60,460 drops the
      mount (a heavy dark cone cut by the frame bottom at EVERY angle, which is
      why the lost fraction never goes below ~11% without it) and brings the
      loss to 3.4% mean / 15.9% worst.  Nothing removes it entirely; that would
      need nobj > n, i.e. the nobj=2688 this scan's README argues for.

  CAVEAT on x -- a rigid object whose centre of mass is off the rotation axis has
      COM_x(theta) = x0 + A*cos(theta) + B*sin(theta) even with zero drift, and
      over a 180 deg scan a low-order polynomial will happily absorb part of that
      orbit and call it drift.  Pass --orbit to fit `polynomial + A*cos + B*sin`
      as an extra curve; if it tracks the points much better than the plain
      degree-5 polynomial, the difference is geometry.  For y the orbit terms
      should come out near zero -- a vertical rotation axis moves nothing
      vertically -- so they double as a check.

OUTPUT (default `{path_out}/drift_bin{bin}/`)
---------------------------------------------
    drift_bin2.png     4 panels: y and x, each with the measured points, the
                       degree 2/3/5 fits (and the orbit fit with --orbit), and
                       the residuals.  x axis is the rotation angle in degrees.
    drift_bin2.txt     per-angle table: theta, dy, dx and every fitted curve,
                       with the polynomial coefficients in the header.
    drift_bin2_profiles.npz   the two profile stacks the centroids came from,
                       so a different estimator can be tried later without
                       redoing the Paganin pass.

RUNNING (flags come after the config, which local_run.sh cannot pass)
---------------------------------------------------------------------
    mpirun -n 4 ./set_affinity_gpu.sh python estimate_drift.py \
        config_steps15.conf
    mpirun -n 4 ./set_affinity_gpu.sh python estimate_drift.py \
        config_steps15.conf --bin 1 --orbit

Needs steps 1-4 to have run; the geometry is read back out of
`{path_out}/{pfile}.h5`, so the raw EDF tree does not have to be mounted.
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
from holotomocupy.logger_config import logger, set_log_level
from holotomocupy.config import parse_args_steps15


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('config', help='the same config_steps15.conf steps15.py reads')
p.add_argument('--bin', type=int, default=None,
               help='binning level to measure at (default: the coarsest, nlevels-1). '
                    'Shifts are reported in UNBINNED pixels regardless.')
p.add_argument('--margin', type=int, default=None,
               help='half-width to trim off each side of the object grid before '
                    'profiling, in BINNED px (default: auto from max|cshifts_final|)')
p.add_argument('--degrees', type=int, nargs='+', default=[2, 3, 5],
               help='polynomial degrees to fit (default: 2 3 5)')
p.add_argument('--no-orbit', action='store_true',
               help='do NOT fit the A*cos(theta)+B*sin(theta) pair to x. That pair is what '
                    'an off-axis centre of mass contributes with zero drift, so it is on by '
                    'default and the reported x drift is what is LEFT after removing it. '
                    'Use this only to see the raw centroid.')
p.add_argument('--orbit-y', action='store_true',
               help='fit the orbit pair to y as well. Physically it should be zero -- a '
                    'vertical rotation axis moves nothing vertically -- so a large fitted '
                    'amplitude is a warning that something else is angle-correlated.')
p.add_argument('--air-pct', type=float, default=None,
               help='force the empty-beam level to this percentile of the measured region '
                    'instead of estimating it from the histogram mode. The mode is '
                    'parameter-free and is the default; a percentile has to guess what '
                    'fraction of the frame the sample covers.')
p.add_argument('--seg', choices=['grad', 'depth', 'quantile', 'sigma'], default='grad',
               help="what to take the centre of mass OF.  'grad' (default) weights by the gradient magnitude |grad phase|, which ignores the smooth background entirely and has no threshold to flip; --thr is then the fraction of the median gradient subtracted as a noise floor.  The other three weight by depth below an air level, and differ in where that level is put. 'depth' "
                    "(default) puts it --thr of the way from the air level down to the p1 "
                    "of the window, which is stable because the object's depth is; "
                    "'quantile' puts it at the --thr percentile, so the weighted area is "
                    "the same at every angle by construction; 'sigma' puts it --thr air "
                    "sigmas below air, which sounds principled and is not -- the air spread "
                    "is dominated by flat-field blotch and varied 3x over this scan, which "
                    "collapsed the segmented area from 51 to 4 percent of the window and "
                    "put 60 px of noise on every point.")
p.add_argument('--thr', type=float, default=None,
               help='what it means, and its default, both depend on --seg. grad: the '
                    'fraction of the median gradient subtracted as a noise floor '
                    '(default 0 -- subtracting nothing measured best on this scan, both '
                    'for the point-to-point noise and for the leak). depth: a fraction of '
                    'the air-to-p1 depth (default 0.2). quantile: the percentile itself '
                    '(default 20). sigma: a number of air sigmas (default 0.2). For the '
                    'three level rules it must be > 0: the weight map has to be exactly '
                    'zero on air, or the air area acts as a pedestal, and a pedestal\'s '
                    'centre of mass is the centre of the analysis window.')
p.add_argument('--window', choices=['common', 'per-angle'], default='common',
               help="which part of the object grid the centroid is taken over. 'common' "
                    "(default) is the INTERSECTION of the measured windows of all angles, "
                    "so every projection contributes the same piece of the object and "
                    "consecutive centroids are comparable; 'per-angle' uses whatever the "
                    "detector saw at that angle, which is a different piece each time and "
                    "measured 63 px of point-to-point jitter on this scan.")
p.add_argument('--max-lost', type=float, default=0.3,
               help='drop angles whose sample mass outside the analysis window exceeds this '
                    'fraction before fitting (default 0.3). They are still plotted, in grey.')
p.add_argument('--box', default=None, metavar='Y0,Y1,X0,X1',
               help='restrict the centroid to this box of the BINNED object grid. Use it '
                    'to drop a feature that is cut by the frame at every angle -- here the '
                    'sample mount, a heavy dark cone running off the bottom, which biases '
                    'the centroid without ever being fully measured. Default: whole grid.')
p.add_argument('--guard', type=int, default=None,
               help='band in BINNED px trimmed off the inside of the measured window '
                    'before the centroid, to keep the Paganin kernel smear from the '
                    'fill boundary out (default: the kernel width itself)')
p.add_argument('--dump', type=int, default=0,
               help='also write the first N Paganin crops of rank 0 as PNG, with the ROI '
                    'and the per-angle valid window drawn on -- for checking that the '
                    'window really is inside the measured data')
p.add_argument('--stride', type=int, default=1,
               help='use every Nth projection (default 1 = all of them)')
p.add_argument('--out', default=None, help='output directory (default: {path_out}/drift_bin{bin})')
p.add_argument('--export-correct3d', action='store_true',
               help='write the fitted drift as correct_correct3D.txt where step 3 '
                    'of steps15.py looks for it, {path}/{pfile}_/.  A PLAIN '
                    'polynomial is exported -- no orbit term is taken out first, '
                    'whatever --no-orbit/--orbit-y say for the report above.  That '
                    'is deliberate: A*cos(theta)+B*sin(theta) in x is exactly what a '
                    'rigid translation of the object projects to, so leaving it in '
                    'only re-centres the volume laterally and is tomographically '
                    'consistent either way.')
p.add_argument('--export-deg', type=int, default=5,
               help='degree of the exported polynomial (default 5)')
p.add_argument('--export-path', default=None,
               help='write the export here instead of {path}/{pfile}_/correct_correct3D.txt')
p.add_argument('--export-force', action='store_true',
               help='overwrite an existing correct_correct3D.txt (refused otherwise)')
a = p.parse_args()

args = parse_args_steps15(a.config)
set_log_level(args.log_level)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

paganin = args.paganin
bin     = a.bin if a.bin is not None else args.nlevels - 1

path_out = args.path_out if args.path_out else args.path.rstrip('/') + '_rec'
fpath    = f'{path_out}/{args.pfile}.h5'
if not os.path.exists(fpath):
    raise SystemExit(f'{fpath} not found -- run steps 1-4 of steps15.py first')


# ---------------------------------------------------------------------------
# Geometry, straight out of the step-1..4 HDF5 (same block as step5_center_sweep.py)
# ---------------------------------------------------------------------------

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
    theta_deg               = fid['/exchange/theta'][:, 0].astype('float64')
    cshifts                 = fid['/exchange/cshifts_final'][:].astype('float32')
    shrink_nd               = fid['/exchange/shrink'][:].astype('float32')
    n                       = int(fid['/exchange/pdata0'].shape[-1])

ntheta              = int(theta_deg.shape[0])
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

npad_bin = n_bin // 16
# sqrt(lambda * z * delta_beta / 4pi) is the 1/e width of the Paganin kernel;
# it is what the boundary smear reaches, and it sets the default GUARD.
paganin_kernel_px = float(np.sqrt(wavelength * distances.max() * paganin / (4 * np.pi))
                          / voxelsize_bin)
# The fixed ROI is now only a coarse pre-crop -- the per-angle measured-window
# mask below is what actually keeps fill out, and unlike a fixed crop it does not
# slice through the sample.  Default 0 = keep the whole grid.
margin = int(a.margin) if a.margin is not None else 0
half = nobj_bin // 2 - margin
if half < 16:
    raise SystemExit(f'margin {margin} leaves only {2*half} px of ROI on a {nobj_bin} grid; '
                     f'pass a smaller --margin or a finer --bin')
roi   = slice(nobj_bin // 2 - half, nobj_bin // 2 + half)
nroi  = 2 * half

AIR_PCT = a.air_pct
SEG     = a.seg
THR     = a.thr if a.thr is not None else {'grad': 0.0, 'quantile': 20.0}.get(SEG, 0.2)
ids   = np.arange(0, ntheta, a.stride)
nsamp = len(ids)
out_dir = a.out if a.out else f'{path_out}/drift_bin{bin}'

# Where --export-correct3d will put the file, resolved and checked NOW rather
# than after the 15 s of GPU work.  Every rank evaluates the same expression, so
# a refusal exits all of them together instead of deadlocking on the barrier.
export_path = ((a.export_path or f'{args.path}/{args.pfile}_/correct_correct3D.txt')
               if a.export_correct3d else None)
if export_path and os.path.exists(export_path) and not a.export_force:
    raise SystemExit(f'{export_path} already exists -- '
                     f'pass --export-force to replace it')
tag     = f'drift_bin{bin}'

ids_per_rank = np.array_split(ids, size)
local_ids    = ids_per_rank[rank]

if rank == 0:
    os.makedirs(out_dir, exist_ok=True)
    logger.info('=' * 62)
    logger.info('  acquisition drift from the Paganin projections')
    logger.info(f'  in                   : {fpath}')
    logger.info(f'  out                  : {out_dir}')
    logger.info(f'  bin                  : {bin}   n_bin={n_bin}  nobj_bin={nobj_bin}')
    logger.info(f'  voxel size           : {voxelsize_bin*1e9:.3f} nm')
    logger.info(f'  ndist / ntheta       : {ndist} / {ntheta}   (using {nsamp}, stride {a.stride})')
    logger.info(f'  paganin              : {paganin}')
    logger.info(f'  rotation_center_shift: {args.rotation_center_shift:.6f} unbinned px')
    logger.info(f'  max |cshifts_final|  : {np.abs(cshifts).max():.1f} unbinned px '
                f'= {np.abs(cshifts*scale).max():.1f} binned')
    logger.info(f'  profile ROI          : {nroi} of {nobj_bin} px (margin {margin} binned px)')
    logger.info(f'  analysis box         : {a.box if a.box else "whole grid"}')
    logger.info(f'  Paganin kernel       : {paganin_kernel_px:.1f} binned px '
                f'-> guard {a.guard if a.guard is not None else int(round(paganin_kernel_px))}')
    logger.info(f'  air level            : '
                + (f'p{AIR_PCT:g}' if AIR_PCT is not None else 'histogram mode')
                + f',  segment: {SEG} {THR:g}')
    logger.info(f'  polynomial degrees   : {a.degrees}'
                + ('' if a.no_orbit else '  + orbit on x')
                + ('  + orbit on y' if a.orbit_y else ''))
    logger.info(f'  n MPI ranks          : {size}')
    logger.info('=' * 62)
comm.Barrier()


# ---------------------------------------------------------------------------
# Step 5's operators, copied verbatim from step5_center_sweep.py
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


fwhm_ref  = 17.0 * (n_bin / 2048)
sigma_ref = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))

with h5py.File(fpath, 'r') as fid:
    ref = fid[f'/exchange/pref_{bin}'][:ndist].astype('float32')
cref_smooth = cp.stack([ndimage.gaussian_filter(cp.array(ref[k]), sigma_ref)
                        for k in range(ndist)])

cl_shift = Shift(n_bin, nobj_bin, n_bin, nobj_bin)
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
        shrink_jk  = float(shrink_nd[j, k])
        eff_mag_jk = float(norm_magnifications[k]) / (1 + shrink_jk)
        mag = cp.array(1.0 / eff_mag_jk).astype('float32')
        tmp = rdata[k].astype('complex64')
        tmp = cl_shift.curlySback(
            cp.log(tmp[None]).astype('complex64'), r_gpu[j:j+1, k], mag
        )[0].real
        tmp = cp.exp(tmp)
        padx0 = int((nobj_bin - n_bin / eff_mag_jk) / 2) - int(r[j, k, 1])
        pady0 = int((nobj_bin - n_bin / eff_mag_jk) / 2) - int(r[j, k, 0])
        padx1 = int((nobj_bin - n_bin / eff_mag_jk) / 2) + int(r[j, k, 1])
        pady1 = int((nobj_bin - n_bin / eff_mag_jk) / 2) + int(r[j, k, 0])
        padx0 = min(nobj_bin, max(0, padx0)) + 5
        pady0 = min(nobj_bin, max(0, pady0)) + 5
        padx1 = min(nobj_bin, max(0, padx1)) + 5
        pady1 = min(nobj_bin, max(0, pady1)) + 5
        # DELIBERATE DEVIATION from steps15.py, which pads vertically with 'edge'.
        # Edge replication copies the top and bottom rows of the measured window
        # outwards, and when the dark mount crosses the top of that window it is
        # smeared into a solid black bar over every filled row -- by far the
        # heaviest thing in the frame, moving with the random displacement.  For
        # a centroid that is fatal.  Ramping to 1 instead makes the fill read as
        # empty beam, exactly as the horizontal pad below already does, so it
        # weighs nothing.  (Step 5 keeps 'edge'; the bar is in the Paganin
        # obj_init there too, which is worth knowing but is not this script's
        # business -- BH only fits the measured region.)
        tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)),
                     'linear_ramp', end_values=((1, 1), (1, 1)))
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
# One Paganin pass; keep two 1-D profiles per angle
# ---------------------------------------------------------------------------
# prof_v[i] = sum over x  (length nroi, indexed by y) -> vertical registration
# prof_h[i] = sum over y  (length nroi, indexed by x) -> horizontal centre of mass

prof_v = np.zeros([nsamp, nroi], dtype='float64')
prof_h = np.zeros([nsamp, nroi], dtype='float64')
cen    = np.zeros([nsamp, 2], dtype='float64')   # (y, x) centroid, ROI coordinates
lost   = np.zeros([nsamp], dtype='float64')     # mass outside the measured window
srdata = cp.zeros([ndist, nobj_bin, nobj_bin], dtype='float32')
yy_gpu = cp.arange(nroi, dtype='float32')[:, None]
xx_gpu = cp.arange(nroi, dtype='float32')[None, :]
GUARD  = a.guard if a.guard is not None else int(round(paganin_kernel_px))
if a.box:
    _b = [int(v) for v in a.box.split(',')]
    box_mask = ((yy_gpu + roi.start >= _b[0]) & (yy_gpu + roi.start < _b[1]) &
                (xx_gpu + roi.start >= _b[2]) & (xx_gpu + roi.start < _b[3]))
else:
    box_mask = True
dist_base = distances / norm_magnifications**2

r     = (cshifts * scale).astype('float32')
r[..., 1] += args.rotation_center_shift * scale
r_gpu = cp.array(r)


def _win(j, guard=None):
    """The measured window at angle j, in ROI coordinates, eroded by `guard`.

    The pad arithmetic is `_stitch`'s, distance by distance -- each distance
    lands on the object grid at its own magnification, so with ndist > 1 they
    do not cover the same rectangle and only their intersection is measured
    everywhere.  With one distance and nobj == n the whole thing collapses to
    5 + max(0, -r) as it did before.
    """
    g = GUARD if guard is None else guard
    y0 = x0 = 0
    y1 = x1 = 0
    for k in range(ndist):
        half = int((nobj_bin - n_bin / (float(norm_magnifications[k])
                                        / (1 + float(shrink_nd[j, k])))) / 2)
        y0 = max(y0, min(nobj_bin, max(0, half - int(r[j, k, 0]))) + 5)
        x0 = max(x0, min(nobj_bin, max(0, half - int(r[j, k, 1]))) + 5)
        y1 = max(y1, min(nobj_bin, max(0, half + int(r[j, k, 0]))) + 5)
        x1 = max(x1, min(nobj_bin, max(0, half + int(r[j, k, 1]))) + 5)
    return (y0 + g - roi.start, nroi - (y1 + g - roi.start),
            x0 + g - roi.start, nroi - (x1 + g - roi.start))


# The measured window moves with the +-300 px random displacement, and the object
# is BIGGER than the window, so a per-angle window means the centroid is taken
# over a different piece of the object at every angle.  That is not a small
# effect: consecutive projections are 0.1 deg apart, so their centroids should be
# identical to well under a pixel, and with a per-angle window they scattered by
# 63 px -- the estimator noise was three times the drift being looked for.  The
# intersection of every angle's window is the same piece of object throughout,
# which is what makes the series comparable at all.  It costs field of view:
# here 300 of 512 binned px, against an object spanning ~385.
_W = np.array([_win(int(j)) for j in ids])
COMMON = (int(_W[:, 0].max()), int(_W[:, 1].min()),
          int(_W[:, 2].max()), int(_W[:, 3].min()))
if a.window == 'common':
    if COMMON[1] - COMMON[0] < 32 or COMMON[3] - COMMON[2] < 32:
        raise SystemExit(f'the windows of all angles intersect in only '
                         f'{COMMON[1]-COMMON[0]} x {COMMON[3]-COMMON[2]} px; '
                         f'use --window per-angle, or a finer --bin')
    if rank == 0:
        logger.info(f'  common window        : y [{COMMON[0]}, {COMMON[1]})  '
                    f'x [{COMMON[2]}, {COMMON[3]})  of {nroi} binned px')

t0 = time.time()
with h5py.File(fpath, 'r') as fid:
    for j in local_ids:
        j  = int(j)
        i  = int(np.searchsorted(ids, j))
        rdata = _rdata(fid, j)
        _stitch(rdata, srdata, j, r, r_gpu)
        pj    = cp.pad(srdata, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
        phase = multiPaganin(pj, dist_base * (1 + shrink_nd[j, :])**2,
                             wavelength, voxelsize_bin, paganin, 0.01)
        crop  = phase[pad8:pad8 + nobj_bin, pad8:pad8 + nobj_bin][roi, roi]
        prof_v[i] = crop.sum(axis=1).get()
        prof_h[i] = crop.sum(axis=0).get()
        # 2-D centre of mass of the NEGATIVE signal.  Both coordinates come from
        # one weight map rather than from the two marginals: the flat field
        # leaves a smooth blotchy background over the whole frame, and in a
        # marginal an air column still carries that background into the sum,
        # whereas here every air pixel is thresholded away on its own.
        # Restrict to the window that was actually MEASURED at this angle.  The
        # fill outside it is air by construction (above), so it contributes no
        # mass, but the Paganin FFT smears the boundary inwards by roughly the
        # kernel width -- GUARD keeps that smear out of the weight map.
        py0, py1, px0, px1 = COMMON if a.window == 'common' else _win(j)
        valid = ((yy_gpu >= max(0, py0)) & (yy_gpu < min(nroi, py1)) &
                 (xx_gpu >= max(0, px0)) & (xx_gpu < min(nroi, px1))) & box_mask
        # The un-eroded measured window, for the lost-mass figure only: what the
        # detector actually saw at this angle, guard and common window aside.
        my0, my1, mx0, mx1 = _win(j, guard=0)
        meas = ((yy_gpu >= max(0, my0)) & (yy_gpu < min(nroi, my1)) &
                (xx_gpu >= max(0, mx0)) & (xx_gpu < min(nroi, mx1))) & box_mask
        # The weight map MUST be exactly zero on air.  The first version used
        # w = clip(air - crop, 0) with air the 95th percentile, which gives
        # nearly every air pixel a small POSITIVE weight -- and rectified noise
        # cannot cancel, so the air area (most of the frame) acted as a uniform
        # pedestal whose own centre of mass is the centre of the measured
        # window.  That window moves with the random displacement, so a third of
        # the displacement leaked straight into the answer: measured slope
        # -0.34, corr(dx, applied dx) = -0.74.  Segmenting the sample -- a hard
        # zero outside it -- removes the pedestal, and with it the leak.
        vcrop = crop[valid]
        if AIR_PCT is not None:
            air0 = float(cp.percentile(vcrop, AIR_PCT))
        else:
            # The tallest histogram bin IS the air level: air dominates the
            # frame and the sample is all on one side of it.  Parameter-free,
            # unlike a percentile, which needs to know the sample's area
            # fraction.  Range clipped at p50/p99.5 so dust cannot stretch it.
            _lo = float(cp.percentile(vcrop, 50.0))
            _hi = float(cp.percentile(vcrop, 99.5))
            _h  = cp.histogram(vcrop, bins=256, range=(_lo, _hi))[0]
            air0 = _lo + (_hi - _lo) * (int(cp.argmax(_h)) + 0.5) / 256
        # Pixels ABOVE that level are pure air, so their half-sample MAD gives
        # the air spread (noise + flat-field blotch) without the sample biasing
        # it.  0.6745 = the half-normal median, so sig is a real sigma.
        _up = vcrop[vcrop > air0]
        sig = float(cp.median(cp.abs(_up - air0))) / 0.6745 if _up.size > 16 else 0.0
        if sig > 0:
            air = float(cp.median(vcrop[cp.abs(vcrop - air0) < 3 * sig]))
        else:
            air = float(air0)
        if SEG == 'grad':
            lvl = np.nan          # not a level-based weight; see the note below
        elif SEG == 'depth':
            lvl = air - THR * (air - float(cp.percentile(vcrop, 1.0)))
        elif SEG == 'quantile':
            lvl = float(cp.percentile(vcrop, THR))
        else:
            lvl = air - THR * sig
        # A SOFT threshold, not a hard mask: the weight falls continuously to
        # zero at `lvl`, so a pixel crossing the level contributes nothing
        # discontinuously and the centroid does not jump when the level wobbles.
        # It is still exactly zero on air, which is the part that matters.
        # Why the default weight is the GRADIENT.  A level threshold has to sit
        # somewhere in
        # the object's depth, and the frame-to-frame normalisation makes the air
        # level jump by ~0.05 and the background spread double between
        # projections 0.1 deg apart -- so a large part of the object crosses the
        # level and the segmented area swings from 4% to 51% of the window.  The
        # centroid of a set that changes that much is noise: measured 45-60 px
        # of scatter between adjacent angles, three times the drift being
        # looked for.  Dropping the threshold and weighting by air - phase does
        # not help either: the Paganin phase carries a large low-frequency
        # background (a bright halo hugging the object, dark corners) that goes
        # ABOVE the air level, so the weight changes sign, the denominator
        # nearly cancels and the centroid runs off to thousands of px.
        #
        # |grad phase| has neither problem.  A smooth background differentiates
        # away, an additive per-frame offset differentiates away entirely, and
        # there is no decision boundary for a wobbling contrast to push pixels
        # across -- the object's edges are where the gradient is, at every
        # angle.  The air's own noise leaves a small uniform pedestal, and
        # --thr subtracts a fraction of the median gradient to suppress it;
        # what is left pulls the centroid towards the centre of the analysis
        # window, which with --window common is the SAME point at every angle,
        # so it attenuates the measured motion slightly rather than injecting
        # the random displacement.  That last part is only true with
        # --window common.
        if SEG == 'grad':
            gy, gx = cp.gradient(crop)
            g      = cp.sqrt(gy * gy + gx * gx)
            floor  = float(cp.median(g[valid])) * THR
            w      = cp.clip(g - floor, 0, None) * valid
            obj    = w > 0
        else:
            obj = crop < lvl
            w   = cp.clip(lvl - crop, 0, None) * valid
        tot = float(w.sum())
        cen[i] = ((float((w * yy_gpu).sum()) / tot, float((w * xx_gpu).sum()) / tot)
                  if tot > 0 else (np.nan, np.nan))
        # How much sample the measured window misses at this angle.  This is the
        # number that decides whether the answer means anything: a centroid over
        # a support that changes from angle to angle is biased by exactly the
        # mass it drops, so if this is not small the "drift" is mostly the window
        # moving, not the sample.
        # How much of the sample the detector missed at this angle.  Deliberately
        # NOT the fraction outside `valid`: `valid` is also eroded by the guard
        # and (with --window common) shrunk to the intersection of every angle's
        # window, and neither of those is data loss -- they are the price of a
        # comparable series.  What BH has no grid for is the mass outside the
        # measured window, so that is what this reports.
        w_all   = (cp.clip(g - floor, 0, None) if SEG == 'grad'
                   else cp.clip(lvl - crop, 0, None)) * box_mask
        tot_all = float(w_all.sum())
        lost[i] = (0.0 if tot_all <= 0
                   else 1.0 - float((w_all * meas).sum()) / tot_all)
        if rank == 0 and i < a.dump:
            logger.info(f'    [{i}] th={theta_deg[j]:6.2f}  air={air:+.5f}  sig={sig:.5f}  '
                        f'depth(p1)={air-float(cp.percentile(vcrop,1)):.5f}  '
                        f'lvl={lvl:+.5f}  '
                        f'area={float((obj & valid).sum())/float(valid.sum()):.3f} of window  '
                        f'tot={tot:.1f}  cen=({cen[i][0]:.2f},{cen[i][1]:.2f})')
            _c  = phase[pad8:pad8 + nobj_bin, pad8:pad8 + nobj_bin].get()
            _lo, _hi = np.percentile(_c, [0.5, 99.5])
            _im = np.clip((_c - _lo) / (_hi - _lo), 0, 1)
            _im = np.repeat(_im[:, :, None], 3, axis=2)
            _px0 = min(nobj_bin, max(0, -int(r[j, 0, 1]))) + 5
            _py0 = min(nobj_bin, max(0, -int(r[j, 0, 0]))) + 5
            _px1 = nobj_bin - (min(nobj_bin, max(0, int(r[j, 0, 1]))) + 5)
            _py1 = nobj_bin - (min(nobj_bin, max(0, int(r[j, 0, 0]))) + 5)
            for _y in (_py0, _py1 - 1):
                _im[_y, :, :] = (1, 0, 0)          # red   = valid window at this angle
            for _x in (_px0, _px1 - 1):
                _im[:, _x, :] = (1, 0, 0)
            for _y in (roi.start, roi.stop - 1):
                _im[_y, :, :] = (0, 1, 0)          # green = the fixed profiling ROI
            for _x in (roi.start, roi.stop - 1):
                _im[:, _x, :] = (0, 1, 0)
            # blue tint = what the segmentation actually weighted
            _m = np.zeros((nobj_bin, nobj_bin), bool)
            _m[roi, roi] = (obj & valid).get()
            _im[_m] = 0.4 * _im[_m] + 0.6 * np.array([0.2, 0.4, 1.0])
            from PIL import Image
            Image.fromarray((_im * 255).astype('uint8')).save(
                f'{out_dir}/{tag}_dump{i:03d}_th{theta_deg[j]:.1f}_'
                f'ry{r[j,0,0]:+.0f}_rx{r[j,0,1]:+.0f}.png')
        if rank == 0 and i % 200 == 0:
            logger.info(f'  projections {i}/{nsamp}')

comm.Allreduce(MPI.IN_PLACE, prof_v, op=MPI.SUM)
comm.Allreduce(MPI.IN_PLACE, prof_h, op=MPI.SUM)
comm.Allreduce(MPI.IN_PLACE, cen,    op=MPI.SUM)
comm.Allreduce(MPI.IN_PLACE, lost,   op=MPI.SUM)
if rank == 0:
    logger.info(f'projections done in {time.time()-t0:.1f} s')
del srdata

# The profiles are what every number below is derived from, and they are small
# (nsamp x nroi x 2 doubles), so keep them: re-fitting or re-thinking the
# estimator afterwards then costs nothing instead of another Paganin pass.
if rank == 0:
    np.savez_compressed(f'{out_dir}/{tag}_profiles.npz',
                        prof_v=prof_v, prof_h=prof_h, cen=cen, lost=lost,
                        theta_deg=theta_deg[ids], ids=ids, bin=bin, roi=nroi,
                        cshifts=cshifts[ids])
    logger.info(f'wrote {out_dir}/{tag}_profiles.npz')

if rank != 0:
    comm.Barrier()
    raise SystemExit


# ---------------------------------------------------------------------------
# The two estimators
# ---------------------------------------------------------------------------

theta_s = theta_deg[ids]
cy, cx  = cen[:, 0], cen[:, 1]
dy = cy - cy[0]
dx = cx - cx[0]

# Report in UNBINNED pixels so the numbers can go straight into cshifts_final,
# rotation_center_shift and the configs, which are all bin-0 quantities.
dy_un = dy * 2**bin
dx_un = dx * 2**bin


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
# Polynomial.fit maps theta onto [-1, 1] internally, so a degree-5 fit over a
# 0..360 abscissa is not the ill-conditioned mess np.polyfit would give.

def fit_poly(x, y, deg):
    return np.polynomial.Polynomial.fit(x, y, deg)


def fit_orbit(x_deg, y, deg):
    """polynomial(theta) + A*cos(theta) + B*sin(theta), least squares.

    The trig pair is what an off-axis centre of mass contributes with no drift
    at all; fitting it alongside the polynomial separates the two instead of
    letting the polynomial swallow the orbit.
    """
    th = np.deg2rad(x_deg)
    xn = 2 * (x_deg - x_deg.min()) / np.ptp(x_deg) - 1     # same [-1,1] scaling
    A  = np.column_stack([xn**k for k in range(deg + 1)] + [np.cos(th), np.sin(th)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


ORBIT_X = not a.no_orbit
ORBIT_Y = a.orbit_y


def fit_drift(xn, th, y, good, deg, orbit):
    """Split a centroid series into an off-axis orbit and a polynomial drift.

    The orbit is removed FIRST and the polynomial is fitted to what is left,
    rather than fitting `poly + A cos + B sin` jointly.  Jointly is what a
    textbook would do and it is unusable here: over a 180 deg arc cos(theta) is
    almost exactly a cubic in theta, so the two blocks are collinear
    (cond = 5.9e4 at degree 5) and the split runs away -- the joint fit returned
    a "drift" of 2.7e5 px sitting on an equal and opposite orbit.  Projecting
    onto {1, cos, sin} first is well conditioned by construction and gives the
    conservative reading: the drift is what CANNOT be explained by a rigid
    object sitting off the rotation axis.

    Fitted on `good` rows only, evaluated on all of them.
    """
    if orbit:
        O  = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
        co, *_ = np.linalg.lstsq(O[good], y[good], rcond=None)
        base = O @ co
    else:
        co   = np.array([np.mean(y[good]), 0.0, 0.0])
        base = np.full_like(y, co[0])
    res = y - base
    A   = np.column_stack([xn**k for k in range(deg + 1)])
    cf, *_ = np.linalg.lstsq(A[good], res[good], rcond=None)
    drift  = A @ cf
    drift -= drift[0]
    return base + A @ cf, drift, cf, co


th_r = np.deg2rad(theta_s)
xn_s = 2 * (theta_s - theta_s.min()) / np.ptp(theta_s) - 1
# A centroid is undefined where the segmentation found nothing, and meaningless
# where most of the sample is outside the analysis window; both are excluded
# from every fit and drawn in grey.
good = ~np.isnan(dy_un) & ~np.isnan(dx_un) & (lost <= a.max_lost)
if good.sum() < 4 * (max(a.degrees) + 3):
    raise SystemExit(f'only {good.sum()} of {nsamp} angles are usable '
                     f'(NaN or lost > {a.max_lost}); loosen --max-lost or --thr')

fits = {}
for deg in a.degrees:
    fy, dry, cfy, coy = fit_drift(xn_s, th_r, dy_un, good, deg, ORBIT_Y)
    fx, drx, cfx, cox = fit_drift(xn_s, th_r, dx_un, good, deg, ORBIT_X)
    fits[f'p{deg}'] = {'y': (fy, dry, cfy, coy), 'x': (fx, drx, cfx, cox)}
names  = list(fits)
deg_hi = max(a.degrees)
orbit_x = fits[f'p{deg_hi}']['x'][3][1:] if ORBIT_X else None
orbit_y = fits[f'p{deg_hi}']['y'][3][1:] if ORBIT_Y else None

# ---------------------------------------------------------------------------
# The curve that gets exported as correct_correct3D.txt
# ---------------------------------------------------------------------------
# Computed here rather than at the write, so the figure can draw it: it is not
# any of the curves above.  A PLAIN polynomial through the raw centroid, no
# orbit term taken out, because A*cos(theta)+B*sin(theta) in x is exactly what a
# rigid translation of the object projects to -- keeping it only re-centres the
# reconstructed volume laterally, which is tomographically consistent and one
# less thing to get wrong.
#
# Fitted on the sampled good angles and evaluated on ALL ntheta of them, so a
# run with --stride still produces a complete file; theta is normalised over the
# whole scan, which makes every evaluation an interpolation.

exp_fit = {}
if a.export_correct3d:
    edeg   = a.export_deg
    th_all = np.asarray(theta_deg[:ntheta], dtype=np.float64)
    t0, tspan = float(th_all.min()), float(np.ptp(th_all))
    xn_all = 2 * (th_all - t0) / tspan - 1
    xn_fit = 2 * (theta_s - t0) / tspan - 1
    V_fit  = np.column_stack([xn_fit**k for k in range(edeg + 1)])
    V_all  = np.column_stack([xn_all**k for k in range(edeg + 1)])
    for key, meas in (('y', dy_un), ('x', dx_un)):
        cf, *_ = np.linalg.lstsq(V_fit[good], meas[good], rcond=None)
        cur    = V_all @ cf
        cur   -= cur[0]                       # projection 0 stays where it is
        exp_fit[key] = (cur, cf,
                        float(np.sqrt(np.mean((meas[good] - V_fit[good] @ cf)**2))))

# ---------------------------------------------------------------------------
# The acceptance test
# ---------------------------------------------------------------------------
# The random +-300 px displacement is KNOWN and was already taken out by step 3,
# so a correct estimator has to be uncorrelated with it.  Any correlation is the
# analysis window moving, not the sample.  This is the number that caught the
# first version of the weight map: it leaked -0.34 of the applied displacement
# into the answer, because rectified air noise put a pedestal over the whole
# measured window and a pedestal's centre of mass IS the window centre.

sy_ap = cshifts[ids, 0, 0]
sx_ap = cshifts[ids, 0, 1]
# ...but only on a scan that HAS a deliberate random sweep.  On an ordinary scan
# cshifts_final is just the encoders' record of the sample's own motion, i.e. an
# independent measurement of the very thing being estimated, and correlating
# with it is a cross-check rather than a failure.  Tell the two apart by how the
# applied shift behaves between neighbouring angles: a random sweep jumps its
# full amplitude every step, a drift barely moves.
_jump = max(float(np.diff(sy_ap).std()), float(np.diff(sx_ap).std())) / np.sqrt(2)
_amp  = max(float(sy_ap.std()), float(sx_ap.std()))
RANDOM_APPLIED = _amp > 0 and _jump > 0.5 * _amp
leak  = {}
for key, meas, s_ap, orb in (('dy', dy_un, sy_ap, ORBIT_Y),
                             ('dx', dx_un, sx_ap, ORBIT_X)):
    cols = [xn_s**k for k in range(deg_hi + 1)]
    if orb:
        cols += [np.cos(th_r), np.sin(th_r)]
    A = np.column_stack(cols + [s_ap])
    coef, *_ = np.linalg.lstsq(A[good], meas[good], rcond=None)
    _sd = s_ap[good].std()
    leak[key] = ((float(np.corrcoef(meas[good], s_ap[good])[0, 1]) if _sd > 0 else 0.0),
                 (float(coef[-1]) if _sd > 0 else 0.0),
                 float((meas[good] - (A @ coef)[good]).std()))
# Consecutive projections are 0.1 deg apart, so whatever the sample is doing it
# cannot move between them: the scatter of the first difference is the
# estimator's own noise, and it is the honest error bar on a single point.
noise = {k: float(np.diff(v[good]).std() / np.sqrt(2))
         for k, v in (('dy', dy_un), ('dx', dx_un))}

# ...and the same statistic at longer lags is the structure function, which says
# whether a polynomial is the right model at all.  If the sample only drifts
# smoothly, the curve is FLAT out to several degrees (nothing but estimator
# noise on that timescale) and only turns up where the drift itself does; a
# polynomial then captures everything there is.  If it climbs from the very
# first lag, the sample is moving at every timescale -- a random walk, not a
# drift -- and no polynomial of any degree can represent the part faster than
# the fit, so the residual rms is motion rather than noise.
_dth = float(np.median(np.diff(theta_deg[ids]))) if nsamp > 1 else 0.0
LAGS = [l for l in (1, 2, 5, 10, 20, 50, 100, 200, 500) if l < nsamp // 3]
sfun = {}
for key, v in (('dy', dy_un), ('dx', dx_un)):
    row = []
    for lag in LAGS:
        d = v[lag:] - v[:-lag]
        d = d[~np.isnan(d)]
        row.append(float(d.std() / np.sqrt(2)) if d.size > 8 else np.nan)
    sfun[key] = row


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 2, figsize=(13, 11))
# The five angle panels share an x axis; the acceptance-test scatter at [2,1] is
# in applied-shift px and must not be dragged onto the degree scale (or vice
# versa -- a plain sharex=True stretched every angle axis to +-300).
for _a in (ax[0, 0], ax[1, 0], ax[1, 1], ax[2, 0]):
    _a.sharex(ax[0, 1])
COLORS = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

for row, (key, meas, label, orb) in enumerate(
        (('y', dy_un, 'vertical  dy', ORBIT_Y),
         ('x', dx_un, 'horizontal  dx', ORBIT_X))):
    ax[row, 0].plot(theta_s[~good], meas[~good], '.', ms=3, color='0.75', alpha=0.7,
                    label=f'excluded ({int((~good).sum())})')
    ax[row, 0].plot(theta_s[good], meas[good], '.', ms=3, color='tab:blue', alpha=0.5,
                    label=f'measured  (noise {noise["d"+key]:.1f} px/point)')
    for ci, name in enumerate(names):
        full, drift, _, _ = fits[name][key]
        rms = np.sqrt(np.mean((meas[good] - full[good])**2))
        ax[row, 0].plot(theta_s, full, '-', lw=1.6, color=COLORS[ci % len(COLORS)],
                        label=f'{name}{"+orbit" if orb else ""}   rms {rms:.1f} px')
        ax[row, 1].plot(theta_s, drift, '-', lw=1.8, color=COLORS[ci % len(COLORS)],
                        label=f'{name}   ptp {np.ptp(drift):.1f} px')
    if key in exp_fit:
        cur, _, rms = exp_fit[key]
        ax[row, 0].plot(th_all, cur, '--', lw=1.8, color='k',
                        label=f'EXPORTED deg {edeg}, no orbit removed   '
                              f'rms {rms:.1f} px, ptp {np.ptp(cur):.1f} px')
        ax[row, 1].plot(th_all, cur, '--', lw=1.8, color='k',
                        label=f'EXPORTED   ptp {np.ptp(cur):.1f} px')
    ax[row, 0].set_ylabel(f'{label}  [unbinned px]')
    ax[row, 0].legend(fontsize=8, loc='best')
    ax[row, 0].grid(alpha=0.3)
    ax[row, 1].axhline(0, color='k', lw=0.6)
    ax[row, 1].set_ylabel(f'{label} DRIFT  [unbinned px]')
    ax[row, 1].legend(fontsize=8, loc='best')
    ax[row, 1].grid(alpha=0.3)
    c, sl, _ = leak[f'd{key}']
    ax[row, 0].set_title(
        f'centre of mass, rel. to projection 0'
        + (f'   (orbit amplitude {np.hypot(*fits[f"p{deg_hi}"][key][3][1:]):.1f} px)'
           if orb else '')
        + f'   leak vs applied shift: corr {c:+.2f}, slope {sl:+.3f}', fontsize=9)
    ax[row, 1].set_title('the polynomial part alone -- the orbit removed, this is the drift'
                         if orb else 'the polynomial part alone -- this is the drift',
                         fontsize=9)

ax[2, 0].plot(theta_s, 100 * lost, '-', lw=0.8, color='tab:red')
ax[2, 0].set_ylabel('sample mass outside\nthe measured window  [%]')
ax[2, 0].set_xlabel('rotation angle  [deg]')
ax[2, 0].axhline(100*a.max_lost, color='k', lw=0.8, ls='--')
ax[2, 0].set_title(f'reliability: mean {100*lost[good].mean():.1f}%, worst {100*lost[good].max():.1f}% '
                   f'-- the centroid is biased by about this times the object half-width',
                   fontsize=9)
ax[2, 0].grid(alpha=0.3)
# The acceptance test, drawn: measured centroid against the displacement that was
# already removed.  A correct estimator scatters with no slope.
ax[2, 1].plot(sx_ap[good], dx_un[good], '.', ms=2, alpha=0.4, color='tab:blue', label='dx')
ax[2, 1].plot(sy_ap[good], dy_un[good], '.', ms=2, alpha=0.4, color='tab:orange', label='dy')
_s = np.linspace(sx_ap.min(), sx_ap.max(), 2)
ax[2, 1].plot(_s, leak['dx'][1] * _s, '-', lw=1.5, color='tab:blue',
              label=f'dx slope {leak["dx"][1]:+.3f}')
ax[2, 1].plot(_s, leak['dy'][1] * _s, '-', lw=1.5, color='tab:orange',
              label=f'dy slope {leak["dy"][1]:+.3f}')
ax[2, 1].set_xlabel('applied cshifts_final  [unbinned px]')
ax[2, 1].set_ylabel('measured centroid  [unbinned px]')
ax[2, 1].set_title('acceptance test: step 3 already removed this, so a correct '
                   'estimator has slope 0' if RANDOM_APPLIED else
                   'cross-check: no deliberate sweep on this scan, so this is the '
                   "encoders' own record of the motion", fontsize=9)
ax[2, 1].legend(fontsize=8)
ax[2, 1].grid(alpha=0.3)
fig.suptitle(f'{os.path.basename(fpath)}   bin={bin}   {nsamp} of {ntheta} angles   '
             f'box {a.box if a.box else "whole grid"}   guard {GUARD}   '
             f'paganin={paganin}   air '
             + (f'p{AIR_PCT:g}' if AIR_PCT is not None else 'mode')
             + f'   segment {SEG} {THR:g}', fontsize=11)
fig.tight_layout()
fig.savefig(f'{out_dir}/{tag}.png', dpi=150)
logger.info(f'wrote {out_dir}/{tag}.png')


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

with open(f'{out_dir}/{tag}.txt', 'w') as f:
    f.write(f'# acquisition drift from {fpath}\n')
    f.write(f'# bin={bin}  nobj_bin={nobj_bin}  ROI={nroi}  voxel={voxelsize_bin*1e9:.3f} nm\n')
    f.write(f'# paganin={paganin}  rotation_center_shift={args.rotation_center_shift:.6f}\n')
    f.write(f'# air level = '
            + (f'p{AIR_PCT:g}' if AIR_PCT is not None else 'histogram mode')
            + f', sample segmented at {SEG} {THR:g}\n')
    f.write('# all shifts in UNBINNED (bin-0) pixels, relative to projection 0\n')
    f.write('# "drift" = the polynomial part of the fit'
            + (', with A*cos+B*sin removed from x' if ORBIT_X else '') + '\n')
    if orbit_x is not None:
        f.write(f'# orbit x: A*cos+B*sin = {orbit_x[0]:+.4f}, {orbit_x[1]:+.4f} px  '
                f'(amplitude {np.hypot(*orbit_x):.4f})  -- geometry, not drift\n')
    if orbit_y is not None:
        f.write(f'# orbit y: A*cos+B*sin = {orbit_y[0]:+.4f}, {orbit_y[1]:+.4f} px  '
                f'(should be ~0)\n')
    for deg in a.degrees:
        for key, lab in (('y', 'dy'), ('x', 'dx')):
            cf = fits[f'p{deg}'][key][2][:deg+1]
            f.write(f'# deg {deg}  {lab} polynomial coef in xn = 2*(theta-{theta_s.min():g})/'
                    f'{np.ptp(theta_s):g}-1, ascending = '
                    f'{np.array2string(cf, precision=6, max_line_width=10**6)}\n')
    for deg in a.degrees:
        ry = np.sqrt(np.mean((dy_un[good] - fits[f'p{deg}']['y'][0][good])**2))
        rx = np.sqrt(np.mean((dx_un[good] - fits[f'p{deg}']['x'][0][good])**2))
        f.write(f'# rms residual deg {deg}:  dy {ry:.4f} px   dx {rx:.4f} px   '
                f'drift ptp  dy {np.ptp(fits[f"p{deg}"]["y"][1]):.4f}  '
                f'dx {np.ptp(fits[f"p{deg}"]["x"][1]):.4f}\n')
    for key in ('dy', 'dx'):
        c, sl, rr = leak[key]
        f.write(f'# leak {key} vs applied shift: corr {c:+.4f}  slope {sl:+.5f}  '
                f'rms with it removed {rr:.4f} px   (both should be ~0)\n')
    f.write(f'# single-point noise from consecutive angles: dy {noise["dy"]:.3f} px  '
            f'dx {noise["dx"]:.3f} px   (0.1 deg apart, so this is the estimator, not the sample)\n')
    f.write('# structure function, rms step [px] over a lag [deg] -- flat means '
            'white noise,\n#   rising means the sample moves on that timescale and no '
            'polynomial can catch it\n')
    f.write('#   lag  ' + ' '.join(f'{l*_dth:8.1f}' for l in LAGS) + '\n')
    for key in ('dy', 'dx'):
        f.write(f'#   {key}   ' + ' '.join(f'{v:8.2f}' for v in sfun[key]) + '\n')
    f.write(f'# sample mass outside the analysis window: mean {100*lost[good].mean():.2f}%  '
            f'max {100*lost[good].max():.2f}%   ({int((~good).sum())} of {nsamp} angles '
            f'excluded: NaN or lost > {a.max_lost})\n')
    hdr = ['index', 'theta_deg', 'dy', 'dx', 'lost']
    hdr += [f'dy_{k}' for k in names] + [f'dx_{k}' for k in names]
    hdr += [f'drifty_{k}' for k in names] + [f'driftx_{k}' for k in names]
    f.write('# ' + '  '.join(f'{h:>12}' for h in hdr) + '\n')
    for i in range(nsamp):
        row = [ids[i], theta_s[i], dy_un[i], dx_un[i], lost[i]]
        row += [fits[k]['y'][0][i] for k in names] + [fits[k]['x'][0][i] for k in names]
        row += [fits[k]['y'][1][i] for k in names] + [fits[k]['x'][1][i] for k in names]
        f.write('  ' + '  '.join(f'{v:12.5f}' if isinstance(v, float) else f'{v:12d}'
                                 for v in row) + '\n')

logger.info(f'wrote {out_dir}/{tag}.txt')
logger.info(f'  mass outside the analysis window: mean {100*lost[good].mean():.2f}%  '
            f'max {100*lost[good].max():.2f}%   ({int((~good).sum())} of {nsamp} angles excluded)')
logger.info(f'  single-point noise (consecutive angles): dy {noise["dy"]:.2f} px   '
            f'dx {noise["dx"]:.2f} px')
_leak_note = ('(both must be ~0, else the window is moving, not the sample)'
              if RANDOM_APPLIED else
              '(no deliberate sweep here, so this is the encoders vs the centroid, '
              'not an error term)')
logger.info('  structure function (rms step over a lag; flat = white noise, '
            'rising = the sample is moving at that timescale)')
logger.info('    lag [deg]  ' + ' '.join(f'{l*_dth:7.1f}' for l in LAGS))
for key in ('dy', 'dx'):
    logger.info(f'    {key} [px]    '
                + ' '.join(f'{v:7.2f}' for v in sfun[key]))
for key in ('dy', 'dx'):
    c, sl, rr = leak[key]
    logger.info(f'  {"leak" if RANDOM_APPLIED else "vs encoders"} {key} vs applied '
                f'shift: corr {c:+.3f}  slope {sl:+.4f}  {_leak_note}')
if orbit_x is not None:
    logger.info(f'  orbit x: amplitude {np.hypot(*orbit_x):.2f} px  -- off-axis geometry, '
                f'removed from the drift below')
if orbit_y is not None:
    logger.info(f'  orbit y: amplitude {np.hypot(*orbit_y):.2f} px  -- should be ~0')
for deg in a.degrees:
    logger.info(f'  deg {deg}: rms dy {np.sqrt(np.mean((dy_un[good]-fits[f"p{deg}"]["y"][0][good])**2)):.2f} px'
                f'   rms dx {np.sqrt(np.mean((dx_un[good]-fits[f"p{deg}"]["x"][0][good])**2)):.2f} px'
                f'   |   DRIFT ptp  dy {np.ptp(fits[f"p{deg}"]["y"][1]):.2f} px'
                f'   dx {np.ptp(fits[f"p{deg}"]["x"][1]):.2f} px')
logger.info(f'  raw centroid peak-to-peak: dy {np.ptp(dy_un[good]):.2f} px   '
            f'dx {np.ptp(dx_un[good]):.2f} px  (unbinned)')


# ---------------------------------------------------------------------------
# Writing the export
# ---------------------------------------------------------------------------
# Step 3 does
#     raw_3d = np.loadtxt(f'{path}/{pfile}_/correct_correct3D.txt')[:ntheta, ::-1]
#     shifts_final = random_shifts + rhapp_shifts + motion_shifts \
#                    + np.tile(raw_3d[:, None], (1, ndist, 1))
# so the file is ntheta rows x 2 columns, COLUMN 0 = x and COLUMN 1 = y (the
# [::-1] flips them), in unbinned object-grid pixels, and the same row is used
# for every distance.
#
# Sign.  Measured with a constant added to r at read time (PROBE_R experiment,
# stride 50): r = (8,0) moved the binned centroid from y=254.024 to y=246.034
# and r = (0,8) moved x from 262.869 to 254.874, i.e. centroid = const - r.  So
# an excursion of +d has to be cancelled by adding +d to the shifts, and since
# r = cshifts / 2**bin the number to write is d in unbinned px -- exactly the
# dy_un / dx_un this script already reports, no rescaling.
#
# Unlike everything above, this fit has NO orbit term: the user asked for a
# plain polynomial, and it is the right thing to export anyway -- A*cos+B*sin in
# x is the projection of a rigid object translation, so keeping it just moves
# the reconstructed volume sideways.

if a.export_correct3d:
    epath = export_path
    out   = np.column_stack([exp_fit['x'][0], exp_fit['y'][0]])  # col0 = x, col1 = y

    edir = os.path.dirname(epath)
    if edir and not os.path.isdir(edir):
        os.makedirs(edir, exist_ok=True)
        logger.info(f'  created {edir}')
    with open(epath, 'w') as f:
        f.write(f'# acquisition drift for {args.pfile}, degree-{edeg} polynomial in theta\n')
        f.write(f'# from {fpath} at bin {bin}, {int(good.sum())} of {ntheta} angles used\n')
        f.write(f'# gradient-weighted centroid of the Paganin phase, seg={SEG} thr={THR:g}\n')
        f.write('# NO orbit term removed -- a plain polynomial through the raw centroid\n')
        f.write('# columns: x  y   in UNBINNED object-grid px, zero at projection 0\n')
        f.write('# read by step 3 of steps15.py as np.loadtxt(...)[:ntheta, ::-1]\n')
        for key in ('x', 'y'):
            cur, cf, rms = exp_fit[key]
            f.write(f'# {key}: ptp {np.ptp(cur):.4f} px  rms residual {rms:.4f} px  '
                    f'coef in xn = 2*(theta-{t0:g})/{tspan:g}-1, ascending = '
                    f'{np.array2string(cf, precision=6, max_line_width=10**6)}\n')
        for row in out:
            f.write(f'{row[0]:14.6f}  {row[1]:14.6f}\n')

    np.savetxt(f'{out_dir}/{tag}_correct3D.txt', out, fmt='%14.6f',
               header='x  y  [unbinned px] -- copy of what went to '
                      + epath)
    logger.info(f'wrote {epath}  ({ntheta} x 2, degree {edeg}, no orbit removed)')
    logger.info(f'  exported ptp: x {np.ptp(exp_fit["x"][0]):.2f} px   '
                f'y {np.ptp(exp_fit["y"][0]):.2f} px   '
                f'(rms residual x {exp_fit["x"][2]:.2f}  y {exp_fit["y"][2]:.2f} px)')
    logger.info(f'  copy in {out_dir}/{tag}_correct3D.txt')

comm.Barrier()
