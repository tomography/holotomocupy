"""Shared pieces for the random-displacement / single-distance study.

Everything here is taken from ../holotomo3d/test.py (same phantom, same ID16A probe,
same acquisition geometry) so that the study is directly comparable to the
holotomo3d reference test.  What changes between runs is the amplitude of the
random sample displacements, how smooth the illumination is (`load_probe`'s
`smooth`, see below) and the number of distances.

Sign/units conventions
----------------------
* `pos[theta, dist, (y, x)]` is the sample displacement in *detector pixels*.
  pos = 0 means the detector window is the centred `n x n` crop of the
  `nobj x nobj` projection grid.  (In memory `Rec` keeps positions
  distance-major, `vars['pos'][dist, local_theta, (y, x)]`; `set_pos` below
  does the transpose.)
* Displacements are drawn uniformly from [-amp, +amp], so `amp` is the
  half-width (maximum absolute displacement) in pixels.
* The shift is a cubic B-spline interpolation on the nobj-grid with symmetric
  boundary extension, followed by the crop to n x n.  Beyond
  |pos| ~ (nobj - n)/2 the crop starts sampling mirrored edge data instead of
  real object, so that is the practical limit; with the default nobj = 3n/2
  it is n/4 (= 64 px at n = 256), and the phantom itself starts leaving the
  field of view above ~0.15*n (= 38 px).
* The ground-truth object is built at the DETECTOR size n -- `gen_object(n)`
  for the phantom, `fill_volume(..., span=n)` for --obj-vol -- and only then
  placed in the middle of the nzobj x nobj x nobj grid, with zeros around it
  (`write_centered`).  nobj therefore sets nothing but the width of the blank
  border: raising the margin buys room for the displacement crop instead of
  shrinking the sample, and the sample keeps the same size relative to the
  field of view whatever the margin is.
"""

import os
import re
import glob
import hashlib

import numpy as np
import h5py
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2
import scipy.ndimage as ndimage

# --- acquisition geometry ---------------------------------------------------
# Two configurations, chosen with DISP_STUDY_GEOMETRY:
#
#   default   what ../holotomo3d/test.py uses and every disp_study_final
#             dataset was generated with
#   brain     the ID16A setting the mosaic volume was measured in: a 2.963 um
#             unbinned pixel and distances whose nearest one puts a 20.1 nm
#             voxel on the sample (run_brain.sh / run_dose_brain.sh export it)
#
# In both, the four distances keep the usual ratio -- the farthest is pi/2
# times the nearest -- so the dose argument in `dose_weights` applies to each.
# rec.py never reads these: it takes energy, pixel size, focus-to-detector
# distance and z1 back out of the dataset, so a file always reconstructs on the
# geometry it was made with.
GEOMETRIES = {
    'default': dict(energy=17.1, pixel_1x=1.4760147601476e-6 * 2, f2d=1.217,
                    z1=[5.110, 5.464, 6.879, 9.817]),
    'brain':   dict(energy=17.1, pixel_1x=2.963e-6,               f2d=1.217,
                    z1=[8.25, 8.60, 10.01, 12.95]),
}
GEOMETRY = os.environ.get('DISP_STUDY_GEOMETRY', 'default')
if GEOMETRY not in GEOMETRIES:
    raise SystemExit(f'DISP_STUDY_GEOMETRY={GEOMETRY!r}: expected one of '
                     f'{sorted(GEOMETRIES)}')
_GEOM = GEOMETRIES[GEOMETRY]

ENERGY                  = _GEOM['energy']               # X-ray energy (keV)
DETECTOR_PIXELSIZE_1X   = _GEOM['pixel_1x']             # unbinned detector pixel (m)
DETECTOR_NDET           = 2048                          # ... on the full 2048^2 frame
FOCUSTODETECTORDISTANCE = _GEOM['f2d']                  # focus-to-detector distance (m)


def detector_pixelsize(n, ndet=DETECTOR_NDET):
    """Effective pixel size (m) of an n x n frame binned down from ndet x ndet.

    The detector is always the same 2048^2 chip; a run at n < 2048 is that chip
    binned by ndet/n, so the pixel gets correspondingly bigger and the field of
    view stays put.  Under the brain geometry n = 2048 is the unbinned 2.963 um
    and n = 512 is 4x that, 11.852 um.
    """
    return DETECTOR_PIXELSIZE_1X * (float(ndet) / float(n))


DETECTOR_PIXELSIZE      = detector_pixelsize(512)       # the study default (4x binned)
Z1_ALL = np.array(_GEOM['z1']) * 1e-3                   # sample-to-focus distances (m)


def voxelsize(n, z1=None, ndet=DETECTOR_NDET):
    """Object voxel (m) at detector size `n`: the detector pixel demagnified
    back to the sample.

    The beam comes out of a focus, so a distance z1 magnifies the sample by
    M = focustodetectordistance/z1 on the way to the detector.  rec_mpi
    normalises every distance to the first one and reconstructs on the grid of
    that magnification (`voxelsize = detector_pixelsize / magnifications[0]`),
    so only the nearest distance sets the voxel -- a single-distance and a
    four-distance scan of the same sample share it.
    """
    z1 = Z1_ALL if z1 is None else z1
    m0 = FOCUSTODETECTORDISTANCE / float(np.atleast_1d(z1)[0])
    return detector_pixelsize(n, ndet) / m0

# Directory holding prb_abs_2048.tiff / prb_phase_2048.tiff (4 x 2048 x 2048).
PRB_DIR = os.environ.get('HOLOTOMO_PRB_DIR', '/home/beams2/VNIKITIN/data/prb_id16a')

# --- radiation dose ---------------------------------------------------------
# The beam comes out of a focus, so at sample-to-focus distance z1 the same
# photons are spread over an area ~ z1^2 and the fluence the sample actually
# sees goes as 1/z1^2 -- a projection at the far position costs a fraction of
# the dose of one at the near position.  (Equivalently: the counts per detector
# pixel are the same at every distance, but one detector pixel maps to a sample
# area (px/M)^2 with M = z2/z1, so the dose per unit sample area scales as M^2.)
def dose_weights(z1=Z1_ALL):
    """Dose per projection at each distance, relative to the first (nearest)."""
    z1 = np.asarray(z1, dtype=np.float64)
    return (z1[0] / z1) ** 2


def dose_equivalent_ntheta(ntheta, z1=Z1_ALL, single=0):
    """Angles a single-distance scan needs to match an ndist-distance one.

    `ntheta` angles are collected at every distance in `z1`; the single-distance
    scan sits at `z1[single]`.  Returns the (fractional) number of angles that
    puts the same dose into the sample -- always less than len(z1)*ntheta when
    the single distance is the nearest one, because the far ones are cheap.
    """
    w = dose_weights(z1)
    return float(ntheta) * float(w.sum() / w[single])

# Default root for generated datasets and reconstructions.
OUT_ROOT = os.environ.get('DISP_STUDY_OUT', '/home/beams2/VNIKITIN/tmp/disp_study')


# --- synthetic phantom (verbatim from ../holotomo3d/test.py) ---------------------------
def _draw_frame_edges_inplace(cube, p1, p2):
    cube[p1:p2, p1, p1] = 1; cube[p1:p2, p1, p2] = 1
    cube[p1:p2, p2, p1] = 1; cube[p1:p2, p2, p2] = 1
    cube[p1, p1:p2, p1] = 1; cube[p1, p1:p2, p2] = 1
    cube[p2, p1:p2, p1] = 1; cube[p2, p1:p2, p2] = 1
    cube[p1, p1, p1:p2] = 1; cube[p1, p2, p1:p2] = 1
    cube[p2, p1, p1:p2] = 1; cube[p2, p2, p1:p2] = 1

ROT_XY_DEG, ROT_XZ_DEG = 28, 45     # the one rigid rotation the phantom gets
ROLL_UNITS = (0.0, 10.0, 15.0)      # and the shift after it, in n/256 px per axis


def rot_matrix(ang_xy_deg=ROT_XY_DEG, ang_xz_deg=ROT_XZ_DEG):
    a = np.deg2rad(ang_xy_deg)
    b = np.deg2rad(ang_xz_deg)
    Rz = np.array([[ np.cos(a), -np.sin(a), 0],
                   [ np.sin(a),  np.cos(a), 0],
                   [ 0,          0,         1]], dtype=np.float64)
    Ry = np.array([[ np.cos(b), 0, np.sin(b)],
                   [ 0,         1, 0        ],
                   [-np.sin(b), 0, np.cos(b)]], dtype=np.float64)
    return Ry @ Rz


def rotate3d_once(vol, ang_xy_deg=ROT_XY_DEG, ang_xz_deg=ROT_XZ_DEG, order=1):
    A = np.linalg.inv(rot_matrix(ang_xy_deg, ang_xz_deg))
    center = (np.array(vol.shape) - 1) / 2.0
    offset = center - A @ center
    return ndimage.affine_transform(
        vol, A, offset=offset, order=order, mode="constant", cval=0.0, prefilter=(order > 1)
    )

# The phantom is a wireframe cube dilated by a ball of radius LAYER_RADII[k]:
# LAYER_RADII[k] is the *outer* radius of shell k, whose value is the running
# sum of LAYER_AMPS up to k (the last shell is the core, r < LAYER_RADII[-1]).
# Radii are in units of n/256 px, so the object keeps its proportions at any n.
#
# The shells are laid out as three identical "resolution target" bands - one
# just under the surface, one at mid radius, one deep inside - each ramping from
# 3 units down to 0.5, separated by 5-unit spacers.  The same set of feature
# sizes therefore appears at three depths, so a reconstruction can be scored not
# only on how thin a layer it still resolves but on how far into the sample it
# still resolves it; the innermost band is the one an ill-posed single-distance
# problem loses first.  Radii are in n/256 px and the phantom is built on the
# n-grid, so 0.5 units is 1 px at n = 512 whatever the margin is - right at the
# sampling limit, deliberately: it is the layer that is meant to disappear.
_BAND  = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]        # one band, thinnest at the outside
_SPACE = 5.0                                     # thick shell between two bands
LAYER_R_OUTER = 49.5                             # outer radius of the whole object

# thickness of every shell but the core, outermost first
LAYER_THICK = np.array(_BAND + [_SPACE] + _BAND + [_SPACE] + _BAND + [_SPACE, 5.25],
                       dtype=np.float32)
# value the phantom takes inside each shell (0 = void); neighbours always differ,
# so every thin layer sits between two contrasting ones
LAYER_VALUES = np.array([2, 0, 2, 0, 2, 0,        # band 1 (just under the surface)
                         3,                       # spacer
                         0, 3, 0, 3, 0, 3,        # band 2 (mid radius)
                         0,                       # spacer
                         4, 0, 4, 0, 4, 0,        # band 3 (deep inside)
                         1,                       # spacer
                         0,                       # gap around the core
                         5], dtype=np.float32)    # core

LAYER_RADII = (LAYER_R_OUTER
               - np.concatenate([[0.0], np.cumsum(LAYER_THICK)])).astype(np.float32)
LAYER_AMPS  = np.diff(np.concatenate([[0.0], LAYER_VALUES])).astype(np.float32)
assert LAYER_RADII.shape == LAYER_AMPS.shape == LAYER_VALUES.shape


# --- small features inside the object ----------------------------------------
# The shells only test how thin a *layer* is resolved; they say nothing about
# isolated small features, which is the harder case for an ill-posed
# single-distance problem (no low-frequency neighbour to lean on).  So the deep
# interior also carries a bead target: spheres of shrinking radius on a
# three-armed cross through the middle of the sample.
#
# The positions below are given in the FINAL volume - after the rotation and the
# roll - and gen_object maps them back through both, so the beads always land in
# the mid slices the figures show, whatever the rotation is.  A bead on the
# a0 = 0 plane appears in the horizontal slice, one on a1 = 0 in the vertical
# one, and the arm along a2 appears in both.  Everything within ~23 units of the
# centre is solidly inside the object, so the beads are embedded, never floating.
BEAD_DIST  = np.array([5.0, 10.0, 14.5, 18.0, 20.5, 22.5], dtype=np.float32)
BEAD_RAD   = np.array([2.0,  1.4,  1.0,  0.7,  0.5,  0.35], dtype=np.float32)
BEAD_VALUE = 6.0        # above every shell value, so the contrast never vanishes
# The roll moves the object off the volume centre; shifting the cross by the same
# amount along a2 re-centres it on the sample without moving any bead off the
# a0 = 0 or a1 = 0 plane, which is what keeps them in the mid slices.
BEAD_CENTER = (0.0, 0.0, -float(ROLL_UNITS[2]))


def _bead_cross(dist=BEAD_DIST, rad=BEAD_RAD, value=BEAD_VALUE, origin=BEAD_CENTER):
    """(a0, a1, a2, radius, value) rows, in n/256 px from the volume centre."""
    rows = []
    for axis in range(3):
        for sgn in (1.0, -1.0):
            for d, r in zip(dist, rad):
                c = list(origin)
                c[axis] += sgn * float(d)
                rows.append(c + [float(r), float(value)])
    return np.array(rows, dtype=np.float32)


BEADS = _bead_cross()


def bead_table(n=512, beads=BEADS):
    """Human-readable listing of the bead sizes, in n/256 units and in voxels."""
    px   = n / 256.0
    lines = [f'n={n}  (1 unit = {px:.3g} voxel)   {len(beads)} beads, '
             f'value {BEAD_VALUE:g}',
             ' dist[u]  radius[u]  radius[px]  diameter[px]']
    for d, r in zip(BEAD_DIST, BEAD_RAD):
        lines.append(f'{d:8.2f}  {r:9.2f}  {r*px:10.2f}  {2*r*px:12.2f}')
    return '\n'.join(lines)


def _stamp_beads(obj, n, beads):
    """Overwrite spheres into an unrotated phantom so that, once gen_object has
    rotated and rolled it, they sit exactly where `beads` says in the result."""
    inv  = np.linalg.inv(rot_matrix())
    # exactly the integer shift gen_object applies, not the rounded-off ideal
    roll = np.array([int(u) * n // 256 for u in ROLL_UNITS], dtype=np.float64)
    cen  = (n - 1) / 2.0
    for b in np.asarray(beads, dtype=np.float64):
        f = b[:3] / 256.0 * n + roll          # final position, minus the roll
        c = inv @ f + cen                     # ... and back through the rotation
        r = b[3] / 256.0 * n
        lo = np.maximum(np.floor(c - r - 1).astype(int), 0)
        hi = np.minimum(np.ceil(c + r + 1).astype(int) + 1, n)
        if np.any(lo >= hi):
            continue
        g = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = ((g[0] - c[0])**2 + (g[1] - c[1])**2 + (g[2] - c[2])**2)
        obj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][d2 <= r*r] = b[4]
    return obj


def layer_thicknesses(radii=LAYER_RADII):
    """Shell thicknesses in n/256 px, outermost first (the core is left out)."""
    return -np.diff(np.asarray(radii, dtype=np.float32))


def layer_table(n=512, radii=LAYER_RADII, values=LAYER_VALUES):
    """Human-readable listing of the shells: outer radius, thickness, value — in
    n/256 units and in voxels of the n-grid the phantom is built on."""
    px   = n / 256.0
    th   = np.append(layer_thicknesses(radii), radii[-1])
    lines = [f'n={n}  (1 unit = {px:.3g} voxel)',
             ' k   r_out[u]  r_out[px]  thick[u]  thick[px]  value']
    for k, (r, t, v) in enumerate(zip(radii, th, values)):
        lines.append(f'{k:2d}   {r:8.2f}  {r*px:9.2f}  {t:8.2f}  {t*px:9.2f}  {v:5.1f}')
    return '\n'.join(lines)


# The third study parameter is how sharp the sample itself is.  The phantom is
# built from hard-edged shells and beads and then low-pass filtered, so the edges
# it presents to the reconstruction are as smooth as this one number says.
# Same convention as PRB_SMOOTH below: the filter is exp(-2 pi^2 sigma^2 |v|^2)
# with v in cycles per voxel, so sigma = 1/(pi*sqrt(2)) is exactly the exp(-|v|^2)
# filter ../holotomo3d/test.py (and every run of this study so far) used - that
# value is the default, and it is a very mild blur.  Unlike PRB_SMOOTH, sigma is
# in voxels of the *object* grid, so it means the same physical blur whatever
# nobj is; at the standard geometry an object voxel and a detector pixel are the
# same size in the sample, so the two sigmas are directly comparable numbers.
OBJ_SMOOTH = float(1.0 / (np.pi * np.sqrt(2.0)))   # ~0.2251 voxel == legacy exp(-|v|^2)


def phantom_id(radii=LAYER_RADII, amps=LAYER_AMPS, beads=BEADS, smooth=OBJ_SMOOTH):
    """Short hash of the phantom specification, used to key the phantom cache —
    edit any of the shells or the beads and the cached volume is not reused.

    The smoothing is folded in only when it differs from the default, so every
    phantom already cached on disk keeps the id stored in its attrs and is still
    reused; a non-default blur can never collide with one of them."""
    h = hashlib.md5(np.asarray(radii, dtype='float32').tobytes() +
                    np.asarray(amps,  dtype='float32').tobytes() +
                    np.asarray(beads, dtype='float32').tobytes())
    if abs(float(smooth) - OBJ_SMOOTH) > 1e-6:
        h.update(np.float32(smooth).tobytes())
    return h.hexdigest()[:8]


def gen_object(n, delta, beta, radii=LAYER_RADII, amps=LAYER_AMPS, beads=BEADS,
               smooth=OBJ_SMOOTH):
    obj  = np.zeros((n, n, n), dtype=np.float32)
    amps = np.asarray(amps, dtype=np.float32)
    dil  = np.asarray(radii, dtype=np.float32) / 256.0 * n
    r_frame = int(n * 0.18)

    ax = np.arange(-n//2, n//2, dtype=np.float32)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = x*x + y*y + z*z
    del x, y, z

    # Every shell dilates the same wireframe, so its spectrum is built once, and
    # the balls are consumed one at a time: keeping a list of them would cost
    # len(dil) * n^3 * 8 bytes, which is tens of GB once there are many layers.
    cube = np.zeros((n, n, n), dtype=np.float32)
    p1 = n//2 - r_frame//2
    p2 = n//2 + r_frame//2
    _draw_frame_edges_inplace(cube, p1, p2)
    fcube = fftn(fftshift(cube), workers=-1).astype(np.complex64, copy=False)
    del cube

    work = np.empty((n, n, n), dtype=np.complex64)
    for a, d in zip(amps, dil):
        circ  = (r2 < (d*d)).astype(np.float32, copy=False)
        fcirc = fftn(fftshift(circ), workers=-1).astype(np.complex64, copy=False)
        np.multiply(fcirc, fcube, out=work)
        del circ, fcirc
        conv = fftshift(ifftn(work, workers=-1)).real
        obj += a * (conv > 1.0)
    del work, r2, fcube

    if beads is not None and len(beads):
        _stamp_beads(obj, n, beads)

    obj = rotate3d_once(obj, order=1)
    obj = np.roll(obj, -int(ROLL_UNITS[2]) * n // 256, axis=2)
    obj = np.roll(obj, -int(ROLL_UNITS[1]) * n // 256, axis=1)
    np.maximum(obj, 0, out=obj)
    if smooth > 0:
        # exp(-2 pi^2 sigma^2 |v|^2), v in cycles/voxel — the same transfer
        # function probe_lowpass() applies to the illumination.  Built by
        # broadcasting rather than meshgrid: three n^3 grids are 11 GB apiece at
        # nobj=1408, and one array is all the filter needs.
        v  = (np.arange(-n//2, n//2, dtype=np.float32) / n)      # cycles / voxel
        v2 = v * v
        r2 = v2[:, None, None] + v2[None, :, None] + v2[None, None, :]
        r2 *= -2.0 * np.pi**2 * float(smooth)**2
        filt = fftshift(np.exp(r2, out=r2))
        fu = fftn((obj))
        obj = ifftn((fu * filt)).real
        obj[obj < 0] = 0
    return (obj * (-delta + 1j*beta)).astype(np.complex64, copy=False)


def write_centered(ds_re, ds_im, src_re, src_im, nzobj, nobj):
    """Write a span^3 ground-truth object into the middle of the object grid.

    `src_re` / `src_im` are the real and imaginary parts of the object as it was
    built at the detector size -- arrays or open HDF5 datasets, normally of
    shape (span, span, span) -- and they are copied into the centred core of the
    (nzobj, nobj, nobj) /obj_re,/obj_im datasets, with 0 (transmission 1)
    everywhere else.  This is the same placement fill_volume makes for
    --obj-vol, so both kinds of ground truth follow the one rule: the sample is
    generated for the detector, and the object grid only adds a blank margin.

    One destination z-batch is resident at a time, so a phantom that is itself
    streamed off disk never needs the whole padded grid in memory.
    """
    sz, sy, sx = (int(v) for v in src_re.shape)    # cubic for both callers, but
    if sz > nzobj or sy > nobj or sx > nobj:       # not assumed to be
        raise ValueError(f'object of {sz} x {sy} x {sx} px does not fit in a '
                         f'{nzobj} x {nobj} x {nobj} grid')
    z0, y0, x0 = (nzobj - sz) // 2, (nobj - sy) // 2, (nobj - sx) // 2
    batch = min(h5_batch(nobj * nobj * 4), nzobj)
    buf   = np.zeros((batch, nobj, nobj), dtype='float32')
    for i0 in range(0, nzobj, batch):
        i1 = min(i0 + batch, nzobj)
        k  = i1 - i0
        j0, j1 = i0 - z0, i1 - z0                  # the same rows within the core
        c0, c1 = max(j0, 0), min(j1, sz)           # ... clipped to it (empty = border)
        for ds, src in ((ds_re, src_re), (ds_im, src_im)):
            buf[:k] = 0
            if c1 > c0:
                buf[c0 - j0:c1 - j0, y0:y0 + sy, x0:x0 + sx] = src[c0:c1]
            ds[i0:i1] = buf[:k]


# --- probe ------------------------------------------------------------------
# The second study parameter is how smooth the illumination is.  The ID16A probe
# is low-pass filtered with an isotropic Gaussian of standard deviation
# `smooth` *detector pixels* before it is normalised; smooth = 0 leaves the
# measured probe untouched, and large values wash the speckle out until the
# illumination is essentially flat.  Since a single distance recovers the object
# only through the diversity that the structured probe + the random
# displacements put into the data, this is the knob that removes that diversity.
#
# In Fourier space the filter is exp(-2 pi^2 sigma^2 |v|^2) with v in cycles per
# pixel, so sigma = 1/(pi*sqrt(2)) reproduces exactly the exp(-|v|^2) filter that
# ../holotomo3d/test.py (and the first runs of this study) used - that value is
# the default, and it is a very mild blur (0.225 px), i.e. "the probe as measured".
PRB_SMOOTH = float(1.0 / (np.pi * np.sqrt(2.0)))   # ~0.2251 px == legacy exp(-|v|^2)


def probe_lowpass(n, smooth):
    """fftshifted transfer function of a Gaussian blur of `smooth` px, or None."""
    if smooth <= 0:
        return None
    v = (np.arange(-n//2, n//2, dtype=np.float32) / n)     # cycles / px
    vx, vy = np.meshgrid(v, v, indexing="ij")
    return fftshift(np.exp(-2.0 * np.pi**2 * float(smooth)**2
                           * (vx*vx + vy*vy)).astype(np.float32))


def load_probe(n, ndist, prb_dir=PRB_DIR, smooth=PRB_SMOOTH):
    """ID16A probe, cropped to n x n, Gaussian-smoothed by `smooth` px and normalised.

    `smooth` is the standard deviation of the blur in detector pixels (see above);
    the default reproduces ../holotomo3d/test.py.  The mean-|prb| normalisation is
    applied after the blur, so the flat-field level is the same for every value.
    """
    from holotomocupy.utils import read_tiff
    prb_abs   = read_tiff(f'{prb_dir}/prb_abs_2048.tiff')[:ndist]
    prb_phase = read_tiff(f'{prb_dir}/prb_phase_2048.tiff')[:ndist]
    if prb_abs.shape[0] < ndist:
        raise SystemExit(f'{prb_dir} holds {prb_abs.shape[0]} distances, need {ndist}')
    prb = prb_abs * np.exp(1j * prb_phase).astype('complex64')
    prb = prb[:, prb.shape[1]//2-n//2:prb.shape[1]//2+n//2,
                 prb.shape[2]//2-n//2:prb.shape[2]//2+n//2]
    filt = probe_lowpass(n, smooth)
    if filt is not None:
        prb = ifft2(fft2(prb) * filt)
    prb /= np.mean(np.abs(prb), axis=(1, 2))[:, None, None]
    return prb.astype('complex64')


def probe_contrast(prb):
    """std(|prb|)/mean(|prb|) per distance - a one-number summary of how much
    structure the blur has left in the illumination."""
    a = np.abs(np.asarray(prb))
    ax = tuple(range(1, a.ndim))
    return a.std(axis=ax) / np.maximum(a.mean(axis=ax), 1e-30)


# --- a real sample volume instead of the phantom -----------------------------
# The phantom above is convenient but small and synthetic.  For a harder test the
# object can instead be read from a file holding a real reconstructed volume
# (e.g. /data3/vnikitin/mosaic_brain/init.h5), rescaled onto the nobj grid.
#
# Sign convention, the same one gen_object uses: the stored volume IS the real
# part of the object, i.e. it is already -delta-like (mostly negative), and the
# imaginary part follows from the delta/beta ratio as obj_im = -obj_re/(delta/beta).

def open_volume(spec, nzobj=None, nobj=None):
    """Open a sample volume for slice-wise reading; returns (accessor, file_or_None).

    `spec` is a path, optionally "path::dataset" for HDF5.  The accessor is
    indexable by z-slice and has .shape/.dtype, so an h5py dataset, a memmap and
    an mmap'd .npy all work the same way.  Close the second element when done.

      .h5/.hdf5   the named dataset, or the largest 3-D one in the file
      .npy        memory-mapped
      anything    raw float32 of exactly (nzobj, nobj, nobj)
    """
    path, _, dset = spec.partition('::')
    if path.endswith(('.h5', '.hdf5')):
        f = h5py.File(path, 'r')
        if not dset:
            cand = []
            f.visititems(lambda k, v: cand.append((v.size, k))
                         if isinstance(v, h5py.Dataset) and v.ndim == 3 else None)
            if not cand:
                f.close()
                raise SystemExit(f'{path}: no 3-D dataset found')
            dset = max(cand)[1]
        return f[dset], f
    if path.endswith('.npy'):
        return np.load(path, mmap_mode='r'), None
    if nzobj is None or nobj is None:
        raise SystemExit(f'{path}: raw volumes need an explicit shape')
    return np.memmap(path, dtype='float32', mode='r',
                     shape=(nzobj, nobj, nobj)), None


def volume_id(spec, span, scale, delta_beta, nobj, nzobj):
    """Short hash of everything fill_volume's output depends on (cache key)."""
    path = spec.partition('::')[0]
    try:
        st = os.stat(path)
        stamp = f'{st.st_size}:{int(st.st_mtime)}'
    except OSError:
        stamp = 'missing'
    h = hashlib.md5(f'{spec}|{stamp}|{span}|{scale:.9g}|{delta_beta:.9g}|'
                    f'{nobj}|{nzobj}'.encode())
    return h.hexdigest()[:8]


def fill_volume(spec, ds_re, ds_im, nzobj, nobj, span, scale=1.0, delta_beta=100.0,
                log=None, gpu=True):
    """Write a rescaled sample volume into the /obj_re,/obj_im of a data file.

    Two steps, as in ../mosaic_brain/gen_data.py:

      1. CROP VERTICALLY -- only the central `nzobj/q` source rows can land
         inside the object grid; the rest are never read.
      2. SCALE by one isotropic factor q = span/sx on all three axes, so the
         sample keeps its aspect ratio and ends up `span` object px wide,
         centred on the grid.  Whatever still falls outside is cropped and
         whatever is not covered stays 0 (transmission 1).

    Downsampling is an integer block-average, then a Gaussian pre-filter for the
    residual factor, then a linear zoom -- otherwise a 2x-smaller grid aliases.
    Work is done one destination slice at a time, so only a couple of source
    slices and one nobj^2 plane are ever resident.
    """
    log = log or (lambda m: None)
    if gpu:
        try:
            import cupy as xp
            from cupyx.scipy.ndimage import zoom, gaussian_filter
        except Exception:
            gpu = False
    if not gpu:
        import numpy as xp
        from scipy.ndimage import zoom, gaussian_filter

    vol, fh = open_volume(spec, nzobj, nobj)
    try:
        sz, sy, sx = vol.shape
        q = float(span) / sx

        keep = min(sz, int(np.ceil(nzobj / q)))
        z0   = (sz - keep) // 2
        fi   = max(1, int(1.0 / q))          # integer pre-average, antialiasing
        q2   = q * fi                        # residual zoom applied after averaging
        szp, syp, sxp = keep // fi, sy // fi, sx // fi
        # the linear zoom alone is a poor antialias filter below 1x; a Gaussian of
        # half the sampling-step increase brings it back to roughly band-limited
        pre = 0.5 * (1.0 / q2 - 1.0) if q2 < 1.0 else 0.0
        log(f'  volume {vol.shape} -> {span} px wide  (q={q:.4f})')
        log(f'  crop z {z0}:{z0+keep} of {sz}, block-average {fi}x -> '
            f'{(szp, syp, sxp)}, presmooth {pre:.3f} px, zoom {q2:.4f}x')
        log(f'  obj_re = {scale:g} * volume,  obj_im = -obj_re/{delta_beta:g}')

        # every step below is a mean or a linear interpolation, so the sample's
        # amplitude carries through unchanged and only the finest texture is
        # averaged away.  Accumulated here and reported at the end, because
        # "did the rescale preserve the values?" is the one thing that would
        # silently invalidate a run.
        acc = dict(ss=0.0, sn=0, ds=0.0, dn=0)

        def pre_slice(j):
            """Pre-averaged slice j of the cropped band, as an xp float32 plane."""
            s0  = z0 + j * fi
            raw = np.asarray(vol[s0:s0 + fi, :syp * fi, :sxp * fi], dtype='float32')
            g   = xp.asarray(raw).mean(axis=0)
            g   = g.reshape(syp, fi, sxp, fi).mean(axis=(1, 3))
            nz  = g != 0                     # the background is exactly 0
            acc['ss'] += float(g[nz].sum())
            acc['sn'] += int(nz.sum())
            return g

        zero  = np.zeros((nobj, nobj), dtype='float32')
        cache = {}
        step  = max(1, nzobj // 20)
        for i in range(nzobj):
            if i % step == 0:
                log(f'  ... slice {i}/{nzobj}')
            # destination row i, measured from the grid centre, mapped back onto
            # the pre-averaged source grid
            zc = (i - (nzobj - 1) / 2) / q2 + (szp - 1) / 2
            j0 = int(np.floor(zc))
            if j0 < 0 or j0 > szp - 1:
                ds_re[i] = zero
                ds_im[i] = zero
                continue
            j1 = min(j0 + 1, szp - 1)
            for j in (j0, j1):
                if j not in cache:
                    cache[j] = pre_slice(j)
            w = np.float32(zc - j0)
            s = cache[j0] * (1 - w) + cache[j1] * w
            for j in list(cache):
                if j < j0:
                    del cache[j]

            if pre > 0:
                s = gaussian_filter(s, pre, mode='constant', cval=0.0)
            s = zoom(s, q2, order=1, mode='constant', cval=0.0)

            out = xp.zeros((nobj, nobj), dtype='float32')
            sh, sw   = s.shape
            y0, x0   = (nobj - sh) // 2, (nobj - sw) // 2
            sy0, sx0 = max(0, -y0), max(0, -x0)
            dy0, dx0 = max(0, y0), max(0, x0)
            h  = min(sh - sy0, nobj - dy0)
            wd = min(sw - sx0, nobj - dx0)
            out[dy0:dy0 + h, dx0:dx0 + wd] = s[sy0:sy0 + h, sx0:sx0 + wd]
            out *= np.float32(scale)

            nz = out != 0
            acc['ds'] += float(out[nz].sum())
            acc['dn'] += int(nz.sum())

            re = out.get() if hasattr(out, 'get') else out
            ds_re[i] = re
            ds_im[i] = -re / np.float32(delta_beta)
        log(f'  ... slice {nzobj}/{nzobj}')
        if acc['sn'] and acc['dn']:
            src = acc['ss'] / acc['sn']
            dst = acc['ds'] / acc['dn'] / max(scale, 1e-30)
            log(f'  mean amplitude in the sample {src:+.5g} -> {dst:+.5g}  '
                f'(ratio {dst / src:.4f}, 1 = the rescale preserved it)')
    finally:
        if fh is not None:
            fh.close()


# --- large 3-D Gaussian blur --------------------------------------------------
# scipy's gaussian_filter is fine for the n = 512 study (sigma = 8 on a ~190 x
# 614^2 slab), but at n = 2048 the starting object is a ~750 x 2458^2 slab and
# sigma = 32 means a 257-tap kernel on 4.5e9 voxels along each of three axes --
# hours on a CPU.  Below GPU_BLUR_BYTES nothing changes and scipy is used, so the
# n = 512 results stay bit-for-bit what they were.
GPU_BLUR_BYTES = int(os.environ.get('DISP_STUDY_GPU_BLUR_BYTES', 1 << 31))


def gaussian_blur3d(vol, sigma, gpu=None, band_bytes=1 << 29):
    """gaussian_filter(vol, sigma, mode='constant', cval=0) for big float32 slabs.

    Separable, so it is split into an in-plane pass over single z-slices and a
    z pass over bands of rows; the GPU only ever holds one slice or one band.
    The result is scipy's to float32 rounding -- the axis order of a separable
    filter does not matter.
    """
    vol = np.ascontiguousarray(vol, dtype='float32')
    if sigma <= 0:
        return vol
    if gpu is None:
        gpu = vol.nbytes >= GPU_BLUR_BYTES
    if gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter1d as g1d
        except Exception:
            gpu = False
    if not gpu:
        return ndimage.gaussian_filter(vol, sigma, mode='constant', cval=0.0)

    nz, ny, nx = vol.shape
    for i in range(nz):                          # in-plane (y, x)
        g = cp.asarray(vol[i])
        g = g1d(g1d(g, sigma, axis=0, mode='constant', cval=0.0),
                sigma, axis=1, mode='constant', cval=0.0)
        vol[i] = cp.asnumpy(g)
    band = max(1, int(band_bytes // max(1, nz * nx * 4)))
    for y0 in range(0, ny, band):                # along z
        y1 = min(y0 + band, ny)
        g = cp.asarray(vol[:, y0:y1])
        g = g1d(g, sigma, axis=0, mode='constant', cval=0.0)
        vol[:, y0:y1] = cp.asnumpy(g)
    del g
    cp.get_default_memory_pool().free_all_blocks()
    return vol


# --- naming -----------------------------------------------------------------
def case_name(amp, ndist, prb_smooth=PRB_SMOOTH, obj_smooth=OBJ_SMOOTH):
    """Directory name of one (amp, ndist, probe-, object-smoothness) case.

    Each smoothness suffix is left off at its default, so the directories of the
    pure displacement sweep keep the names they already have on disk.
    """
    name = f'amp{float(amp):g}_ndist{int(ndist)}'
    if abs(float(prb_smooth) - PRB_SMOOTH) > 1e-6:
        name += f'_prbs{float(prb_smooth):g}'
    if abs(float(obj_smooth) - OBJ_SMOOTH) > 1e-6:
        name += f'_objs{float(obj_smooth):g}'
    return name


# --- reading back a finished run --------------------------------------------
REC_RE = re.compile(r'rec_n(\d+)_ntheta(\d+)$')


def read_summary(rec_dir):
    """summary.txt of a reconstruction as a dict, {} if it is not there yet."""
    out = {}
    try:
        with open(os.path.join(rec_dir, 'summary.txt')) as fh:
            for line in fh:
                k, _, v = line.partition(' ')
                out[k] = v.strip()
    except OSError:
        pass
    return out


def read_conv(rec_dir):
    """conv.csv of a reconstruction as (iter, err, time) arrays, None if absent.

    rec_mpi writes one row per checkpoint step: `err` is the data misfit
    F0 summed over distances and divided by the number of data points, so the
    two legs of a dose comparison -- which differ in ntheta, hence in data size
    -- are on the same scale.  The iter = -1 row is the initial guess and is
    returned as-is; callers that plot against iteration usually drop it.
    """
    path = os.path.join(rec_dir, 'conv.csv')
    if not os.path.isfile(path):
        return None
    try:
        t = np.genfromtxt(path, delimiter=',', names=True)
        t = np.atleast_1d(t)
        return t['iter'].astype(int), t['err'].astype(float), t['time'].astype(float)
    except (OSError, ValueError, KeyError):
        return None


def rec_dirs(case_dir):
    """{(n, ntheta): path} for the reconstruction folders of one case.

    rec.py names them rec_n<n>_ntheta<ntheta>; a plain 'rec' left by an older
    run is keyed by what its summary.txt says, or (0, 0) when it says nothing.
    """
    found = {}
    for d in sorted(glob.glob(os.path.join(case_dir, 'rec*'))):
        if not os.path.isdir(d):
            continue
        m = REC_RE.search(d)
        if m:
            found[(int(m.group(1)), int(m.group(2)))] = d
        elif os.path.basename(d) == 'rec':
            s = read_summary(d)
            found[(int(s.get('n', 0)), int(s.get('ntheta', 0)))] = d
    return found


def last_checkpoint(rec_dir):
    """Path of the highest-numbered checkpoint in rec_dir, or None."""
    cps = glob.glob(os.path.join(rec_dir, 'checkpoints', 'checkpoint_*.h5'))
    if not cps:
        return None
    return max(cps, key=lambda f: int(re.search(r'checkpoint_(\d+)', f).group(1)))


def read_slices(path):
    """Middle horizontal (z) and vertical (y) slices of obj_re."""
    with h5py.File(path, 'r') as f:
        d = f['obj_re']
        return d[d.shape[0] // 2], d[:, d.shape[1] // 2]


def dose_case_name(ndist, ntheta, amp, prb_smooth=PRB_SMOOTH, obj_smooth=OBJ_SMOOTH):
    """Directory name of one case of the dose-matched comparison.

    Unlike case_name(), ntheta is in the name: the whole point of the comparison
    is two runs that differ in ndist *and* in the number of angles.
    """
    name = f'ndist{int(ndist)}_ntheta{int(ntheta)}_amp{float(amp):g}'
    if abs(float(prb_smooth) - PRB_SMOOTH) > 1e-6:
        name += f'_prbs{float(prb_smooth):g}'
    if abs(float(obj_smooth) - OBJ_SMOOTH) > 1e-6:
        name += f'_objs{float(obj_smooth):g}'
    return name


# --- random displacements ---------------------------------------------------
def gen_positions(ntheta, ndist, amp, seed=10):
    """Uniform random sample displacements in [-amp, amp] px, shape [ntheta, ndist, 2]."""
    rng = np.random.default_rng(seed)
    pos = (2.0 * amp) * (rng.random([ntheta, ndist, 2]) - 0.5)
    return pos.astype('float32')


def set_pos(cl, pos_global):
    """Copy this rank's theta-slice of a global [ntheta, ndist, 2] array into
    cl.vars['pos'], which is pinned numpy [ndist, local_ntheta, 2]."""
    loc = np.asarray(pos_global, dtype='float32')[cl.st_theta:cl.end_theta]
    cl.vars['pos'][:] = np.ascontiguousarray(loc.transpose(1, 0, 2))


# --- reference (flat field) -------------------------------------------------
def gen_ref(cl, prb):
    """|D prb_j| for each distance — the noiseless flat field, same as
    PrbfitTerm.gen_sqrt_ref but without requiring lam_prbfit > 0.
    Returns numpy [ndist, nz, n] float32."""
    import cupy as cp
    ref = np.empty([cl.ndist, cl.nz, cl.n], dtype='float32')
    for j in range(cl.ndist):
        cl._dist_idx = j
        ref[j] = cp.abs(cl.cl_prop.D(cp.asarray(prb[j])[cp.newaxis], j)[0]).get()
    return ref


# --- Poisson noise ----------------------------------------------------------
def add_poisson_noise(data, photons, seed):
    """In-place Poisson noise on INTENSITY data of any leading shape.

    `photons` is the mean number of detected photons per pixel for an
    unattenuated beam (the probe is normalised to mean |prb| = 1, so the
    intensity is ~1 in the flat regions).  photons <= 0 leaves the data alone.

    Takes and returns intensity, matching Rec.gen_data and Reader.read_data --
    it used to do the same job on amplitudes, squaring and un-squaring around
    the draw, back when the misfit was on amplitudes.
    """
    if photons <= 0:
        return data
    rng = np.random.default_rng(seed)
    # one frame at a time, so the float64 temporary stays small
    for idx in np.ndindex(data.shape[:-2]):
        inten = rng.poisson(data[idx].astype('float64') * photons)
        data[idx] = (inten / photons).astype('float32')
    return data


# --- dataset file layout ----------------------------------------------------
# {out}/data.h5
#   /data      (ndist, ntheta, nz, n)  float32  measured intensity
#                                               (distance-major: same order as
#                                                Rec.data, so reads are direct)
#   /ref       (ndist, nz, n)          float32  sqrt of flat-field intensity
#   /pos       (ntheta, ndist, 2)      float32  true displacements [px]
#                                               (theta-major: same order as the
#                                                /pos of a Writer checkpoint)
#   /prb_abs   (ndist, nz, n)          float32
#   /prb_phase (ndist, nz, n)          float32
#   /obj_re    (nzobj, nobj, nobj)     float32  ground-truth object, real part
#   /obj_im    (nzobj, nobj, nobj)     float32  ground-truth object, imag part
#   /theta     (ntheta,)               float32
#   /z1        (ndist,)                float64
#   attrs: n nz nobj nzobj ntheta ndist energy detector_pixelsize
#          focustodetectordistance amp seed photons delta beta nobj_factor
#          prb_smooth prb_contrast

def h5_batch(nbytes_row, cap=1 << 28):
    """Rows per HDF5 read/write so a single MPI-IO call stays well under 2 GiB."""
    return max(1, int(cap // max(1, nbytes_row)))


def read_attrs(path, comm):
    """Read the scalar metadata of a dataset file on every rank."""
    import h5py
    with h5py.File(path, 'r', driver='mpio', comm=comm) as f:
        a = dict(f.attrs)
        a['theta'] = f['theta'][:].astype('float32')
        a['z1']    = f['z1'][:]
    return a
