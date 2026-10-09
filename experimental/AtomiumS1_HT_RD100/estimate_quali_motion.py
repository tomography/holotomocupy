#!/usr/bin/env python
"""
Measure per-plane sample drift from the post-scan quality retakes.

    python estimate_quali_motion.py config_steps15.conf [--validate]

Step 3 of steps15.py calls `write_measured()` itself and feeds the result into
the rhapp search; running this as a script is for checking the answer, not for
feeding the pipeline.

WHAT IS MEASURED.  Every scan in this sweep writes ntheta+3 frames.  Frame
ntheta is the last scan frame, at omega = 180; frames ntheta+1 and ntheta+2 are
RETAKES made immediately after the scan ends, at omega = 90 and omega = 0.
Comparing a retake with the in-scan frame at the same angle measures how far the
sample moved between the two exposures.  Frame ntheta is its own retake, so its
drift is zero by construction and everything is referenced to the end of scan:

    omega   0   frame 0          vs retake ntheta+2    measured
    omega  90   frame ntheta/2   vs retake ntheta+1    measured
    omega 180   frame ntheta     --                    (0, 0) by construction

That is Peter Cloetens' method -- it is what quali.mat's corr_imagesafterscan
records, and correct_motion_calc.m turns into the ref_h/ref_v that his
correct_motion.txt is literally random_disp minus.

SIGN, MEASURED NOT ARGUED.  `_peak(_corr(a, b))` returns the lag s with
a[y] = b[y - s] (checked against np.roll).  Writing the sample position in
frame j as p_j = r_j + d_j, commanded plus drift, and the retake's as p_ret,

    s = p_j - p_ret      ->      d_j = s - (r_j - r_ret)

since the retake defines zero drift.  So the commanded displacement is
subtracted ARITHMETICALLY from the answer, never undone by shifting the image --
the same arrangement as quali_series.m:562.  d_j is then in the same sense as
the commanded displacement, which step 3 adds unnegated, so there is NO sign
flip here, unlike rhapp.  `--validate` is what settles it before anything is
reconstructed.

ON THESE SCANS THE COMMANDED DIFFERENCE IS ZERO.  The piezo is parked for the
omega=180 anchor and for both retakes, and also at frames 0 and ntheta/2 -- the
very frames the retakes duplicate.  So r_j - r_ret = 0 at every point measured
here and the subtraction above is a no-op in practice.  It is still done,
because it is what makes the quantity well defined, and because an RD scan that
parked its piezo elsewhere would otherwise be read wrong in silence.

WHY THE CORRELATION IS UNWHITENED.  Because the commanded difference is zero,
the sample peak sits at the drift and the detector-fixed illumination sits at
zero lag, and nothing separates them but the drift itself.  Phase correlation
whitens, which boosts the shared high-frequency detector pattern to the same
weight as the sample, and it loses: an earlier version of this file used
phase_corr and reported +0.15 px against a true -113.42.  In that same run the
unwhitened 1-D profile cross-check read -104.  So this file correlates with
estimate_rhapp._corr -- mean removed, L2 scaled, NOT whitened -- and keeps the
1-D profile alongside as an independent check that fails differently.

Three further defences, in order of how much they are worth:

 1. FLAT-FIELD, ALWAYS.  (raw - dark) / interpolated(ref0, ref1), then -log.
    Without it the static background dominates and the answer is [0, 0].
 2. A STATIC TEMPLATE, subtracted before correlating: the mean of --template
    frames spread over the scan, in which the sample smears over its own
    hundreds of px of commanded displacement while the illumination does not.
    The frames being compared are nudged out of that average -- including one
    puts a negative copy of its own photon noise into the frame it is compared
    with, which digs a hole at exactly zero lag.
 3. --zero-mask, a disc blanked at zero lag, DEFAULT OFF.  It is off because it
    cannot be right for every scan in the sweep: FT_RD300 drifts 113 px and
    would welcome it, HT_RD025 drifts 1-4 px and would be destroyed by it.
    Turn it on per scan, and only when the 1-D cross-check says the 2-D peak
    has gone to zero.

Stripe removal and flux normalisation are deliberately NOT done here -- steps
1/2 already do them for the pipeline, and this estimator reads raw frames only
because the retakes are past the end of what step 2 writes.

POLYNOMIAL MODEL -- PETER'S, NOT A POLYNOMIAL IN FRAME INDEX FOR BOTH.
`arg = j/ntheta - 1`, zero at the END of the scan, and both components are
fitted with NO CONSTANT TERM, which is what enforces the omega=180 anchor
structurally instead of spending a fit point on it.

 * VERTICAL: degree min(npts, 3) = 2 on the two measured points, then the mean
   is removed.  Vertical drift is rotation-invariant, so a curve in time is the
   whole model.  This reproduces ESRF's ref_v exactly (residual 0.0000).
 * HORIZONTAL: degree min(npts//2, 3) = 1, and it is NOT a curve in frame
   index.  The sample drifts by a 2-D vector in the LAB frame and the detector
   sees its projection, so two time curves sx(t), sy(t) are fitted and combined
   as sx*cos(theta) + sy*sin(theta).  A pure sinusoid in theta is
   indistinguishable from a rotation-axis shift, which step 4b owns, so it is
   subtracted afterwards.  (The result does not depend on the sign convention
   for theta: flipping it flips sy and leaves sx*cos + sy*sin unchanged.)

   This is why the old quadratic-in-frame-index horizontal only ever reached
   corr +0.56 against ESRF's ref_h while the vertical matched exactly.  It was
   the wrong model, not an is360 effect.

The fit residual is meaningless here and is deliberately not reported as a
quality number -- both components are exactly determined.  The honest error bar
is the MAD across crops at each point, propagated through the fit as the band.

PETER'S NUMBERS ARE ON A 2x2-BINNED GRID AND OURS ARE NOT.  Both quali.mat and
reference_motion.mat are written at his 9 nm pixel against our 4.5 nm, so every
comparison carries a factor --motion-bin (default 2).  validate() does not
merely apply it: it reports the implied least-squares scale, so a wrong bin
factor shows up as 1 or 4 instead of being absorbed silently.  His numbers:

  FT_RD300  ref_v ptp 56.72 (x2 -> 113.44 object px; ndist 1, so nmag = 1)
            quali corrtot (x, y) at omega 0   (-24.6877, -56.7116)
                  corrtot90       at omega 90   (3.6087, -27.3888)
  HT_RD300  ref_v ptp 8.2691 (x2 -> 16.54); reference_plane 3, i.e. ref_dist 2
            quali, all four planes, (omega 0 x, y) then (omega 90 x, y):
              plane 1  ( 0.5885, -22.2422)  ( 2.6022, -11.4734)
              plane 2  (-2.0119, -18.0614)  ( 3.6796, -10.1350)
              plane 3  (-1.3755,  -6.8023)  ( 2.5119,  -4.8915)   <- ref_dist 2
              plane 4  (-3.2351,  -4.7954)  ( 1.8553,  -2.9798)
  HT_RD025  no reference_motion.mat
            quali corrtot    (-0.3329, -0.8442)
                  corrtot90  (3.9848, -2.0511)

quali rows are (x, y) in his binned px of that plane's DETECTOR grid: multiply
by --motion-bin, then divide by norm_magnifications[k] to reach the object px
this file writes.  The other four scans have neither file; they are the reason
this exists.

THERE IS NO THIRD SCALE FACTOR, and that was worth checking.  ref_v is not on
plane 3's detector grid -- holotomo_slave.m:1731 multiplies it by
maxM/Mv(reference_plane) first.  On FT that is 1 and invisible; on HT_RD300 the
no-constant quadratic through plane 3's two quali points has ptp 6.8023 while
ref_v's is 8.2691, a ratio of 1.2156.  Our 1/norm_magnifications[2] is 1.21450
-- the same number to 0.09%.  So his maxM/Mv IS our divide-by-nmag, both files
land on the object grid, and --motion-bin alone closes the comparison.  That is
why validate() prints the implied least-squares scale: it should read 1, and a
0.5, 2 or 1.21 instead would say the unit chain has drifted.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esrf_layout import Layout
# The repo's own correlator and peak finder, not phase correlation: _corr is
# mean-removed and L2-scaled but NOT whitened, and _peak can search around a
# known expected lag.  _band is its low-pass/high-pass (scipy, imported lazily).
from estimate_rhapp import _band, _corr, _peak

DEFAULTS = dict(
    path=None, pfile=None,
    # Several crops, not one: a single crop can still land on a wrong local
    # peak when a pair is weak, and the scatter across crops is the error bar.
    # Nothing much above 2048 -- the flat-fielding near the frame edges is poor
    # enough that wider crops start voting for the wrong peak.
    crop=(1024, 1280, 1536, 1792, 2048),
    # Wide on purpose: the FT RD300 vertical drift reaches 113 object px, so
    # the +-40 px window this file's ancestor used would have missed it.
    search=256,
    zero_mask=0,       # disc blanked at zero lag; off, see the module docstring
    smooth=0.0,        # Gaussian low-pass sigma, detector px
    highpass=0.0,      # unsharp high-pass sigma; off, it removes the sample too
    template=32,       # frames averaged into the static template; 0 disables
    nflat=8,           # flats/darks averaged
    min_peak=0.010,    # a pair that locked onto the sample gives ~0.02-0.03
    # A point whose crops disagree by more than this is DROPPED, not medianed:
    # the median of three crops that disagree by 100 px is not an estimate.
    max_scatter=5.0,
    profile_tol=6.0,   # 2-D vs 1-D profile disagreement that earns a warning
    out=None, fig=None,
)


def options(**kw):
    """Estimator settings as a namespace, for callers that have no argparse."""
    bad = set(kw) - set(DEFAULTS)
    if bad:
        raise TypeError(f'unknown estimator option(s): {sorted(bad)}')
    o = dict(DEFAULTS)
    o.update(kw)
    return argparse.Namespace(**o)


# --- raw frames ------------------------------------------------------------

def shift_table(lay, k, log=print):
    """Plane k's commanded displacement, padded to cover the retakes.

    The tables are not all the same length.  FT RD300 writes ntheta+3 rows with
    the last three exactly `0.000 -0.000`; HT RD300 writes ntheta and stops.
    FT's explicit zeros are the evidence that zero is the right fill: the piezo
    is parked for the omega=180 anchor and for both retakes.
    """
    src = lay.shift_source(k)
    if not os.path.exists(src):
        # Only ever true of an RD000 scan, where nothing was commanded.
        log(f'  plane {k + 1}: no shift table ({src}) -- zeros')
        return np.zeros([lay.ntheta + 3, 2], dtype='float64')
    t = np.atleast_2d(np.loadtxt(src, dtype='float64'))
    if len(t) < lay.ntheta + 3:
        log(f'  plane {k + 1}: shift table has {len(t)} rows, padding '
            f'{lay.ntheta + 3 - len(t)} zero rows to reach ntheta+3')
        t = np.vstack([t, np.zeros([lay.ntheta + 3 - len(t), 2])])
    return t[:lay.ntheta + 3]


class Plane:
    """Flat-fielded -log frames of one distance plane, any layout flavour.

    estimate_center.Scan does the same thing but reads EDFs directly, which the
    nxvds flavour does not have; everything here goes through Layout's
    flavour-agnostic readers instead.
    """

    def __init__(self, lay, k, nflat, shifts):
        self.lay, self.k, self.ntheta = lay, k, lay.ntheta
        self.shifts = np.asarray(shifts, dtype='float64')

        dk = lay.read_darks(k, nflat)
        w0 = lay.read_refs(k, 0, nflat)
        if not w0:
            raise SystemExit(f'estimate_quali_motion: no flat frames for plane {k + 1}')
        self.dark = (np.mean(dk, axis=0).astype('float32') if dk
                     else np.zeros(w0[0].shape, dtype='float32'))
        self.ref0 = np.mean(w0, axis=0).astype('float32') - self.dark
        # The two ends of the scan are compared against each other here, so the
        # flat is interpolated rather than taken from the start of the scan as
        # step 2 does: it leaves less detector-fixed residual at theta=180.
        w1 = lay.read_refs(k, lay.ntheta, nflat)
        self.ref1 = (np.mean(w1, axis=0).astype('float32') - self.dark
                     if w1 else self.ref0)
        self.n = self.dark.shape[0]

    def frame(self, j):
        # Clamped: a retake sits past ntheta and would otherwise extrapolate the
        # flat a little way past the end of the scan.
        w = min(j, self.ntheta) / self.ntheta
        img = np.asarray(self.lay.read_proj(self.k, j), dtype='float32') - self.dark
        img = img / ((1 - w) * self.ref0 + w * self.ref1 + 1e-3)
        return -np.log(np.clip(img, 1e-3, None))

    def commanded(self, scan_j, retake):
        """Commanded difference r_j - r_retake, (row, col), in detector px.

        This is what is subtracted from the measured lag.  The tables are
        (x, y) per row, hence the index swap.
        """
        s = self.shifts
        return (s[scan_j, 1] - s[retake, 1], s[scan_j, 0] - s[retake, 0])


def find_retakes(lay, k, log=print):
    """[(retake index, in-scan index, omega)] for plane k.

    The index convention -- ntheta+1 is omega 90 and ntheta+2 is omega 0 -- is
    the same on all eight scans in this sweep, so it leads; the angle array is
    read as a CHECK when the flavour has one, rather than being relied on.
    """
    ntheta = lay.ntheta
    try:
        ang = lay.angles(k)
    except Exception:
        ang = None
    nproj = len(ang) if ang is not None else None

    out = []
    for jr, want in ((ntheta + 1, 90.0), (ntheta + 2, 0.0)):
        if nproj is not None and jr >= nproj:
            log(f'  plane {k + 1}: no frame {jr} (only {nproj} projections) -- '
                f'the omega={want:.0f} retake is missing')
            continue
        if ang is not None:
            got = float(ang[jr])
            if abs(abs(got) - want) > 1.0:
                raise SystemExit(
                    f'estimate_quali_motion: plane {k + 1} frame {jr} is at '
                    f'omega={got:.2f}, expected {want:.0f}.  The retake tail of '
                    f'this scan is not [180, 90, 0]; check the scan before '
                    f'trusting any drift measured from it.')
        # Pick the in-scan partner from the angle array when there is one;
        # the ramp is the fallback for a flavour that records no angles.
        if ang is not None:
            js = int(np.argmin(np.abs(np.asarray(ang[:ntheta], dtype='float64')
                                      - want)))
        else:
            js = int(round(want / 180.0 * ntheta))
        out.append((jr, js, want))
    return out


def scan_angles(lay, k, log=print):
    """Omega of each of the ntheta scan frames, degrees.

    Used only by the horizontal model.  The dataset's own angles are preferred;
    a scan whose flavour has none falls back to an even 0..180 ramp.
    """
    try:
        ang = np.asarray(lay.angles(k), dtype='float64')[:lay.ntheta]
    except Exception:
        ang = None
    if ang is None or len(ang) < lay.ntheta:
        log(f'  plane {k + 1}: no angle array, assuming an even 0..180 ramp')
        return np.linspace(0.0, 180.0, lay.ntheta, endpoint=False)
    return ang


# --- correlation -----------------------------------------------------------

def masked_peak(cc, search, zero_mask):
    """Sub-pixel peak within `search` px of zero lag, zero-lag disc optional.

    Returns (dy, dx, height, zero_height).  The detector-fixed illumination
    correlates at zero lag -- the frames are never pre-shifted and the commanded
    difference is zero at every point measured here -- so its height is always
    reported even when it is not masked.
    """
    ny, nx = cc.shape
    r = int(min(search, ny // 2 - 2, nx // 2 - 2))
    idx = np.arange(-r, r + 1)
    sub = cc[np.ix_(idx % ny, idx % nx)].astype('float64').copy()

    disc = idx[:, None] ** 2 + idx[None, :] ** 2 <= max(zero_mask, 1) ** 2
    zpk = float(sub[disc].max())
    if zero_mask > 0:
        sub[disc] = sub.min()

    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    h = float(sub[i, j])
    off = []
    for v, p in ((sub[:, j], i), (sub[i, :], j)):
        if 0 < p < len(v) - 1:
            a0, a1, a2 = v[p - 1], v[p], v[p + 1]
            den = a0 - 2 * a1 + a2
            off.append(p + (0.5 * (a0 - a2) / den if den else 0.0))
        else:
            off.append(float(p))
    return off[0] - r, off[1] - r, h, zpk


def profile_lag(a, b, search, zero_mask):
    """Unwhitened 1-D lags (dy, dx) from the collapsed profiles.

    An independent check that fails differently from the 2-D peak: collapsing
    to a profile averages away the detector pattern's fine structure but keeps
    the sample's, so a run where the 2-D surface has gone to zero lag shows up
    as a disagreement here.  This is the check that was right -- it read -104
    while the whitened 2-D peak read +0.15.
    """
    out = []
    for axis in (1, 0):
        pa, pb = a.mean(axis=axis), b.mean(axis=axis)
        pa, pb = pa - pa.mean(), pb - pb.mean()
        na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
        if na == 0 or nb == 0:
            out.append(float('nan'))
            continue
        cc = np.real(np.fft.ifft(np.fft.fft(pa / na) * np.conj(np.fft.fft(pb / nb))))
        n = len(cc)
        rr = int(min(search, n // 2 - 2))
        kk = np.arange(-rr, rr + 1)
        w = cc[kk % n].copy()
        if zero_mask > 0:
            w[np.abs(kk) <= zero_mask] = w.min()
        out.append(float(kk[int(np.argmax(w))]))
    return out[0], out[1]


def measure_point(pl, static, scan_j, retake, args, log=print):
    """Drift at frame scan_j, in this plane's detector px.

    Several crops, combined by median; the scatter across them is the error
    bar.  A single crop can lock onto a wrong local peak, and the median of
    crops that disagree is caught by --max-scatter rather than used.
    """
    A = pl.frame(scan_j) - static        # a: the in-scan frame
    B = pl.frame(retake) - static        # b: the retake
    A = _band(A, args.highpass, args.smooth)
    B = _band(B, args.highpass, args.smooth)
    cy, cx = pl.commanded(scan_j, retake)
    n = pl.n

    est = []
    for crop in args.crop:
        if crop > n:
            continue
        lo, hi = (n - crop) // 2, (n + crop) // 2
        a, b = A[lo:hi, lo:hi], B[lo:hi, lo:hi]
        cc = _corr(a, b)
        # The exact zero-lag bin is unusable and is interpolated over.  Any
        # WHITE noise shared between the two frames lands entirely in cc[0,0],
        # and here it is shared with a MINUS sign -- after subtracting a
        # template that averages over the whole scan, the frame at the start
        # carries +1/2 log(ref0) - 1/2 log(ref1) of flat noise and the retake at
        # the end carries exactly the negative of it.  So it digs a one-pixel
        # hole rather than adding a peak, deep enough on a weak pair to invert
        # it and send the peak finder to a ring lobe.  The sample peak is
        # several px wide, so replacing the centre bin costs nothing real.
        cc[0, 0] = 0.25 * (cc[0, 1] + cc[0, -1] + cc[1, 0] + cc[-1, 0])
        dy, dx, pk, zpk = masked_peak(cc, args.search, args.zero_mask)
        py, px = profile_lag(a, b, args.search, args.zero_mask)
        # Peter's arithmetic subtraction (quali_series.m:562): the commanded
        # move is taken off the ANSWER, the images are never shifted to undo it.
        est.append(dict(crop=crop, dy=dy - cy, dx=dx - cx, peak=pk, zpeak=zpk,
                        py=py - cy, px=px - cx, cc=cc))
    if not est:
        raise SystemExit(f'estimate_quali_motion: every crop in {args.crop} is '
                         f'larger than the {n} px frame')
    return est, (cy, cx)


# --- the fit ---------------------------------------------------------------

def fit_drift(pts_j, pts_d, pts_th, ntheta, th_all, log=print):
    """correct_motion_calc.m's model, in our sign.  Returns (curve, raw).

    `arg = j/ntheta - 1` is zero at the END of the scan, and neither component
    carries a constant term, which is what pins the omega=180 anchor without
    spending a fit point on it.  Peter builds the same matrices with a leading
    minus (`M = start_reference - powerfit(...)`) because his ref_v/ref_h are
    the NEGATION of the drift; ours is the drift, so the minus is simply absent.

    `raw` is the model before the mean / sinusoid is taken out, so the figure
    can place the measured points against the curve it actually drew.
    """
    npts = len(pts_j)
    arg_p = np.asarray(pts_j, dtype='float64') / ntheta - 1.0
    arg_a = np.arange(ntheta, dtype='float64') / ntheta - 1.0
    curve = np.zeros([ntheta, 2])
    raw = np.zeros([ntheta, 2])

    # Vertical: a curve in time, nothing more -- vertical drift does not care
    # which way the sample is facing.
    deg_v = min(npts, 3)
    Mv = np.vstack([arg_p ** d for d in range(deg_v, 0, -1)])
    pv = np.linalg.lstsq(Mv.T, np.asarray(pts_d)[:, 0], rcond=None)[0]
    raw[:, 0] = np.polyval(np.append(pv, 0.0), arg_a)
    curve[:, 0] = raw[:, 0] - raw[:, 0].mean()

    # Horizontal: a 2-D drift in the lab frame, seen in projection.  Two time
    # curves sx, sy of degree min(npts//2, 3), combined with cos/sin(theta).
    deg_h = max(1, min(npts // 2, 3))
    if 2 * deg_h > npts:
        log(f'    WARNING: {npts} point(s) cannot determine a {2 * deg_h}-'
            f'parameter horizontal model; it is least-squares, not exact')
    Mh = np.vstack([arg_p ** d for d in range(deg_h, 0, -1)])
    cp, sp = np.cos(np.deg2rad(pts_th)), np.sin(np.deg2rad(pts_th))
    M = np.vstack([Mh * cp, Mh * sp])
    ph = np.linalg.lstsq(M.T, np.asarray(pts_d)[:, 1], rcond=None)[0]
    sx = np.polyval(np.append(ph[:deg_h], 0.0), arg_a)
    sy = np.polyval(np.append(ph[deg_h:], 0.0), arg_a)
    ca, sa = np.cos(np.deg2rad(th_all)), np.sin(np.deg2rad(th_all))
    raw[:, 1] = sx * ca + sy * sa
    # A pure sinusoid in theta is a rotation-axis shift, which step 4b owns.
    B = np.vstack([ca, sa])
    q = np.linalg.lstsq(B.T, raw[:, 1], rcond=None)[0]
    curve[:, 1] = raw[:, 1] - q @ B
    return curve, raw, (deg_v, deg_h)


# --- the measurement -------------------------------------------------------

def measure_motion(lay, args, log=print):
    """Measure the drift of every plane.  Returns a dict; writes nothing."""
    ndist, ntheta = lay.ndist, lay.ntheta
    nmag = lay.geometry()['norm_magnifications']

    motion = np.zeros([ntheta, ndist, 2], dtype='float64')
    band = np.zeros([ntheta, ndist, 2], dtype='float64')
    per_plane = []

    for k in range(ndist):
        log(f'\nplane {k + 1} of {ndist}')
        sh = shift_table(lay, k, log)
        retakes = find_retakes(lay, k, log)
        if not retakes:
            raise SystemExit(
                f'estimate_quali_motion: plane {k + 1} has no post-scan retake '
                f'frames after {ntheta}.  Without them there is nothing to '
                f'measure the drift against.')
        pl = Plane(lay, k, args.nflat, sh)
        th_all = scan_angles(lay, k, log)

        # Static illumination template.  Spread over the scan so the sample
        # smears over its commanded displacement while the illumination does
        # not, and with the compared frames held out (see the cc[0,0] note).
        static = np.zeros((pl.n, pl.n), dtype='float32')
        if args.template > 0:
            used = {js for _, js, _ in retakes} | {jr for jr, _, _ in retakes} | {0, ntheta}
            tj = []
            for j in np.linspace(0, ntheta, args.template, dtype=int):
                j = int(j)
                while j in used or j in tj:
                    j += 7 if j < ntheta - 7 else -7
                tj.append(j)
            log(f'  template: mean of {len(tj)} frames, excluding {sorted(used)}')
            for j in tj:
                static += pl.frame(j)
            static /= len(tj)

        log(f'  {"retake":>7} {"frame":>6} {"crop":>5} {"dy":>8} {"dx":>8} '
            f'{"peak":>7} | {"1-D dy":>7} {"dx":>7} | {"zero pk":>8}')
        pts_j, pts_d, pts_th, sd, panels, notes = [], [], [], [], {}, []
        for jr, js, omega in retakes:
            est, cmd = measure_point(pl, static, js, jr, args, log)
            for e in est:
                log(f'  {jr:7d} {js:6d} {e["crop"]:5d} {e["dy"]:+8.2f} '
                    f'{e["dx"]:+8.2f} {e["peak"]:7.4f} | {e["py"]:+7.1f} '
                    f'{e["px"]:+7.1f} | {e["zpeak"]:8.4f}')
            dy = float(np.median([e['dy'] for e in est]))
            dx = float(np.median([e['dx'] for e in est]))
            pk = float(np.median([e['peak'] for e in est]))
            zpk = float(np.median([e['zpeak'] for e in est]))
            udy = float(np.median(np.abs([e['dy'] - dy for e in est]))) * 1.4826
            udx = float(np.median(np.abs([e['dx'] - dx for e in est]))) * 1.4826
            log(f'  {"":7} {"":6} {"median":>5} {dy:+8.2f} {dx:+8.2f} {pk:7.4f}'
                f'   +- {udy:.2f} / {udx:.2f} px across crops')
            if any(cmd):
                log(f'  {"":7} commanded difference subtracted: '
                    f'({cmd[0]:+.2f}, {cmd[1]:+.2f}) px')

            if pk < args.min_peak:
                notes.append(f'omega={omega:.0f}: peak {pk:.4f} is below '
                             f'min_peak {args.min_peak}; a pair that locked on '
                             f'gives ~0.02')
                log(f'  {"":7} WARNING: {notes[-1]}')
            # The illumination lives at zero lag.  If it outranks the peak that
            # was taken, the measurement is the thing that failed before.
            if zpk >= pk and args.zero_mask <= 0:
                notes.append(f'omega={omega:.0f}: zero lag is as tall as the '
                             f'chosen peak ({zpk:.4f} vs {pk:.4f}) -- the '
                             f'detector-fixed illumination may have won; '
                             f'compare the 1-D column and consider --zero-mask')
                log(f'  {"":7} WARNING: {notes[-1]}')
            py = float(np.median([e['py'] for e in est]))
            px = float(np.median([e['px'] for e in est]))
            if abs(py - dy) > args.profile_tol or abs(px - dx) > args.profile_tol:
                notes.append(f'omega={omega:.0f}: the unwhitened 1-D profile says '
                             f'({py:+.1f}, {px:+.1f}) against the 2-D peak '
                             f'({dy:+.2f}, {dx:+.2f})')
                log(f'  {"":7} WARNING: {notes[-1]}')

            # A point whose crops disagree wildly is dropped rather than
            # medianed: with three crops reading -0.4 / -93.4 / +99.3 the median
            # is the one that happens to sit in the middle, not the right one.
            if max(udy, udx) > args.max_scatter:
                notes.append(f'omega={omega:.0f}: DROPPED, crops scatter by '
                             f'{udy:.1f} / {udx:.1f} px (limit '
                             f'{args.max_scatter}); the fit degrades a degree')
                log(f'  {"":7} WARNING: {notes[-1]}')
                continue

            pts_j.append(float(js))
            pts_d.append([dy, dx])
            pts_th.append(float(omega))
            sd.append([udy, udx])
            panels[js] = (est[len(est) // 2]['cc'], omega, dy, dx, pk)

        # NOTE there is no point appended for frame ntheta.  Its drift is zero
        # by construction and the no-constant-term fit enforces that exactly,
        # so adding it as a data point would waste a degree of freedom.
        pts_j, pts_d = np.array(pts_j), np.array(pts_d)
        pts_th, sd = np.array(pts_th), np.array(sd)
        if len(pts_j) < 1:
            raise SystemExit(
                f'estimate_quali_motion: plane {k + 1} kept no usable retake '
                f'point; there is nothing to fit.')
        if len(pts_j) < 2:
            log(f'  WARNING: only {len(pts_j)} usable point -- the horizontal '
                f'model is underdetermined and the vertical drops to degree 1')

        curve, raw, (deg_v, deg_h) = fit_drift(pts_j, pts_d, pts_th, ntheta,
                                               th_all, log)
        motion[:, k] = curve
        # The fit runs exactly through its points, so each point's uncertainty
        # moves the whole curve; refit with each pushed by its own MAD and keep
        # the envelope.
        for c in range(2):
            for i in range(len(pts_j)):
                if not sd[i, c]:
                    continue
                v = pts_d.copy()
                v[i, c] += sd[i, c]
                alt = fit_drift(pts_j, v, pts_th, ntheta, th_all, lambda *a: None)[0]
                band[:, k, c] = np.maximum(band[:, k, c],
                                           np.abs(alt[:, c] - curve[:, c]))

        # Detector px of this plane -> the object grid cshifts_final lives on.
        motion[:, k] /= nmag[k]
        band[:, k] /= nmag[k]
        raw = raw / nmag[k]
        pts_d = pts_d / nmag[k]
        sd = sd / nmag[k]

        log(f'  fitted drift, object px, degree {deg_v} vertical / {deg_h} '
            f'horizontal (rotating frame):')
        for c, nm in ((0, 'row (y, vertical)  '), (1, 'col (x, horizontal)')):
            amp = np.ptp(motion[:, k, c])
            log(f'    {nm}  ptp {amp:7.3f}  rms {motion[:, k, c].std():7.4f}'
                f'   +- {band[:, k, c].max():.3f} from the retake error bars')
            if amp and sd.size and amp < sd[:, c].max():
                log(f'    WARNING: that amplitude is below the '
                    f'{sd[:, c].max():.2f} px scatter -- treat it as unmeasured')

        per_plane.append(dict(k=k, pts_j=pts_j, pts_d=pts_d, pts_th=pts_th,
                              sd=sd, raw=raw, deg=(deg_v, deg_h),
                              panels=panels, notes=notes, shifts=sh))

    return dict(motion=motion, band=band, planes=per_plane,
                norm_magnifications=nmag, ntheta=ntheta, ndist=ndist)


def write_measured(lay, out_path, log=print, **kw):
    """Measure the drift and write out_path as .npy.  The call steps15 makes.

    out_path is under path_out, never in the raw scan directory -- a measurement
    is ours, ESRF's correct_motion.txt is theirs.
    """
    args = options(**kw)
    res = measure_motion(lay, args, log=log)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, res['motion'].astype('float32'))
    m = res['motion']
    log(f'wrote {out_path}  {m.shape}')
    log(f'  per-plane ptp, object px:  y={np.round(np.ptp(m[:, :, 0], axis=0), 2)}'
        f'  x={np.round(np.ptp(m[:, :, 1], axis=0), 2)}')
    return res


# --- validation against ESRF's own numbers ---------------------------------

def validate(lay, res, ref_dist=0, motion_bin=2, log=print):
    """Compare with reference_motion.mat and quali.mat, when the scan has them.

    reference_motion.mat is the better of the two: ref_v/ref_h are what Peter's
    correct_motion.txt is literally built from (correct_motion = random_disp -
    (ref_h, ref_v), to 0.0000 px), so our term is their NEGATION.  A correct run
    therefore shows corr = +1 against -ref.

    THE SCALE IS REPORTED, NOT ASSUMED.  His grid is 2x2-binned against ours, so
    the amplitudes should differ by motion_bin; rather than apply that and hope,
    the least-squares scale ours/theirs is printed.  It landing on 1 is a real
    check -- 0.5 or 2 instead would mean the binning is handled wrongly
    somewhere, and corr stays +1 either way.
    """
    from holotomocupy.reader import load_octave_text_mat
    out = {}
    ours = res['motion']

    p = f'{lay.path}/{lay.pfile}_/reference_motion.mat'
    if not os.path.exists(p):
        log(f'\nno {p}: nothing to validate the curve against')
    else:
        try:
            rv = np.asarray(load_octave_text_mat(p, 'ref_v'), dtype='float64').ravel()
            rh = np.asarray(load_octave_text_mat(p, 'ref_h'), dtype='float64').ravel()
        except (KeyError, ValueError) as e:
            log(f'\n{os.path.basename(p)}: could not read ref_v/ref_h ({e})')
            rv = rh = None
        if rv is not None and len(rv) >= lay.ntheta:
            rv, rh = rv[:lay.ntheta], rh[:lay.ntheta]
            try:
                rp = int(np.asarray(load_octave_text_mat(p, 'reference_plane')).ravel()[0])
                kref = rp - 1
            except (KeyError, ValueError, IndexError):
                kref = ref_dist
            # Negated into our sense, mean removed, and put on our unbinned grid.
            theirs = motion_bin * np.stack([-(rv - rv.mean()),
                                            -(rh - rh.mean())], axis=1)
            out['ref'] = (kref, theirs)
            log(f'\nvalidation against reference_motion.mat, plane {kref + 1}, '
                f'object px (theirs negated, mean removed, x{motion_bin})')
            log(f'{"comp":>5} {"ours ptp":>9} {"theirs ptp":>11} '
                f'{"max|diff|":>10} {"corr":>7} {"scale":>7}')
            for c, lbl in ((0, 'y'), (1, 'x')):
                a = ours[:, kref, c] - ours[:, kref, c].mean()
                t = theirs[:, c]
                r = (np.corrcoef(a, t)[0, 1]
                     if np.std(a) > 1e-9 and np.std(t) > 1e-9 else np.nan)
                s = np.dot(a, t) / np.dot(t, t) if np.dot(t, t) > 0 else np.nan
                log(f'{lbl:>5} {np.ptp(a):>9.2f} {np.ptp(t):>11.2f} '
                    f'{np.max(np.abs(a - t)):>10.2f} {r:>7.3f} {s:>7.3f}')
            log(f'  corr near +1 means the sign convention matches; near -1 means '
                f'this file is writing the drift backwards.  scale near 1 means '
                f'the x{motion_bin} is right.')

    # Every plane has its own quali.mat -- on HT that is four independent
    # measurements of the same quantity, not one, so all of them are compared.
    log(f'\nquali.mat, per point, OBJECT px (his x{motion_bin} / nmag[k]).')
    log(f'{"plane":>5} {"omega":>5} {"their y":>9} {"our y":>9} {"dy":>7} '
        f'{"their x":>9} {"our x":>9} {"dx":>7}')
    any_q = False
    for k in range(res['ndist']):
        p = f'{lay.path}/{lay.pfile}_{k + 1}_/quali.mat'
        if not os.path.exists(p):
            continue
        q = {}
        for key in ('corr_imagesafterscan', 'corrtot', 'corrtot90'):
            try:
                q[key] = np.asarray(load_octave_text_mat(p, key),
                                    dtype='float64').ravel()
            except (KeyError, ValueError):
                continue
        if not q:
            continue
        any_q = True
        if k == ref_dist:
            out['quali'] = q
        nmag = res['norm_magnifications'][k]
        pts = res['planes'][k]
        # corrtot is the omega-0 point, corrtot90 the omega-90 one; his rows
        # are (x, y) and ours are (y, x).
        theirs = {0.0: q.get('corrtot'), 90.0: q.get('corrtot90')}
        for om, t in theirs.items():
            if t is None or t.size < 2:
                continue
            ty, tx = t[1] * motion_bin / nmag, t[0] * motion_bin / nmag
            sel = np.where(np.abs(pts['pts_th'] - om) < 1.0)[0]
            if not len(sel):
                log(f'{k + 1:>5} {om:>5.0f} {ty:>9.3f} {"dropped":>9} '
                    f'{"":>7} {tx:>9.3f} {"dropped":>9}')
                continue
            oy, ox = pts['pts_d'][sel[0]]
            log(f'{k + 1:>5} {om:>5.0f} {ty:>9.3f} {oy:>9.3f} {oy - ty:>+7.2f} '
                f'{tx:>9.3f} {ox:>9.3f} {ox - tx:>+7.2f}')
    if not any_q:
        log('  none of the planes has a quali.mat -- nothing to compare')
    else:
        log('  ours is before mean removal, as his is.  A dy of a few tenths '
            'is the crop scatter; a sign flip or a factor 2 is not.')
    return out or None


# --- figure ----------------------------------------------------------------

def make_figure(path, res, validation=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ndist = res['ndist']
    jj = np.arange(res['ntheta'])
    fig, ax = plt.subplots(ndist, 4, figsize=(16, 3.4 * ndist), squeeze=False)

    R = 28
    for k in range(ndist):
        pp = res['planes'][k]
        panels = sorted(pp['panels'].items())
        for col in range(2):
            a = ax[k][col]
            if col >= len(panels):
                a.axis('off')
                continue
            js, (cc, omega, dy, dx, pk) = panels[col]
            idx = np.arange(-R, R + 1)
            sub = cc[np.ix_(idx % cc.shape[0], idx % cc.shape[1])]
            a.imshow(sub, cmap='inferno', extent=[-R, R, R, -R],
                     interpolation='nearest')
            a.plot(dx, dy, 'o', mfc='none', mec='cyan', ms=13, mew=1.8)
            a.axhline(0, color='w', lw=0.5, alpha=0.4)
            a.axvline(0, color='w', lw=0.5, alpha=0.4)
            a.set_title(f'plane {k + 1}, omega={omega:.0f} vs frame {js}\n'
                        f'dy={dy:+.2f} dx={dx:+.2f} px, height {pk:.4f}',
                        fontsize=9)
            a.set_xlabel('dx [px]', fontsize=8)
            a.set_ylabel('dy [px]', fontsize=8)
            a.tick_params(labelsize=7)

        for c, (lbl, col) in enumerate((('row (y, vertical)', 'tab:red'),
                                        ('col (x, horizontal)', 'tab:blue'))):
            a = ax[k][2 + c]
            m, b = res['motion'][:, k, c], res['band'][:, k, c]
            a.fill_between(jj, m - b, m + b, color=col, alpha=0.2, lw=0,
                           label='retake error bars')
            a.plot(jj, m, color=col, lw=1.6,
                   label=f'degree-{pp["deg"][c]} fit')
            # The curve has had its mean (vertical) or its sinusoid in theta
            # (horizontal) removed, so the measured points are moved by the
            # same amount at their own frame index before being drawn.
            ji = np.clip(pp['pts_j'].astype(int), 0, res['ntheta'] - 1)
            a.plot(pp['pts_j'], pp['pts_d'][:, c] - (pp['raw'][ji, c] - m[ji]),
                   'ko', ms=7, label='measured retakes', zorder=5)
            if validation and 'ref' in validation and validation['ref'][0] == k:
                a.plot(jj, validation['ref'][1][:, c], 'k--', lw=1.2,
                       label='ESRF reference_motion')
            a.axhline(0, color='0.7', lw=0.7)
            a.set_title(f'plane {k + 1}: {lbl}', fontsize=10)
            a.set_xlabel('projection index')
            a.set_ylabel('object px')
            a.legend(fontsize=7)
            a.grid(alpha=0.3)

    fig.suptitle('Sample drift from the post-scan quality retakes', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', nargs='?', help='config_steps15.conf')
    p.add_argument('--path', help='override the config path')
    p.add_argument('--pfile', help='override the config pfile')
    p.add_argument('--ref-dist', type=int, dest='ref_dist',
                   help="reference plane, 0-based; default is the config's")
    p.add_argument('--crop', default=','.join(str(c) for c in DEFAULTS['crop']),
                   type=lambda v: tuple(int(x) for x in v.split(',')),
                   help='central crops, bin-0 px; each gives one estimate and '
                        'the median is used, so this wants several values')
    p.add_argument('--search', type=int, default=DEFAULTS['search'],
                   help='peak searched within this many px of zero lag')
    p.add_argument('--zero-mask', type=int, dest='zero_mask',
                   default=DEFAULTS['zero_mask'],
                   help='blank a disc of this radius at zero lag, where the '
                        'detector-fixed illumination correlates; OFF by '
                        'default because it would erase a small real drift')
    p.add_argument('--smooth', type=float, default=DEFAULTS['smooth'],
                   help='Gaussian low-pass sigma in detector px before '
                        'correlating')
    p.add_argument('--highpass', type=float, default=DEFAULTS['highpass'])
    p.add_argument('--template', type=int, default=DEFAULTS['template'],
                   help='frames averaged into the static template; 0 disables')
    p.add_argument('--nflat', type=int, default=DEFAULTS['nflat'])
    p.add_argument('--min-peak', type=float, dest='min_peak',
                   default=DEFAULTS['min_peak'])
    p.add_argument('--max-scatter', type=float, dest='max_scatter',
                   default=DEFAULTS['max_scatter'],
                   help='drop a point whose crops disagree by more than this')
    p.add_argument('--profile-tol', type=float, dest='profile_tol',
                   default=DEFAULTS['profile_tol'])
    p.add_argument('--motion-bin', type=float, dest='motion_bin', default=2.0,
                   help="binning Peter's quali.mat / reference_motion.mat are "
                        'on, relative to ours; --validate only')
    p.add_argument('--out', help='default <path_out>/measured/motion_measured.npy')
    p.add_argument('--fig', default='quali_motion_estimate.png')
    p.add_argument('--validate', action='store_true',
                   help='compare with reference_motion.mat and quali.mat')
    args = p.parse_args()

    path, pfile, ref_dist, out = args.path, args.pfile, args.ref_dist, args.out
    if args.config:
        from holotomocupy.config import parse_args_steps15
        cfg = parse_args_steps15(args.config)
        path = path or cfg.path
        pfile = pfile or cfg.pfile
        ref_dist = cfg.ref_dist if ref_dist is None else ref_dist
        out = out or f'{cfg.path_out}/measured/motion_measured.npy'
    if not (path and pfile):
        p.error('give a config, or both --path and --pfile')
    ref_dist = 0 if ref_dist is None else ref_dist

    lay = Layout(path, pfile)
    print(lay)
    res = write_measured(lay, out or 'motion_measured.npy',
                         crop=args.crop, search=args.search,
                         zero_mask=args.zero_mask, smooth=args.smooth,
                         highpass=args.highpass, template=args.template,
                         nflat=args.nflat, min_peak=args.min_peak,
                         max_scatter=args.max_scatter,
                         profile_tol=args.profile_tol)
    v = validate(lay, res, ref_dist, args.motion_bin) if args.validate else None
    make_figure(args.fig, res, v)
    print(f'wrote {args.fig}')

    notes = [n for pp in res['planes'] for n in pp['notes']]
    if notes:
        print(f'\n{len(notes)} warning(s) were raised during the measurement; '
              f'read them before using this.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
