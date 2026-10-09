#!/usr/bin/env python
"""
Measure the inter-plane registration residual -- our own rhapp.

    python estimate_rhapp.py config_steps15.conf [--validate]

Step 3 of steps15.py calls `write_measured()` itself whenever ndist > 1 --
there is no knob.  Running this as a script is for checking the answer
(--validate compares with ESRF's rhapp.mat), not for feeding the pipeline.

WHAT rhapp IS.  Step 4 resamples every distance plane onto one object grid --
divide by the plane's effective magnification, then shift by cshifts_final --
and the planes have to land on top of each other to within a fraction of a
pixel or step 6 fits a blurred object.  The commanded random displacement gets
most of the way there, but each plane sits at its own piezo position and the
stage is not perfectly repeatable, so a per-angle per-plane residual is left
over.  ESRF's Octave pipeline measures it and writes rhapp.mat; this measures
the same thing from the frames.

WHY WE MEASURE IT OURSELVES.  rhapp.mat only exists for the scans Peter's
pipeline has been run on -- AtomiumS1 HT RD300 and RD025 have one, HT RD100
does not -- and at ndist=4 dropping the term to zeros is not survivable.
Measuring it here makes every RD amplitude behave the same way.

UNITS.  The output is in the OBJECT-GRID pixels of cshifts_final: plane k's
frame resampled by 1/norm_magnifications[k], which at plane 0 is the raw
detector pitch.  That is the same grid `random_shifts` and `motion_shifts` are
in, so step 3 adds the array with no scaling -- unlike rhapp.mat, which is on
Peter's 2x2-binned grid and has to be multiplied by rhapp_bin.

HOW BIG IT IS.  Not a small residual: on HT RD300 the plane-to-plane offset
left after the commanded displacement is removed is 270-390 OBJECT px, with
only ~13 px of variation over the scan riding on top.  That number drove two of
the four design choices below, and getting it wrong is what made the first
version of this file return noise.

METHOD, and why each piece is there.
 1. Sample --nangles angles spread over the scan rather than all ntheta.  Peter's
    rhapp is EXACTLY a degree-7 polynomial in angle (residual rms 0.0000 for
    every plane and component on RD300), so the curve is fully determined by a
    sparse sample and the fit averages the per-angle correlation noise down.
 2. Flat-field, bin to the --bin grid, resample plane k by 1/norm_mag[k] onto
    the common grid, and undo that plane's OWN commanded displacement.  Each
    plane has an independent random sequence -- the four tables do not
    correlate -- so comparing plane k with the reference at the same angle
    without undoing both leaves hundreds of px of pure noise on top.
    Both operations are step 4's `Shift.curlySback`, not a local copy, so what
    is measured is the residual step 4's own resampling leaves.  That makes
    this file need a GPU; it was scipy affine_transform + a Fourier shift until
    2026-10-07, which interpolated linearly and replicated at the border where
    curlySback is cubic and zero-fills.
 3. LOW-pass, never high-pass, and that is measured rather than assumed.  The
    planes are holograms at different propagation distances: their Fresnel
    fringes do not match, and what they share is the LOW-frequency content
    below the first CTF zero of the most-propagated plane.  A high-pass keeps
    exactly the half that disagrees.  Swept on HT RD300 against rhapp.mat, rms
    error in binned px: no filter 1.68 (peak 0.49), --smooth 4 0.48 (0.81),
    --smooth 8 0.42 (0.91), --highpass 8 3.71 (0.14).  --highpass is kept only
    so the sweep can be repeated; it has no setting that helps.
 4. Two-stage search.  Stage 1 searches the whole frame at --ncoarse angles and
    takes the median lag per plane; stage 2 searches only +-`search` around it.
    A narrow window centred on zero would never reach the true 70-100 binned px
    offset, and a wide window per angle invites a mis-lock -- which is the
    failure the first version of this file actually hit.
 5. Parabolic sub-pixel interpolation on the correlation peak, then the
    degree-7 fit in angle.

READ THE VALIDATION BEFORE TRUSTING A NUMBER.  `--validate` compares the fit
with rhapp.mat on a scan that has one.  A correlation estimator on this sample
has already been caught returning a confident zero -- the retake-based drift
search, since removed -- so an unvalidated run of this file means nothing.

Validated on HT RD300, ref_dist=2, --nangles 40 at the defaults: correlation
against rhapp.mat 0.875-0.998, max|diff| 1.05-4.06 object px, per-angle fit
residual 1.46-1.87 object px.  For scale, ESRF's own per-angle scatter about
their own fitted curve is ~1.77 object px (rhappnofit.mat, res_shift_sigma_*
at their 9 nm pixel), so this is at parity per angle.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esrf_layout import Layout

DEFAULTS = dict(
    path=None, pfile=None, ref_dist=0,
    nangles=64,        # angles sampled across the scan
    bin=2,             # correlate on the 4096/2**bin grid
    ncoarse=4,         # angles used for the stage-1 full-frame search
    search=16,         # stage-2 radius about the stage-1 centre, binned px
    deg=7,             # polynomial degree in angle; Peter's rhapp is exactly 7
    highpass=0.0,      # trend removed, Gaussian sigma in binned px (0 = none)
    smooth=8.0,        # pre-smoothing, Gaussian sigma in binned px (0 = none)
    nflat=8,           # flats averaged per plane
    # Rejects dead angles only.  Height does NOT rank settings: at smooth=32 it
    # is still 0.88 while the error is 4x the optimum.  Do not tune on it.
    min_peak=0.15,     # correlation height below this -> angle dropped
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


# --- small numeric helpers -------------------------------------------------

def _bin2(a, b):
    """Average-bin a 2-D array by 2**b in both directions."""
    for _ in range(b):
        a = 0.25 * (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2])
    return a


def _band(a, highpass, smooth):
    """Low-pass on, high-pass off, and both are measured choices.

    The planes are holograms at different propagation distances, so their
    Fresnel fringes do not match; what they share is the low-frequency content
    below the first CTF zero of the most-propagated plane.  A high-pass keeps
    the part that disagrees: on HT RD300 a sigma-6 high-pass drops the
    correlation peak from ~0.50 to ~0.06 and the lag becomes noise.  Smoothing
    instead is worth 4x: rms error against rhapp.mat goes 1.68 -> 0.42 binned
    px from sigma 0 to 8, flat over 8-12, and back to 1.61 by 32.
    """
    if highpass <= 0 and smooth <= 0:
        return a
    from scipy.ndimage import gaussian_filter
    if smooth > 0:
        a = gaussian_filter(a, smooth)
    if highpass > 0:
        a = a - gaussian_filter(a, highpass)
    return a


def _resample(cl, stack, shifts, nmag):
    """Step 4's own resampler: magnify about the centre and shift, in one call.

    `Shift.curlySback` samples at `(t - c_out + r)/m + c_in` per axis, so
    m = 1/nmag[k] with r the plane's own commanded displacement reproduces what
    step 4 does to the same frame -- which is the point, since rhapp is the
    residual step 4's resampling leaves behind.  Note r enters UNNEGATED: the
    kernel already translates by -r.

    All ndist planes of one angle go in a single batched call; `cl` is built
    once by the caller because the constructor allocates cuFFT plans.
    """
    import cupy as cp
    m = cp.asarray(np.repeat((1.0 / nmag)[:, None], 2, axis=1), dtype='float32')
    r = cp.asarray(shifts, dtype='float32')
    out = cl.curlySback(cp.asarray(stack, dtype='complex64'), r, m)
    return cp.asnumpy(out.real)


def _peak(cc, search, c0=(0, 0)):
    """Brightest lag within +-search of c0, parabolic sub-pixel, plus height.

    Returns (dy, dx, height) with the lag in the usual cross-correlation
    sense: `b` sits at +(dy, dx) relative to `a`.  c0 exists because the
    inter-plane offset here is 270-390 object px, far outside any sane
    per-angle window -- stage 1 finds it, stage 2 searches around it.
    """
    cc = np.roll(cc, (-int(round(c0[0])), -int(round(c0[1]))), axis=(0, 1))
    n = cc.shape[0]
    s = min(search, n // 2 - 2)
    w = np.fft.fftshift(cc)[n // 2 - s:n // 2 + s + 1, n // 2 - s:n // 2 + s + 1]
    i, j = np.unravel_index(np.argmax(w), w.shape)
    h = float(w[i, j])
    sub = []
    for v, idx in ((w[:, j], i), (w[i, :], j)):
        if 0 < idx < len(v) - 1:
            a0, a1, a2 = v[idx - 1], v[idx], v[idx + 1]
            den = a0 - 2 * a1 + a2
            sub.append(idx + (0.5 * (a0 - a2) / den if den != 0 else 0.0))
        else:
            sub.append(float(idx))
    return sub[0] - s + round(c0[0]), sub[1] - s + round(c0[1]), h


def _corr(a, b):
    """Unnormalised cross-correlation, both inputs mean-removed and scaled.

    NOT phase correlation.  Whitening is what makes a 2-D correlation on these
    holograms report zero lag with total confidence -- it boosts the shared
    high-frequency detector pattern to the same weight as the sample.
    """
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.zeros_like(a)
    return np.real(np.fft.ifft2(np.fft.fft2(a / na) * np.conj(np.fft.fft2(b / nb))))


# --- the measurement -------------------------------------------------------

def measure_rhapp(lay, ref_dist, args, motion=None):
    """Measure the inter-plane residual.  Returns a dict; writes nothing.

    motion is [ntheta, ndist, 2] object px (row, col) from
    estimate_quali_motion, or None.  It MUST be passed whenever step 3 carries
    a motion term: this routine re-reads the shift tables itself rather than
    taking cshifts_final, so a drift that is not undone here is measured again
    as part of the residual and ends up counted twice.
    """
    ndist = lay.ndist
    ntheta = lay.ntheta
    geo = lay.geometry()
    nmag = geo['norm_magnifications']
    b = args.bin

    # Each plane's own commanded random displacement.  Columns are (x, y) in
    # the plane's detector px; step 3 reads the same files.
    tables = []
    for k in range(ndist):
        src = lay.shift_source(k)
        if not os.path.exists(src):
            raise SystemExit(f'estimate_rhapp: no shift table for plane {k + 1}: {src}')
        tables.append(np.loadtxt(src, dtype='float64')[:ntheta])
    tables = np.stack(tables, axis=1)                       # [ntheta, ndist, 2]

    # Same conversion step 3 makes: (x, y) detector px -> (row, col) object px.
    rand = np.empty_like(tables)
    rand[..., 0] = tables[..., 1] / nmag
    rand[..., 1] = tables[..., 0] / nmag

    # Undo the measured drift along with the commanded move, so the residual
    # left to correlate is the inter-plane offset and nothing else.
    if motion is not None:
        motion = np.asarray(motion, dtype='float64')
        if motion.shape != rand.shape:
            raise SystemExit(f'estimate_rhapp: motion is {motion.shape}, '
                             f'expected {rand.shape}')
        rand = rand + motion
        print(f'  undoing a motion term as well, ptp per plane: '
              f'{np.round(np.ptp(motion, axis=0).max(axis=1), 2)} object px')

    # Flats and darks, once per plane.
    flat, dark = [], []
    for k in range(ndist):
        dk = lay.read_darks(k, args.nflat)
        wk = lay.read_refs(k, 0, args.nflat)
        if not wk:
            raise SystemExit(f'estimate_rhapp: no flat frames for plane {k + 1}')
        dark.append(np.mean(dk, axis=0).astype('float32') if dk
                    else np.zeros(wk[0].shape, dtype='float32'))
        flat.append(np.mean(wk, axis=0).astype('float32'))

    jj = np.unique(np.linspace(0, ntheta - 1, args.nangles).astype(int))
    est = np.full([len(jj), ndist, 2], np.nan)
    hgt = np.zeros([len(jj), ndist])
    ncoarse = min(args.ncoarse, len(jj))
    c0 = {k: (0, 0) for k in range(ndist)}
    held = []          # stage-1 correlation surfaces, re-peaked once c0 is known

    # One operator for the whole run: the binned frame is both the input and
    # the output grid, as it was when this used affine_transform in place.
    from holotomocupy.shift import Shift
    nyb, nxb = _bin2(flat[0], b).shape
    cl = Shift(nxb, nxb, nyb, nyb)

    for ji, j in enumerate(jj):
        stack = np.empty([ndist, nyb, nxb], dtype='float32')
        for k in range(ndist):
            raw = np.asarray(lay.read_proj(k, int(j)), dtype='float32')
            img = (raw - dark[k]) / np.maximum(flat[k] - dark[k], 1e-6)
            stack[k] = _bin2(img, b)
        # Onto the common object grid, and undo each plane's own piezo move so
        # what is left is only the inter-plane offset.
        res = _resample(cl, stack, rand[j] / 2**b, nmag)
        frames = [_band(res[k], args.highpass, args.smooth) for k in range(ndist)]

        est[ji, ref_dist] = 0.0
        hgt[ji, ref_dist] = 1.0
        cc = {k: _corr(frames[ref_dist], frames[k])
              for k in range(ndist) if k != ref_dist}
        if ji < ncoarse:
            # Stage 1: no window.  The offset is hundreds of px, so a narrow
            # search centred on zero would lock onto whatever is nearby.
            held.append((ji, cc))
            for k, c in cc.items():
                est[ji, k] = _peak(c, frames[0].shape[0] // 2 - 2)[:2]
            print(f'  angle {j:5d} (coarse)  ' + '  '.join(
                f'p{k+1} ({est[ji,k,0]:+8.2f},{est[ji,k,1]:+8.2f})'
                for k in range(ndist)), flush=True)
            if ji == ncoarse - 1:
                # Stage 2 centre: the median coarse lag, one per plane.  A
                # median so a single mis-locked angle cannot move it.
                for k in range(ndist):
                    if k != ref_dist:
                        c0[k] = (float(np.median(est[:ncoarse, k, 0])),
                                 float(np.median(est[:ncoarse, k, 1])))
                print(f'  stage-1 centres, binned px: ' + '  '.join(
                    f'p{k+1} ({c0[k][0]:+8.1f},{c0[k][1]:+8.1f})'
                    for k in range(ndist)), flush=True)
                for ji2, cc2 in held:       # re-peak the coarse angles in-window
                    for k, c in cc2.items():
                        dy, dx, h = _peak(c, args.search, c0[k])
                        hgt[ji2, k] = h
                        est[ji2, k] = (dy, dx) if h >= args.min_peak else np.nan
                held = []
            continue

        for k, c in cc.items():
            dy, dx, h = _peak(c, args.search, c0[k])
            hgt[ji, k] = h
            est[ji, k] = (dy, dx) if h >= args.min_peak else np.nan
        print(f'  angle {j:5d}  ' + '  '.join(
            f'p{k+1} ({est[ji,k,0]:+8.2f},{est[ji,k,1]:+8.2f}) h={hgt[ji,k]:.3f}'
            for k in range(ndist)), flush=True)

    # Degree-`deg` fit in angle, evaluated at every projection.
    x = jj / max(ntheta - 1, 1)
    xa = np.arange(ntheta) / max(ntheta - 1, 1)
    rhapp = np.zeros([ntheta, ndist, 2], dtype='float64')
    resid = np.zeros([ndist, 2])
    nused = np.zeros(ndist, dtype=int)
    for k in range(ndist):
        ok = np.isfinite(est[:, k, 0])
        nused[k] = ok.sum()
        if k == ref_dist:
            continue
        # Hard fail, not zeros.  This term is 270-390 object px; a plane left
        # at zero because its peaks were rejected would misalign that plane
        # completely and nothing downstream would say so.
        if nused[k] < args.deg + 2:
            raise SystemExit(
                f'estimate_rhapp: plane {k + 1} kept only {nused[k]} of '
                f'{len(jj)} angles (need {args.deg + 2} for a degree-{args.deg} '
                f'fit).  Peak heights {np.round(hgt[:, k], 3)} against '
                f'--min-peak {args.min_peak}')
        for c in range(2):
            p = np.polyfit(x[ok], est[ok, k, c], args.deg)
            rhapp[:, k, c] = np.polyval(p, xa)
            resid[k, c] = float(np.std(est[ok, k, c] - np.polyval(p, x[ok])))

    # Sign: _peak reports where plane k sits relative to the reference, and
    # step 3 wants the shift that puts it back.  Verified against rhapp.mat on
    # HT RD300 -- the raw lags match ESRF's numbers with the sign flipped.
    rhapp *= -(2**b)                                # binned lag -> object px
    rhapp -= rhapp[:, ref_dist:ref_dist + 1]        # reference plane is exactly 0
    # Mirror ESRF: plane 0's time-mean is removed from every plane.  It is one
    # constant per component, i.e. a global object translation, and dropping it
    # keeps our numbers directly comparable with rhapp.mat.
    rhapp -= rhapp[:, 0].mean(axis=0)[None, None, :]

    return dict(rhapp=rhapp, angles=jj, est=est * -(2**b), heights=hgt,
                resid=resid * 2**b, nused=nused, ref_dist=ref_dist,
                norm_magnifications=nmag)


def write_measured(lay, ref_dist, out_path, motion=None, **kw):
    """Measure rhapp and write out_path as .npy.  The one call steps15 makes.

    out_path is under path_out, never in the raw scan directory -- a
    measurement is ours, ESRF's rhapp.mat is theirs.
    """
    args = options(**kw)
    res = measure_rhapp(lay, ref_dist, args, motion=motion)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, res['rhapp'].astype('float32'))
    m = res['rhapp'].mean(axis=0)
    print(f'wrote {out_path}  {res["rhapp"].shape}')
    print(f'  per-plane mean, object px:  y={np.round(m[:, 0], 2)}  '
          f'x={np.round(m[:, 1], 2)}')
    print(f'  angles used per plane: {res["nused"]}  '
          f'fit residual rms: {np.round(res["resid"], 3).tolist()}')
    return res


# --- validation against ESRF's rhapp.mat -----------------------------------

def validate(lay, res, rhapp_bin=2):
    """Compare the fit with rhapp.mat, when the scan has one.

    rhapp.mat is on Peter's 2x2-binned grid, differenced against HIS reference
    plane; step 3 re-differences it against ref_dist and multiplies by
    rhapp_bin, and the sign is flipped.  This reproduces exactly that chain so
    the two arrays are in the same units before they are compared.
    """
    from holotomocupy.reader import load_octave_text_mat
    path = f'{lay.path}/{lay.pfile}_/rhapp.mat'
    if not os.path.exists(path):
        print(f'no {path}: nothing to validate against')
        return None
    ref_dist = res['ref_dist']
    ntheta = lay.ntheta
    raw = load_octave_text_mat(path, 'rhapp').swapaxes(0, 2)[:ntheta]
    raw = raw - raw[:, ref_dist:ref_dist + 1]
    raw = raw - raw[:, 0].mean(axis=0)[None, None, :]
    theirs = -raw * rhapp_bin
    ours = res['rhapp']
    print(f'\nvalidation against {os.path.basename(path)}  (object px)')
    print(f'{"plane":>6} {"comp":>5} {"ours ptp":>9} {"theirs ptp":>11} '
          f'{"max|diff|":>10} {"corr":>7}')
    for k in range(lay.ndist):
        for c, lbl in ((0, 'y'), (1, 'x')):
            a, t = ours[:, k, c], theirs[:, k, c]
            r = (np.corrcoef(a, t)[0, 1]
                 if np.std(a) > 1e-9 and np.std(t) > 1e-9 else np.nan)
            print(f'{k+1:>6} {lbl:>5} {np.ptp(a):>9.2f} {np.ptp(t):>11.2f} '
                  f'{np.max(np.abs(a - t)):>10.2f} {r:>7.3f}')
    return theirs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', nargs='?', help='config_steps15.conf')
    p.add_argument('--path', help='override the config path')
    p.add_argument('--pfile', help='override the config pfile')
    p.add_argument('--ref-dist', type=int, dest='ref_dist',
                   help="reference plane, 0-based; default is the config's")
    p.add_argument('--nangles', type=int, default=DEFAULTS['nangles'])
    p.add_argument('--bin', type=int, default=DEFAULTS['bin'],
                   help='correlate on the 4096/2**bin grid')
    p.add_argument('--ncoarse', type=int, default=DEFAULTS['ncoarse'],
                   help='angles given a full-frame search to locate the offset')
    p.add_argument('--search', type=int, default=DEFAULTS['search'],
                   help='+- stage-2 radius about the stage-1 centre, binned px')
    p.add_argument('--deg', type=int, default=DEFAULTS['deg'])
    p.add_argument('--highpass', type=float, default=DEFAULTS['highpass'],
                   help='trend removed, binned px; no setting of it helps')
    p.add_argument('--smooth', type=float, default=DEFAULTS['smooth'],
                   help='pre-smoothing, binned px; flat optimum over 8-12')
    p.add_argument('--nflat', type=int, default=DEFAULTS['nflat'])
    p.add_argument('--min-peak', type=float, dest='min_peak',
                   default=DEFAULTS['min_peak'])
    p.add_argument('--out', help='default <path_out>/measured/rhapp_measured.npy')
    p.add_argument('--validate', action='store_true',
                   help='compare with ESRF rhapp.mat when the scan has one')
    args = p.parse_args()

    path, pfile, ref_dist, out = args.path, args.pfile, args.ref_dist, args.out
    rhapp_bin = 2
    if args.config:
        from holotomocupy.config import parse_args_steps15
        cfg = parse_args_steps15(args.config)
        path = path or cfg.path
        pfile = pfile or cfg.pfile
        ref_dist = cfg.ref_dist if ref_dist is None else ref_dist
        out = out or f'{cfg.path_out}/measured/rhapp_measured.npy'
        rhapp_bin = getattr(cfg, 'rhapp_bin', 0) or rhapp_bin
    if not (path and pfile):
        p.error('give a config, or both --path and --pfile')
    ref_dist = 0 if ref_dist is None else ref_dist

    lay = Layout(path, pfile)
    print(lay)
    res = write_measured(lay, ref_dist, out or 'rhapp_measured.npy',
                         nangles=args.nangles, bin=args.bin, search=args.search,
                         ncoarse=args.ncoarse, deg=args.deg, nflat=args.nflat,
                         highpass=args.highpass, smooth=args.smooth,
                         min_peak=args.min_peak)
    if args.validate:
        validate(lay, res, rhapp_bin)
    return 0


if __name__ == '__main__':
    sys.exit(main())
