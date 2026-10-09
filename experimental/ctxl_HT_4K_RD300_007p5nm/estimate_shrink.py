#!/usr/bin/env python
"""
Measure sample shrinkage from the post-scan retakes, without launching a job.

    python estimate_shrink.py config_steps15.conf [--path ...] [--grid 5]

THE IDEA.  estimate_motion.py registers each post-scan retake against the scan
frame it repeats and keeps the TRANSLATION, which is the stage drift.  What it
throws away is the rest of the transform.  A sample that shrank during the scan
gives a retake that is a slightly SCALED copy of the original, so the residual
displacement field is not constant -- it points inward (or outward) and grows
linearly with distance from the centre.  Measuring that slope is measuring the
shrinkage, using nothing but data the scan already contains.

    frame 4002 (omega=0)  vs scan frame 0     -> the whole scan   (dt = 1)
    frame 4001 (omega=90) vs scan frame 2000  -> the second half  (dt = 0.5)

That pairing is the built-in consistency check: the pipeline's shrink model is
linear in time, shrink(t) = A*t + B with t = theta_idx/(ntheta-1), so if what we
measure is real shrinkage rather than noise or a magnification wobble, the
omega=0 pair must show about TWICE the scale change of the omega=90 pair.  Two
independent pairs, one prediction, no free parameters.

METHOD.  Both frames are flat-fielded and moved back to a common piezo position
exactly as in estimate_motion.py (same template, same zero-lag repair -- see the
long note in its measure() for why the centre bin of the correlation has to be
interpolated over).  The overlap is then cut into a --grid x --grid array of
blocks, each block phase-correlated on its own, and the resulting displacement
field fitted with

    d(r) = t + M r,   r = block centre relative to the frame centre

The interesting part of M is its diagonal: M[0,0] is the fractional scale change
along y, M[1,1] along x.  The antisymmetric part is a rotation and is reported
as a diagnostic -- a real shrinkage has none, so a large rotation means the fit
is picking up something else.

Per-axis, not averaged: the rotation axis is vertical at BOTH omega=0 and
omega=90, so the y scale is measured twice on the same physical direction and
the two values have to agree.  The x scale is a different direction of the
sample in each pair, so those two may legitimately differ.  This is also how
the pipeline stores it -- shrink_nd's trailing axis is (y, x).

SIGN.  phase_corr(a, b) peaks at the shift that carries b onto a, so a positive
diagonal means the retake is LARGER than the original, i.e. expansion; shrinkage
is negative.  s is a fraction: multiply by n/2 for the edge displacement in px.

Deliberately standalone: numpy / fabio / matplotlib, no cupy and no MPI, so it
runs on a login node.  Shares Scan and phase_corr with estimate_center.py and
the retake bookkeeping with estimate_motion.py rather than copying them.
"""

import argparse
import configparser

import numpy as np

import estimate_center as ec
import estimate_motion as em
from esrf_layout import Layout


def zoom(img, k):
    """img resampled about its centre by factor k, i.e. out(r) = img(r/k).

    Separable Catmull-Rom, written out in numpy rather than called from
    scipy.ndimage: the Polaris htc env has no scipy, and --inject is the one
    thing in this script that needs an interpolator.  Separability is what
    makes it affordable at 4096^2 -- two passes of four weighted shifts each,
    instead of a 16-tap gather over a 2-D coordinate array.

    Catmull-Rom is interpolating (it reproduces the samples it is given) and
    has a flat passband over the low frequencies the affine fit actually uses,
    so it neither invents nor damps the scale term being injected; the
    round-trip is checked by the recovered-minus-injected number the caller
    prints.
    """
    n = img.shape[0]
    c = (n - 1) / 2.0
    g = (np.arange(n) - c) / k + c
    i0 = np.floor(g).astype(int)
    t = (g - i0)[:, None]                                   # (n, 1)
    w = np.hstack([-0.5 * t**3 + t**2 - 0.5 * t,            # tap i0-1
                   1.5 * t**3 - 2.5 * t**2 + 1.0,           # tap i0
                   -1.5 * t**3 + 2.0 * t**2 + 0.5 * t,      # tap i0+1
                   0.5 * t**3 - 0.5 * t**2])                # tap i0+2
    idx = np.clip(i0[:, None] + np.arange(-1, 3)[None, :], 0, n - 1)  # (n, 4)

    # float32 throughout: the (n, 4, n) gather below is 268 MB at n=4096 in
    # single and twice that in double, for an interpolation whose error is
    # already far under the 0.03 px correlation noise.
    out = np.ascontiguousarray(img, dtype='float32')
    w = w.astype('float32')
    for _ in range(2):                                      # rows, then cols
        out = np.einsum('nt,ntx->nx', w, out[idx])          # (n, 4, n) gather
        out = np.ascontiguousarray(out.T)
    return out


# --- what --inject actually measures -----------------------------------------
# The response of this estimator is not 1:1 at small slopes.  Driven with a
# synthetic band-limited frame and a known injection, the identical
# field()/affine() chain returns
#
#     injected    0 ppm  ->    0 +- 100 ppm
#     injected  500 ppm  ->  630 ..  970 ppm   (median ~750)
#     injected 1000 ppm  ->  890 .. 1310 ppm   (median ~930)
#
# so the transfer function is roughly 1.5x at 500 ppm and 1.0x at 1000 ppm.
# The cause is the parabolic sub-pixel peak fit in em.local_peak: a
# phase-correlation peak is a Dirichlet kernel, not a parabola, and fitting a
# parabola to it inflates sub-pixel displacements.  500 ppm at a block centre
# 1200 px out is 0.6 px, squarely in the inflated regime; 1000 ppm is 1.2 px
# and mostly out of it.
#
# This does not need fixing for the question being asked, and it is important
# to see why.  The bias is an OVER-response, so a measured slope is an upper
# bound on the true one: reading +107 +- 210 ppm means the truth is at most
# that and probably smaller.  A null result from a hot estimator is a stronger
# null, not a weaker one.  It would matter if this script were ever used to
# MEASURE a shrinkage rather than to bound one -- at that point calibrate
# against --inject at the amplitude in question, or replace local_peak's
# parabola with an upsampled-DFT peak.

def peak_near(cc, ctr, r):
    """Sub-pixel peak of cc within r px of ctr=(dy,dx), as (dy, dx, height)."""
    iy, ix = int(round(ctr[0])), int(round(ctr[1]))
    rolled = np.roll(np.roll(cc, -iy, axis=0), -ix, axis=1)
    dy, dx, pk = em.local_peak(rolled, r)
    return dy + iy, dx + ix, pk


def corr(a, b, repair=True):
    """phase_corr with the unusable zero-lag bin interpolated over."""
    cc = ec.phase_corr(a, b)
    if repair:
        cc[0, 0] = 0.25 * (cc[0, 1] + cc[0, -1] + cc[1, 0] + cc[-1, 0])
    return cc


def field(A, B, crop, grid, search, n):
    """Block displacement field between two prepared frames.

    Returns (r, d, w): block centres relative to the frame centre, the measured
    (dy, dx) at each, and the correlation peak height as a weight.  The global
    translation is measured first on the whole crop and each block then searched
    around it -- the block offsets differ from it by only the shrinkage term,
    which is well under a pixel, so a search centred on zero would be a search
    centred on the wrong place once the drift is several px.
    """
    lo, hi = (n - crop) // 2, (n + crop) // 2
    a, b = A[lo:hi, lo:hi], B[lo:hi, lo:hi]
    gy, gx, _ = em.local_peak(corr(a, b), search)

    h = crop // grid
    r, d, w = [], [], []
    for iy in range(grid):
        for ix in range(grid):
            sy, sx = slice(iy * h, (iy + 1) * h), slice(ix * h, (ix + 1) * h)
            cc = corr(a[sy, sx], b[sy, sx])
            dy, dx, pk = peak_near(cc, (gy, gx), search)
            r.append([lo + (iy + 0.5) * h - n / 2, lo + (ix + 0.5) * h - n / 2])
            d.append([dy, dx])
            w.append(pk)
    return np.array(r), np.array(d), np.array(w), (gy, gx)


def affine(r, d, keep):
    """Least-squares [dy;dx] = t + M r over the kept blocks.

    Returns M, t and the per-element standard error of M, taken from the fit
    residual (nothing else here knows the true measurement noise, and the
    residual scatter of the blocks is the honest estimate of it).
    """
    X = np.column_stack([np.ones(keep.sum()), r[keep, 0], r[keep, 1]])
    M, t, se = np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2))
    dof = max(keep.sum() - 3, 1)
    cov = np.linalg.pinv(X.T @ X)
    for k in range(2):
        c, *_ = np.linalg.lstsq(X, d[keep, k], rcond=None)
        t[k], M[k] = c[0], c[1:]
        res = d[keep, k] - X @ c
        s2 = float(res @ res) / dof
        se[k] = np.sqrt(s2 * np.diag(cov)[1:])
    return M, t, se


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--path', help="override the config's raw-data root")
    ap.add_argument('--pfile', help="override the config's scan prefix")
    ap.add_argument('--dist', type=int, default=1,
                    help='1-based distance plane whose retakes are measured')
    ap.add_argument('--crop', default='2048,2560,3072',
                    type=lambda v: [int(x) for x in v.split(',')],
                    help='comma-separated crops to repeat the fit over; the '
                         'median is reported and the scatter is the error bar. '
                         'Larger crops than in estimate_motion.py are fine and '
                         'wanted here -- the shrinkage signal grows with the '
                         'lever arm, and a block near the edge that flat-fields '
                         'badly is dropped by --min-peak rather than dragging '
                         'a translation with it.')
    ap.add_argument('--grid', type=int, default=5,
                    help='blocks per side; --grid 5 on --crop 2048 gives 25 '
                         'blocks of 409 px.  Fewer, bigger blocks correlate '
                         'more reliably but shorten the lever arm.')
    ap.add_argument('--search', type=int, default=8,
                    help='block peak is searched this far from the global '
                         'translation; the shrinkage term is sub-pixel, so this '
                         'only needs to cover block-to-block noise')
    ap.add_argument('--min-peak', type=float, default=0.004,
                    help='blocks with a weaker correlation peak are dropped '
                         'from the fit')
    ap.add_argument('--template', type=int, default=32)
    ap.add_argument('--nflat', type=int, default=8)
    ap.add_argument('--inject', type=float, default=0.0,
                    help='SELF-TEST.  Rescale the retake by this many ppm '
                         'before measuring, and report what is recovered.  A '
                         'null result is only worth as much as the sensitivity '
                         'behind it, so this is how the upper limit below is '
                         'earned: inject 500 and the fit has to come back with '
                         '500 plus whatever it reads with --inject 0.')
    ap.add_argument('--fig', default='shrink_estimate.png')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(args.config, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']
    path = (args.path or cfg.get('path')).rstrip('/')
    pfile = args.pfile or cfg.get('pfile')
    lay = Layout(path, pfile)
    kdist = args.dist - 1
    dname = lay.dname(kdist)
    ntheta = lay.ntheta

    print(f'scan     : {dname}   ({lay.flavour})')
    print(f'ntheta   : {ntheta}')
    shifts = np.loadtxt(lay.shift_source(kdist))
    sc = ec.Scan(lay, args.nflat, dist=kdist, shifts=shifts)
    scales = em.px_per_um(lay, kdist, shifts)
    retakes = em.find_retakes(lay, kdist, ntheta)
    if not retakes:
        raise SystemExit(f'no post-scan retake frames after {ntheta} in {dname}')
    print(f'retakes  : ' + ', '.join(f'{j} (omega={w:.0f} -> frame {js})'
                                     for j, js, w in retakes))

    used = {js for _, js, _ in retakes} | {0, ntheta}
    tj = []
    for j in np.linspace(0, ntheta, args.template, dtype=int):
        j = int(j)
        while j in used or j in tj:
            j += 7 if j < ntheta - 7 else -7
        tj.append(j)
    print(f'template : {len(tj)} frames, excluding {sorted(used)}', flush=True)
    static = np.zeros((sc.n, sc.n), dtype='float32')
    for j in tj:
        static += sc.frame(j)
    static /= len(tj)

    fy, fx = sc._fy, sc._fx

    def prep(j, dy, dx):
        ph = np.exp(2j * np.pi * (fy * dy + fx * dx))
        return np.real(np.fft.ifft2(np.fft.fft2(sc.frame(j) - static) * ph))

    results = []
    for j, js, omega in retakes:
        disp = em.retake_disp(lay, kdist, shifts, scales, j)
        A = prep(j, disp[1], disp[0])
        if args.inject:
            A = zoom(A, 1.0 + args.inject * 1e-6)
        B = prep(js, shifts[js, 1], shifts[js, 0])
        dt = abs(ntheta - js) / ntheta      # fraction of the scan spanned

        print(f'\n=== retake {j} (omega={omega:.0f}) vs frame {js}   '
              f'dt = {dt:.2f} of the scan ===')
        print(f'{"crop":>6} {"blocks":>7} {"s_y":>12} {"s_x":>12} '
              f'{"rot (urad)":>11} {"resid px":>9}  block peaks')
        per = []
        for crop in args.crop:
            r, d, w, g = field(A, B, crop, args.grid, args.search, sc.n)
            keep = w >= args.min_peak
            if keep.sum() < 6:
                print(f'{crop:6d} {keep.sum():3d}/{len(w):<3d}   too few blocks '
                      f'above --min-peak {args.min_peak}')
                continue
            M, t, se = affine(r, d, keep)
            X = np.column_stack([np.ones(keep.sum()), r[keep, 0], r[keep, 1]])
            resid = np.hypot(*[d[keep, k] - X @ np.r_[t[k], M[k]] for k in (0, 1)])
            rot = 0.5 * (M[1, 0] - M[0, 1])
            print(f'{crop:6d} {keep.sum():3d}/{len(w):<3d} '
                  f'{M[0,0]*1e6:+7.1f}+-{se[0,0]*1e6:<4.1f} '
                  f'{M[1,1]*1e6:+7.1f}+-{se[1,1]*1e6:<4.1f} '
                  f'{rot*1e6:+11.1f} {resid.std():9.3f} '
                  f'  peak {np.min(w):.4f}-{np.max(w):.4f}')
            per.append((M, se, rot, keep.sum(), r, d, w, keep, crop))
        if not per:
            continue
        syv = np.array([p[0][0, 0] for p in per])
        sxv = np.array([p[0][1, 1] for p in per])
        usy = max(np.median(np.abs(syv - np.median(syv))) * 1.4826,
                  float(np.median([p[1][0, 0] for p in per])))
        usx = max(np.median(np.abs(sxv - np.median(sxv))) * 1.4826,
                  float(np.median([p[1][1, 1] for p in per])))
        sy_, sx_ = float(np.median(syv)), float(np.median(sxv))
        print(f'{"median":>6} {"":7} {sy_*1e6:+7.1f}+-{usy*1e6:<4.1f} '
              f'{sx_*1e6:+7.1f}+-{usx*1e6:<4.1f}   ppm of the frame')
        print(f'       edge displacement at r = n/2 = {sc.n//2} px: '
              f'y {sy_*sc.n/2:+.2f} +- {usy*sc.n/2:.2f} px, '
              f'x {sx_*sc.n/2:+.2f} +- {usx*sc.n/2:.2f} px')
        for nm, v, u in (('y', sy_, usy), ('x', sx_, usx)):
            sig = abs(v) / u if u else 0
            print(f'       s_{nm}: {sig:.1f} sigma -- '
                  + ('SIGNIFICANT' if sig >= 3 else
                     'marginal' if sig >= 2 else 'consistent with zero'))
        results.append(dict(j=j, js=js, omega=omega, dt=dt, sy=sy_, sx=sx_,
                            usy=usy, usx=usx, per=per))

    # --- the linearity cross-check ----------------------------------------
    print('\n=== is it shrinkage? ===')
    if len(results) < 2:
        print('only one usable retake -- no cross-check possible')
    else:
        print('A linear shrink model predicts s proportional to the fraction of '
              'the scan\nspanned, so s(dt=1) should be twice s(dt=0.5).  '
              'Implied A = s/dt:')
        print(f'{"pair":>16} {"dt":>5} {"A_y (ppm)":>16} {"A_x (ppm)":>16}')
        for R in results:
            print(f'{str(R["j"]) + "/" + str(R["js"]):>16} {R["dt"]:5.2f} '
                  f'{R["sy"]/R["dt"]*1e6:+9.1f}+-{R["usy"]/R["dt"]*1e6:<5.1f} '
                  f'{R["sx"]/R["dt"]*1e6:+9.1f}+-{R["usx"]/R["dt"]*1e6:<5.1f}')
        for k, nm in ((('sy', 'usy'), 'y'), (('sx', 'usx'), 'x')):
            a = [R[k[0]] / R['dt'] for R in results]
            u = [R[k[1]] / R['dt'] for R in results]
            diff = abs(a[0] - a[1]) / np.hypot(u[0], u[1])
            print(f'  A_{nm}: the two pairs differ by {diff:.1f} sigma -- '
                  + ('consistent, the linear model holds' if diff < 2 else
                     'INCONSISTENT, not a simple linear shrink'))
        if all(abs(R['sy']) / R['usy'] < 2 and abs(R['sx']) / R['usx'] < 2
               for R in results):
            print('\n  Every scale term is within 2 sigma of zero: there is no\n'
                  '  measurable shrinkage in this scan.  Leave the shrink model\n'
                  '  at A = B = 0 (rho[tp] = 0) -- fitting it would be fitting noise.')

    make_figure(args, results, sc.n, dname)


def make_figure(args, results, n, dname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not results:
        return
    fig, axes = plt.subplots(2, len(results), figsize=(6.5 * len(results), 11),
                             squeeze=False)
    for c, R in enumerate(results):
        M, se, rot, nk, r, d, w, keep, crop = R['per'][len(R['per']) // 2]
        X = np.column_stack([np.ones(len(r)), r[:, 0], r[:, 1]])
        pred = np.column_stack([X @ np.r_[0, M[k]] for k in (0, 1)])
        res = d - d[keep].mean(0)

        ax = axes[0][c]
        sc_ = 400.0
        ax.quiver(r[keep, 1], r[keep, 0], res[keep, 1] * sc_, res[keep, 0] * sc_,
                  angles='xy', scale_units='xy', scale=1, color='tab:blue',
                  width=0.004, label='measured')
        ax.quiver(r[keep, 1], r[keep, 0], pred[keep, 1] * sc_, pred[keep, 0] * sc_,
                  angles='xy', scale_units='xy', scale=1, color='tab:red',
                  width=0.002, label='affine fit')
        if (~keep).any():
            ax.plot(r[~keep, 1], r[~keep, 0], 'kx', ms=7, label='dropped')
        ax.set_title(f'retake {R["j"]} (omega={R["omega"]:.0f}) vs frame '
                     f'{R["js"]}\ncrop {crop}, arrows x{sc_:.0f}, '
                     f'translation removed')
        ax.set_xlabel('x - n/2 (px)'); ax.set_ylabel('y - n/2 (px)')
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1][c]
        for k, nm, col in ((0, 'y', 'tab:blue'), (1, 'x', 'tab:orange')):
            ax.plot(r[keep, k], res[keep, k], 'o', color=col, ms=5,
                    label=f'd{nm} vs {nm}')
            xs = np.linspace(r[:, k].min(), r[:, k].max(), 2)
            ax.plot(xs, M[k, k] * xs, '-', color=col,
                    label=f's_{nm} = {M[k,k]*1e6:+.1f} +- '
                          f'{R["usy" if k == 0 else "usx"]*1e6:.1f} ppm')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('distance from centre along that axis (px)')
        ax.set_ylabel('residual displacement (px)')
        ax.set_title('slope of this line IS the shrinkage')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'Shrinkage from post-scan retakes -- {dname.split("/")[-1]}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=110)
    print(f'\nwrote {args.fig}')


if __name__ == '__main__':
    main()
