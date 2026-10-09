#!/usr/bin/env python
"""
Rotation-axis estimate from opposed (theta, theta+180) projection pairs.

    python estimate_center.py config_steps15.conf [options]

Two projections 180 deg apart are mirror images of each other about the
rotation axis: the ray through detector column u at theta is the same physical
ray as the one through column 2c-u at theta+180, so the two line integrals are
equal and

    A(u) = B(2c - u)                                                   (*)

with c the axis column.  Propagation to the detector does not break this --
Fresnel propagation commutes with a mirror -- so (*) holds for the raw holograms
and no reconstruction is needed.  Cross-correlating A against the horizontally
flipped B therefore measures the axis directly: if the alignment shift is t,

    c = (t + n - 1) / 2        and       rotation_center_shift = c - n/2
                                                               = (t - 1) / 2

which is the quantity `rotation_center_shift=` wants in config_steps15.conf and
config_step6_bin*.conf, in unbinned (bin-0) detector pixels.

THE SCAN COVERS 180 DEG, NOT 360.  With ScanRange=-180 and TOMO_N=ntheta the
angles are theta_j = -180*j/ntheta, so the opposed partner of frame j is frame
ntheta-1-j.  (Pairing j with j+ntheta/2, the natural guess, gives 90 deg and
produces pure noise -- that is a 4000-frame mistake worth not repeating.)  Pair
j is 180 - (0.045 + 0.09*j) deg apart, and that residual rotation smears a
feature at radius r by r*dtheta: 1.6 px at j=0, 4.8 px at j=1, 8 px at j=2 for
r = 2048.  How much that costs depends on how many pixels the sample structure
spans, i.e. on the voxel size -- on the 20 nm scan pairs 0..3 all work, on this
6 nm one only pairs 0 and 1 do.  Hence --pairs 3 by default, and the two
rejection rules below rather than blind averaging.

REJECTING BAD ESTIMATES.  The axis is vertical, so opposed views must line up in
the vertical direction once the encoder displacement is undone: an estimate
whose fitted dy is far from zero has locked onto something that is not the
sample, and is dropped (--max-dy).  What survives that is then clipped at 3 MAD
about the median.  Both are reported per estimate, so a run that keeps very few
is visible rather than silent.

TWO THINGS HAVE TO BE REMOVED BEFORE THE CORRELATION MEANS ANYTHING:

 1. The encoder displacement.  This scan is an FT large-random-displacement
    acquisition: correct.txt moves the sample by up to +-300 px between frames.
    Column 0 of correct.txt is the horizontal (detector column) displacement and
    column 1 the vertical one -- the same order steps15.py assumes at step 3
    (random_shifts[...,0] = shifts[...,1]) -- and the sample sits at
    +(correct[j,1], correct[j,0]) in (row, col).  Each frame is Fourier-shifted
    back by that before it is used.

 2. The detector-fixed illumination.  The residual speckle left after dividing
    by the flat is much stronger than the sample signal in these holograms, and
    it does not move when the sample does: correlating two raw frames returns a
    peak at exactly (0, 0) every time, whatever the sample did.  Subtracting the
    mean over frames spread across the scan removes it -- the sample smears over
    +-300 px of random displacement in that mean, the illumination does not.

WHICH SHIFT TABLE.  --shifts defaults to `auto`, i.e. whatever
esrf_layout.Layout.shift_source() says holds the commanded random displacement
for this scan flavour (correct.txt in 2025, projections/<pfile>_000k.txt in
2026).  --motion adds the slow drift on top, taken as
(correct_motion.txt - random displacement at the reference plane), rescaled to
this plane by norm_mag[k]/norm_mag[ref].  Use it when the reconstruction will
use correct_motion.txt, because a constant added to the horizontal shift column
is exactly degenerate with the axis position -- the mirror flips B, so a common
offset ADDS instead of cancelling and moves the answer by -(d[0] + d[ntheta]).

Deliberately standalone: numpy / fabio / matplotlib, no cupy and no MPI, so it
runs on a login node.  It reads the raw EDF tree, not the steps15 HDF5, so it
can be run before steps15 has ever been started.

Output: the number on stdout, and --fig (default center_estimate.png) showing
one pair overlaid at the fitted axis, the residual against the naive centre, and
the correlation curve behind every individual estimate.
"""

import argparse
import configparser
import os

import fabio
import numpy as np

from esrf_layout import Layout


# ---------------------------------------------------------------------------
# raw frames
# ---------------------------------------------------------------------------

def read_edf(fname):
    return fabio.open(fname).data.astype('float32')


class Scan:
    """Flat-fielded -log frames from one raw ESRF EDF distance directory.

    `lay` is an esrf_layout.Layout and `dist` a 0-based distance index, so the
    2025 and 2026 ref/dark/projection naming conventions are both handled
    without this class knowing which is in play.  `shifts` is the (nrows, 2)
    displacement table in THIS plane's detector pixels, column 0 horizontal and
    column 1 vertical -- the same order correct.txt uses and the same order
    step 3 of steps15.py assumes; pass None to read the layout's own file.
    """

    def __init__(self, lay, nflat, dist=0, shifts=None):
        self.lay, self.dist = lay, dist
        self.dname, self.pfile, self.ntheta = lay.dname(dist), lay.pfile, lay.ntheta
        ntheta = self.ntheta

        dark_f = lay.darks(dist, nflat)
        ref0_f = lay.refs(dist, 0, nflat)
        ref1_f = lay.refs(dist, ntheta, nflat)
        if not (dark_f and ref0_f):
            raise SystemExit(f'no dark*/ref (angle 0) EDFs in {self.dname}')

        self.dark = np.mean([read_edf(f) for f in dark_f], axis=0)
        self.ref0 = np.mean([read_edf(f) for f in ref0_f], axis=0) - self.dark
        # Step 2 of steps15.py normalises with the start-of-scan flats only.
        # Here the two ends of the scan are compared against each other, so the
        # flat is interpolated instead: it leaves less detector-fixed residual
        # at theta=180, which is what the correlation has to see through.
        self.ref1 = (np.mean([read_edf(f) for f in ref1_f], axis=0) - self.dark
                     if ref1_f else self.ref0)
        self.n = self.dark.shape[0]

        # NOT truncated to ntheta: frame ntheta exists (it is the last scan
        # frame, at exactly 180 deg) and the shift table has at least ntheta+1
        # rows, so index ntheta has to stay reachable for the exact pairing.
        self.shifts = (np.loadtxt(lay.shift_source(dist), dtype='float32')
                       if shifts is None else np.asarray(shifts, dtype='float32'))
        if len(self.shifts) < ntheta + 1:
            raise SystemExit(f'shift table has {len(self.shifts)} rows, '
                             f'need at least {ntheta + 1}')
        fy = np.fft.fftfreq(self.n)[:, None]
        fx = np.fft.fftfreq(self.n)[None, :]
        self._fy, self._fx = fy, fx

    def frame(self, j):
        w = j / self.ntheta
        img = read_edf(self.lay.proj(self.dist, j)) - self.dark
        img = img / ((1 - w) * self.ref0 + w * self.ref1 + 1e-3)
        return -np.log(np.clip(img, 1e-3, None))

    def unshift(self, img, j):
        """Move frame j back so the sample sits where it does at j = 0."""
        dy, dx = float(self.shifts[j, 1]), float(self.shifts[j, 0])
        ph = np.exp(2j * np.pi * (self._fy * dy + self._fx * dx))
        return np.real(np.fft.ifft2(np.fft.fft2(img) * ph))


def load_shift_table(lay, dist, shifts_arg='auto', motion=False):
    """Displacement table for plane `dist`, in that plane's detector pixels.

    `shifts_arg` is `auto` (the layout's own random-displacement file), a bare
    filename looked up inside the distance directory, or a path used as given.
    With `motion` the slow drift from correct_motion.txt is added: step 3 reads
    that file at the REFERENCE plane and applies
        (correct_motion - random[ref]) / norm_mag[ref]
    to every plane in object pixels, so at plane k the detector-pixel
    equivalent is that difference times norm_mag[k]/norm_mag[ref].  A message
    describing what was loaded is returned alongside the table.
    """
    if shifts_arg in (None, 'auto'):
        sp = lay.shift_source(dist)
    elif os.sep in shifts_arg:
        sp = shifts_arg
    else:
        sp = f'{lay.dname(dist)}/{shifts_arg}'
    tab = np.loadtxt(sp, dtype='float32')
    msg = sp

    if motion is not False:
        ref = 0 if motion is True else int(motion)
        mp = f'{lay.dname(ref)}/correct_motion.txt'
        if not os.path.exists(mp):
            raise SystemExit(f'--motion asked for, but {mp} does not exist')
        nm = lay.geometry()['norm_magnifications']
        raw = np.loadtxt(mp, dtype='float32')
        rnd = np.loadtxt(lay.shift_source(ref), dtype='float32')
        m = min(len(tab), len(raw), len(rnd))
        drift = (raw[:m] - rnd[:m]) * (nm[dist] / nm[ref])
        tab = tab[:m] + drift
        msg += (f'  +  drift from {mp} '
                f'(plane {ref + 1}, x {drift[:, 0].min():+.2f}..{drift[:, 0].max():+.2f}, '
                f'y {drift[:, 1].min():+.2f}..{drift[:, 1].max():+.2f} px)')
    return tab, msg


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

def phase_corr(a, b):
    """Cross-correlation surface of a against b, both windowed."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2(a * win), np.fft.fft2(b * win)
    x = A * np.conj(B)
    return np.real(np.fft.ifft2(x / (np.abs(x) + 1e-9)))


def peak(cc):
    """Sub-pixel peak of cc, as a signed (row, col) shift mapping b onto a."""
    p = np.unravel_index(np.argmax(cc), cc.shape)
    out = []
    for ax, i in enumerate(p):
        s = list(p)
        s[ax] = (i - 1) % cc.shape[ax]; lo = cc[tuple(s)]
        s[ax] = (i + 1) % cc.shape[ax]; hi = cc[tuple(s)]
        den = lo - 2 * cc[p] + hi
        v = i + (0.5 * (lo - hi) / den if den else 0.0)
        out.append(v - cc.shape[ax] if v > cc.shape[ax] / 2 else v)
    return out[0], out[1], float(cc[p])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--pairs', type=int, default=3,
                    help='opposed pairs (j, ntheta-1-j) for j = 0 .. pairs-1; '
                         'pair j is off 180 deg by 0.045 + 0.09*j deg, so the '
                         'later ones decorrelate -- raise it only if the fitted '
                         'values stay tight')
    ap.add_argument('--bands', type=int, default=3,
                    help='horizontal bands per pair, each giving its own estimate')
    ap.add_argument('--crop', type=int, default=2048,
                    help='central region used, in bin-0 px')
    ap.add_argument('--template', type=int, default=32,
                    help='frames spread over the scan that build the static '
                         'illumination template')
    ap.add_argument('--max-dy', type=float, default=15.0,
                    help='reject an estimate whose fitted vertical offset '
                         'exceeds this; opposed views cannot be shifted '
                         'vertically once the encoder motion is undone')
    ap.add_argument('--nflat', type=int, default=8, help='flats/darks averaged')
    ap.add_argument('--path', help="override the config's raw-data root, e.g. to "
                                   'run over an sshfs mount of eagle')
    ap.add_argument('--pfile', help="override the config's scan prefix")
    ap.add_argument('--dist', type=int, default=1,
                    help='1-based distance plane whose projections are used')
    ap.add_argument('--shifts', default='auto',
                    help='displacement table to undo before correlating.  '
                         '`auto` takes the layout\'s own random-displacement '
                         'file; a bare name is looked up in the distance '
                         'directory; a value containing a path separator is '
                         'used as given, so a candidate file can be tried '
                         'without writing it into the raw scan directory.')
    ap.add_argument('--motion', nargs='?', type=int, const=-1, default=None,
                    metavar='REFDIST',
                    help='also undo the slow drift from correct_motion.txt at '
                         'the reference plane (1-based; default: ref_dist+1 '
                         'from the config).  The centre has to be estimated '
                         'against whichever shift set the reconstruction will '
                         'actually use -- a constant in the horizontal column '
                         'is exactly degenerate with the axis position.')
    ap.add_argument('--no-exact-pairs', dest='exact_pairs',
                    action='store_false',
                    help='pair frame j with ntheta-1-j instead of ntheta-j.  '
                         'The scan runs 0..180 deg in ntheta steps and writes '
                         'ntheta+1 frames, so frame ntheta sits at exactly '
                         '180 deg and (j, ntheta-j) is exactly opposed, which '
                         'is now the default; (j, ntheta-1-j) is off by '
                         '0.045 deg for pair 0 and worsens by 0.09 deg per '
                         'pair.  Exact pairing barely moves the centre (-39.81 '
                         '-> -39.90 on the 6 nm scan) but clearly improves pair '
                         'quality: |dy| rejections drop from 5 of 9 to 3 of 9 '
                         'against correct.txt and from 5 to 2 against '
                         'correct_motion.txt.  Use this flag to reproduce runs '
                         'made before 2026-08-26, when the offset pairing was '
                         'the default.')
    ap.add_argument('--fig', default='center_estimate.png')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(args.config, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']
    lay = Layout((args.path or cfg.get('path')).rstrip('/'),
                 args.pfile or cfg.get('pfile'))
    k = args.dist - 1
    dname = lay.dname(k)
    ntheta = lay.ntheta

    motion = False
    if args.motion is not None:
        motion = (cfg.getint('ref_dist', fallback=0) if args.motion < 0
                  else args.motion - 1)
    table, smsg = load_shift_table(lay, k, args.shifts, motion)

    print(f'scan     : {dname}   ({lay.flavour})')
    print(f'ntheta   : {ntheta}')
    print(f'shifts   : {smsg}')
    sc = Scan(lay, args.nflat, dist=k, shifts=table)
    n = sc.n
    print(f'frame    : {n} x {n}')

    # --- static illumination template ------------------------------------
    tj = np.linspace(0, ntheta - 1, args.template, dtype=int)
    print(f'template : mean of {len(tj)} frames spread over the scan')
    static = np.zeros((n, n), dtype='float32')
    for j in tj:
        static += sc.frame(int(j))
    static /= len(tj)

    # --- opposed pairs ----------------------------------------------------
    lo, hi = (n - args.crop) // 2, (n + args.crop) // 2
    edges = np.linspace(0, args.crop, args.bands + 1, dtype=int)
    rows, curves, panels = [], [], []

    for j in range(args.pairs):
        k = (ntheta - j) if args.exact_pairs else (ntheta - 1 - j)
        dtheta = 180.0 * (k - j) / ntheta
        a = sc.unshift(sc.frame(j) - static, j)[lo:hi, lo:hi]
        b = sc.unshift(sc.frame(k) - static, k)[lo:hi, lo:hi][:, ::-1]
        for ib in range(args.bands):
            r0, r1 = edges[ib], edges[ib + 1]
            cc = phase_corr(a[r0:r1], b[r0:r1])
            dr, t, pk = peak(cc)
            # the crop cancels: lo + hi == n, so t is already a full-grid shift
            shift = (t - 1) / 2
            rows.append(dict(j=j, k=k, band=ib, dtheta=dtheta, dy=dr,
                             t=t, peak=pk, shift=shift))
            print(f'  pair {j:4d}/{k:4d} ({dtheta:7.3f} deg) band {ib}: '
                  f'dy={dr:+7.2f}  t={t:+8.2f}  peak={pk:.4f}  '
                  f'shift={shift:+8.2f} px', flush=True)
            # 1-D slice through the peak row: the correlation as a function of
            # the candidate axis, which is what the figure plots
            lag = np.arange(cc.shape[1])
            lag[lag > cc.shape[1] // 2] -= cc.shape[1]
            order = np.argsort(lag)
            curves.append((lag[order],
                           cc[int(round(dr)) % cc.shape[0]][order], shift))
        if j == 0:
            panels = [a, b, t]

    sh = np.array([r['shift'] for r in rows])
    dy = np.array([r['dy'] for r in rows])
    # Two rejections, in order: a vertical offset the geometry forbids, then
    # anything the surviving cluster disowns.
    keep = np.abs(dy) <= args.max_dy
    if keep.sum() < 3:
        raise SystemExit(f'only {keep.sum()} of {len(sh)} estimates have '
                         f'|dy| <= {args.max_dy}: the pairs are not correlating. '
                         f'Try --pairs 2, a smaller --crop, or more --template.')
    med = np.median(sh[keep])
    mad = np.median(np.abs(sh[keep] - med))
    keep &= np.abs(sh - med) <= max(3 * 1.4826 * mad, 2.0)
    print(f'\nrejected {(np.abs(dy) > args.max_dy).sum()} on |dy| > {args.max_dy} px, '
          f'{len(sh) - keep.sum() - (np.abs(dy) > args.max_dy).sum()} more at 3 MAD; '
          f'{keep.sum()}/{len(sh)} kept')
    print(f'rotation_center_shift = {sh[keep].mean():+.2f} +- {sh[keep].std():.2f} px '
          f'(median {np.median(sh[keep]):+.2f})')

    make_figure(args, sc, panels, rows, curves, sh, keep, dname, ntheta, lo, hi)
    return sh[keep].mean()


def make_figure(args, sc, panels, rows, curves, sh, keep, dname, ntheta, lo, hi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    a, b, t = panels
    n = sc.n
    best = sh[keep].mean()

    def roll(img, s):
        return np.roll(img, int(round(s)), axis=1)

    # b is already flipped; aligning it onto a needs a shift of t = 2*best + 1
    b_al = roll(b, int(round(2 * best + 1)))
    b_na = b                                  # what centre = 0 would give (t = 1)

    def norm(x):
        v = x[x.shape[0] // 4:-x.shape[0] // 4, x.shape[1] // 4:-x.shape[1] // 4]
        m, s = np.mean(v), np.std(v)
        return np.clip((x - m) / (4 * s) + 0.5, 0, 1)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.28, wspace=0.18)

    for ax, img, ttl in [
            (fig.add_subplot(gs[0, 0]), a,
             f'A  theta $\\approx$ 0  (frame {rows[0]["j"]})'),
            (fig.add_subplot(gs[0, 1]), b_al,
             f'B  theta $\\approx$ 180  (frame {rows[0]["k"]}), flipped + aligned'),
    ]:
        ax.imshow(norm(img), cmap='gray', interpolation='nearest')
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    d_na, d_al = a - b_na, a - b_al
    sc_ = 3 * np.std(d_na)
    ax.imshow(np.clip(d_al / (2 * sc_) + 0.5, 0, 1), cmap='gray',
              interpolation='nearest')
    ax.set_title(f'A - B at the fitted axis\nRMS {np.std(d_al):.4f}  '
                 f'(vs {np.std(d_na):.4f} at shift 0)', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0:2])
    for (lag, cur, s), r in zip(curves, rows):
        keep_i = np.abs(lag) < 400
        ax.plot((lag[keep_i] - 1) / 2, cur[keep_i], lw=0.9,
                alpha=0.85 if abs(s - np.median(sh)) < 5 else 0.3,
                label=f'{r["j"]}/{r["k"]} b{r["band"]}')
    ax.axvline(best, color='crimson', lw=1.4,
               label=f'mean {best:+.2f} px')
    ax.axvline(0, color='0.6', lw=1.0, ls=':', label='grid centre')
    ax.set_xlabel('candidate rotation_center_shift  (bin-0 px)')
    ax.set_ylabel('normalised cross-correlation')
    ax.set_title('Correlation of A against the flipped B, per pair and band',
                 fontsize=10)
    ax.set_xlim(-200, 200)
    ax.legend(fontsize=6, ncol=3, loc='upper right')

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(sh))
    ax.scatter(x[keep], sh[keep], s=26, color='tab:blue', label='used')
    if (~keep).any():
        ax.scatter(x[~keep], sh[~keep], s=26, color='0.7', marker='x',
                   label='rejected')
    ax.axhline(best, color='crimson', lw=1.3)
    ax.axhspan(best - sh[keep].std(), best + sh[keep].std(),
               color='crimson', alpha=0.13)
    ax.set_xlabel('estimate  (pair x band)')
    ax.set_ylabel('rotation_center_shift  (bin-0 px)')
    ax.set_title(f'{sh[keep].mean():+.2f} $\\pm$ {sh[keep].std():.2f} px',
                 fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle(f'Rotation axis from opposed projections  --  '
                 f'{os.path.basename(dname.rstrip("/"))}   '
                 f'(n={n}, crop {args.crop}, {args.pairs} pairs x {args.bands} bands)',
                 fontsize=11)
    fig.savefig(args.fig, dpi=110, bbox_inches='tight')
    print(f'figure -> {args.fig}')


if __name__ == '__main__':
    main()
