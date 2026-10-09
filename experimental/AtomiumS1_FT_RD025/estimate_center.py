#!/usr/bin/env python
"""NOT CALLED BY THE PIPELINE ANY MORE.  Step 4b measures the axis on bin-0
Paganin projections instead -- see estimate_axis_paganin.py, which imports
phase_corr/peak from here.  What follows is the raw-frame estimator, kept
as a standalone CLI and as the record of how the axis used to be measured;
its answer is the one the Fresnel fringes contaminate.


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

REJECTING BAD ESTIMATES.  An estimate whose fitted dy is wildly far from zero
has locked onto something that is not the sample, and is dropped (--max-dy).
What survives that is then clipped at 3 MAD about the median.  Both are
reported per estimate, so a run that keeps very few is visible rather than
silent.

--max-dy IS A SANITY FILTER, NOT A REQUIREMENT, and that is why its default is
200 px rather than the 15 it started at.  Opposed frames sit at opposite ends
of the scan, so the whole slow sample drift separates them -- ~114 px on
AtomiumS1 -- and nothing here undoes it.  It does not have to: peak() returns
the row AND the column of the 2-D correlation peak, so the horizontal lag the
axis is read from is measured AT the fitted vertical lag.  A vertical offset
costs overlap between the bands, not accuracy.  At crop=2048 and bands=3 the
bands are 683 rows, so 114 px still leaves ~83 % of each band in common.

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
2026).  That is the only thing undone, and the only thing that has to be: a
constant added to the horizontal shift column is exactly degenerate with the
axis position -- the mirror flips B, so a common offset ADDS instead of
cancelling and moves the answer by -(d[0] + d[ntheta]).  The commanded
displacement is not constant, so leaving it in would be fatal; the slow drift
is handled by max_dy above.

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


def load_shift_table(lay, dist, shifts_arg='auto'):
    """Displacement table for plane `dist`, in that plane's detector pixels.

    `shifts_arg` is `auto` (the layout's own random-displacement file), a bare
    filename looked up inside the distance directory, or a path used as given.
    A message describing what was loaded is returned alongside the table.
    """
    if shifts_arg in (None, 'auto'):
        sp = lay.shift_source(dist)
    elif os.sep in shifts_arg:
        sp = shifts_arg
    else:
        sp = f'{lay.dname(dist)}/{shifts_arg}'
    return np.loadtxt(sp, dtype='float32'), sp


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

def measure_center(lay, dist=0, nflat=8, pairs=3, bands=3, crop=2048,
                   template=32, max_dy=200.0, shift_table=None,
                   exact_pairs=True, log=print):
    """Axis offset from opposed pairs.  Returns a dict; writes nothing.

    `shift_table` is the (>= ntheta+1, 2) displacement to undo, col 0
    horizontal and col 1 vertical, in THIS plane's detector px -- i.e. what
    load_shift_table returns.  None lets Scan read the layout's own file.

    The return value under 'center' is in the units `rotation_center_shift`
    has always been in, so it can be used either way round: typed into a
    config, or added straight to cshifts_final[..., 1].
    """
    sc = Scan(lay, nflat, dist=dist, shifts=shift_table)
    n, ntheta = sc.n, lay.ntheta

    # --- static illumination template ------------------------------------
    tj = np.linspace(0, ntheta - 1, template, dtype=int)
    log(f'template : mean of {len(tj)} frames spread over the scan')
    static = np.zeros((n, n), dtype='float32')
    for j in tj:
        static += sc.frame(int(j))
    static /= len(tj)

    # --- opposed pairs ----------------------------------------------------
    lo, hi = (n - crop) // 2, (n + crop) // 2
    edges = np.linspace(0, crop, bands + 1, dtype=int)
    rows, curves, panels = [], [], []

    for j in range(pairs):
        kk = (ntheta - j) if exact_pairs else (ntheta - 1 - j)
        dtheta = 180.0 * (kk - j) / ntheta
        a = sc.unshift(sc.frame(j) - static, j)[lo:hi, lo:hi]
        b = sc.unshift(sc.frame(kk) - static, kk)[lo:hi, lo:hi][:, ::-1]
        for ib in range(bands):
            r0, r1 = edges[ib], edges[ib + 1]
            cc = phase_corr(a[r0:r1], b[r0:r1])
            dr, t, pk = peak(cc)
            # the crop cancels: lo + hi == n, so t is already a full-grid shift
            shift = (t - 1) / 2
            rows.append(dict(j=j, k=kk, band=ib, dtheta=dtheta, dy=dr,
                             t=t, peak=pk, shift=shift))
            log(f'  pair {j:4d}/{kk:4d} ({dtheta:7.3f} deg) band {ib}: '
                f'dy={dr:+7.2f}  t={t:+8.2f}  peak={pk:.4f}  '
                f'shift={shift:+8.2f} px')
            lag = np.arange(cc.shape[1])
            lag[lag > cc.shape[1] // 2] -= cc.shape[1]
            order = np.argsort(lag)
            curves.append((lag[order],
                           cc[int(round(dr)) % cc.shape[0]][order], shift))
        if j == 0:
            panels = [a, b, t]

    sh = np.array([r['shift'] for r in rows])
    dy = np.array([r['dy'] for r in rows])
    # Two rejections, in order: a vertical lag too large to be the sample
    # drift, then anything the surviving cluster disowns.
    #
    # Always logged, even when nothing is rejected: dy carries the slow drift
    # (nothing undoes it, see the module docstring), so its spread is the one
    # number that says whether max_dy is sitting close to the data or miles
    # away from it.
    log(f'dy over all {len(dy)} estimates: {dy.min():+.1f}..{dy.max():+.1f} px '
        f'(median {np.median(dy):+.1f}), max_dy={max_dy}')
    keep = np.abs(dy) <= max_dy
    if keep.sum() < 3:
        raise SystemExit(
            f'estimate_center: only {keep.sum()} of {len(sh)} estimates have '
            f'|dy| <= {max_dy}.  dy carries the slow sample drift, which is '
            f'NOT undone here -- opposed frames sit at opposite ends of the '
            f'scan, so they are separated by all of it.  If the spread above '
            f'is merely larger than max_dy, raise --max-dy; a vertical lag '
            f'much beyond the drift you expect means the pairs are not '
            f'correlating at all, and no threshold fixes that.')
    med = np.median(sh[keep])
    mad = np.median(np.abs(sh[keep] - med))
    keep &= np.abs(sh - med) <= max(3 * 1.4826 * mad, 2.0)
    n_dy = int((np.abs(dy) > max_dy).sum())
    log(f'rejected {n_dy} on |dy| > {max_dy} px, '
        f'{len(sh) - keep.sum() - n_dy} more at 3 MAD; {keep.sum()}/{len(sh)} kept')
    center = float(sh[keep].mean())
    log(f'rotation_center_shift = {center:+.2f} +- {sh[keep].std():.2f} px '
        f'(median {np.median(sh[keep]):+.2f})')

    return dict(center=center, std=float(sh[keep].std()), sc=sc, rows=rows,
                curves=curves, panels=panels, sh=sh, keep=keep, lo=lo, hi=hi,
                ntheta=ntheta, nkept=int(keep.sum()), ntotal=len(sh))


def write_measured(lay, ref_dist, out_path, dist=0, log=print, **kw):
    """Measure the axis offset and write it as a 1-element .npy.

    The call steps15 makes.  Only the commanded displacement is undone; the
    slow drift is left in and absorbed by max_dy (module docstring).
    """
    tab, msg = load_shift_table(lay, dist, 'auto')
    log(f'estimate_center: shifts {msg}')

    res = measure_center(lay, dist=dist, shift_table=tab, log=log, **kw)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.array([res['center']], dtype='float32'))
    log(f'estimate_center: wrote {out_path}  center {res["center"]:+.4f} px '
        f'({res["nkept"]}/{res["ntotal"]} estimates kept)')
    return res['center']


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
    ap.add_argument('--max-dy', type=float, default=200.0,
                    help='reject an estimate whose fitted vertical offset '
                         'exceeds this.  A sanity filter, not a requirement: '
                         'opposed frames are separated by the whole slow '
                         'sample drift (~114 px here) and the horizontal lag '
                         'is read at the fitted vertical one, so the drift '
                         'costs band overlap rather than accuracy.')
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
                         'against correct.txt.  Use this flag to reproduce '
                         'runs made before 2026-08-26, when the offset pairing '
                         'was the default.')
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

    table, smsg = load_shift_table(lay, k, args.shifts)

    print(f'scan     : {dname}   ({lay.flavour})')
    print(f'ntheta   : {ntheta}')
    print(f'shifts   : {smsg}')
    res = measure_center(lay, dist=k, nflat=args.nflat, pairs=args.pairs,
                         bands=args.bands, crop=args.crop,
                         template=args.template, max_dy=args.max_dy,
                         shift_table=table, exact_pairs=args.exact_pairs,
                         log=lambda s: print(s, flush=True))

    make_figure(args, res['sc'], res['panels'], res['rows'], res['curves'],
                res['sh'], res['keep'], dname, ntheta, res['lo'], res['hi'])
    return res['center']


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
