#!/usr/bin/env python
"""
Estimate sample drift during the scan and write correct_motion.txt.

    python estimate_motion.py config_steps15.conf [--path ...] [--pfile ...]

WHAT THE FILE IS.  correct.txt holds the commanded random displacement of the
sample piezos, one row per projection.  It does not know that the sample also
drifts slowly over the ~70 min of the scan.  correct_motion.txt is the same
table with that drift folded in, and step 3 of steps15.py uses it INSTEAD of
correct.txt, not on top of it:

    raw_motion   = loadtxt(correct_motion.txt)[:ntheta, ::-1]
    motion_base  = raw_motion / norm_mag - random_shifts
    shifts_final = random_shifts + rhapp + motion + correct3d

which at ndist=1, with no rhapp and no correct3D, telescopes to
shifts_final = raw_motion.  So correct_motion.txt must be in exactly the units
and column order of correct.txt: object-plane pixels at the scan's own voxel
size, col 0 = horizontal (+spy), col 1 = vertical (-spz).

HOW THE DRIFT IS MEASURED.  The scan writes 4003 frames, not 4000.  Frames
ntheta+1 and ntheta+2 are retakes made right after the scan ends, at
omega = 90 and omega = 0, both at frame 0's piezo position -- and frame ntheta
itself is the last scan frame, at omega = 180.  Comparing each retake with the
scan frame at the same angle measures how far the sample moved between the two
exposures.  Frame ntheta is its own retake, so its drift is zero by definition
and everything is measured relative to the end of the scan.  Three points at
j = 0, ntheta/2, ntheta, one quadratic through them, mean removed.

That is Peter Cloetens' method, and it is what his quali.mat records
(corr_imagesafterscan at rot_positions 0/90/180).  It was reverse-engineered
from the 20 nm sibling scan, where both quali.mat and the real
correct_motion.txt exist -- run --validate there to see the comparison.

TWO THINGS THE VALIDATION SHOWS.  (1) quali.mat is in bin-2 pixels: the shift
this script measures on raw 4096 px frames is 2.02x Peter's value at omega=0.
He writes it into correct_motion.txt unscaled, next to a correct.txt that is in
unbinned pixels.  This script does NOT copy that -- it writes the drift in
unbinned pixels, which is what our step 3 needs, so its output is ~2x Peter's
in the vertical.  --peter-scale reproduces his convention instead.  (2) Peter's
horizontal column is a degree-5 polynomial, not a quadratic, so it comes from
some other measurement in his pipeline that has not been identified.  Its
amplitude is small (0.14 px rms against 0.61 px vertical), and three points
cannot produce a quintic, so the horizontal here is a quadratic like the
vertical and is the weaker of the two numbers.

WHY THE CORRELATION NEEDS CARE.  Residual detector-fixed illumination survives
flat-fielding and is stronger than the sample.  It is removed by subtracting a
mean over --template frames spread across the scan, in which the sample smears
over its own +-300 px of random displacement.  What survives still pins a peak
at the position the two frames' illumination lines up at, which for the
omega=90 pair is ~200 px away from the sample peak because the scan frame and
the retake sit at different piezo positions.  The peak is therefore searched
only within --search px of the expected answer; the illumination peak is
reported alongside so a run where it won is visible rather than silent.
"""

import argparse
import configparser
import os

import fabio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import estimate_center as ec
from esrf_layout import Layout


# ---------------------------------------------------------------------------
# raw-scan metadata
# ---------------------------------------------------------------------------

def motors(lay, k, j):
    """All motors of frame j at distance k, as a dict, from its EDF header."""
    return lay.motors(k, j)


def px_per_um(lay, k, shifts, nsample=7):
    """Scale and signs that map (spy, spz) onto the shift table, from the data.

    On the 2025 scans the table turns out to be exactly (spy - spy0)/voxel in
    column 0 and -(spz - spz0)/voxel in column 1, so the scale can be read off
    rather than taken from PixelSize in the .info -- one less thing to get
    wrong on a scan whose sidecar disagrees with its own pixels.

    The 2026 EDF headers carry only somega, so there is nothing to fit against.
    Returns None in that case; the caller then gets the retake displacement
    straight out of the shift table instead, which is what these scales were
    only ever used to reconstruct.
    """
    step = max(1, (len(shifts) - 1) // (nsample - 1))
    js = np.arange(0, len(shifts) - 1, step)[:nsample]
    m0 = motors(lay, k, int(js[0]))
    if 'spy' not in m0 or 'spz' not in m0:
        return None
    sp = np.array([[motors(lay, k, int(j))['spy'],
                    motors(lay, k, int(j))['spz']] for j in js])
    a0 = np.polyfit(sp[:, 0], shifts[js, 0], 1)
    a1 = np.polyfit(sp[:, 1], shifts[js, 1], 1)
    r0 = np.corrcoef(sp[:, 0], shifts[js, 0])[0, 1]
    r1 = np.corrcoef(sp[:, 1], shifts[js, 1])[0, 1]
    if min(abs(r0), abs(r1)) < 0.999:
        raise SystemExit(f'the shift table does not track spy/spz (r={r0:.4f}, '
                         f'{r1:.4f}); this scan does not fit the assumed layout')
    return a0[0], a1[0]


def find_retakes(lay, k, ntheta):
    """Locate the post-scan retake frames and the scan frame each one repeats.

    Returns [(retake_index, scan_index, omega)], plus the omega=180 anchor,
    which is frame ntheta itself and therefore has zero drift by construction.

    The angle comes from angles_file.txt when the scan directory has one (the
    2026 layout does, and reading a 4003-line text file beats opening eight
    32 MB EDFs) and from the frame's own EDF header otherwise.
    """
    afile = f'{lay.dname(k)}/angles_file.txt'
    angles = np.loadtxt(afile) if os.path.exists(afile) else None
    out = []
    for j in range(ntheta, ntheta + 8):
        if not os.path.exists(lay.proj(k, j)):
            break
        omega = float(angles[j]) if angles is not None and j < len(angles) \
            else lay.omega(k, j)
        js = int(round(abs(omega) / 180.0 * ntheta))
        if j == ntheta:
            continue          # the scan's own last frame = the drift reference
        out.append((j, js, omega))
    return out


def retake_disp(lay, k, shifts, scales, j):
    """Commanded sample displacement at frame j, relative to frame 0, in px.

    Preferred source is the shift table itself: the 2026 tables have a row for
    every written frame, retakes included (all zero, the piezo is parked), so
    the answer is just shifts[j] - shifts[0].  The 2025 tables stop at
    ntheta + 1 rows and have nothing for the retakes, which is why the scales
    fitted against spy/spz exist at all.
    """
    if j < len(shifts):
        return np.asarray(shifts[j] - shifts[0], dtype='float64')
    if scales is None:
        raise SystemExit(
            f'frame {j} is past the {len(shifts)}-row shift table and the EDF '
            f'header has no spy/spz to fall back on -- cannot tell where the '
            f'sample was parked for this retake')
    m, o0 = motors(lay, k, j), motors(lay, k, 0)
    return np.array([scales[0] * (m['spy'] - o0['spy']),
                     scales[1] * (m['spz'] - o0['spz'])])


# ---------------------------------------------------------------------------
# drift measurement
# ---------------------------------------------------------------------------

def local_peak(cc, r):
    """Sub-pixel peak of cc within r px of zero lag, as (drow, dcol, height)."""
    idx = np.arange(-r, r + 1)
    sub = cc[np.ix_(idx % cc.shape[0], idx % cc.shape[1])]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    out = []
    for ax in (0, 1):
        lo = sub[k[0] - 1, k[1]] if ax == 0 else sub[k[0], k[1] - 1]
        hi = sub[(k[0] + 1) % sub.shape[0], k[1]] if ax == 0 \
            else sub[k[0], (k[1] + 1) % sub.shape[1]]
        den = lo - 2 * sub[k] + hi
        out.append(k[ax] + (0.5 * (lo - hi) / den if den else 0.0) - r)
    return out[0], out[1], float(sub[k])


def measure(sc, static, shifts, retake, scan_j, disp_retake, args):
    """Drift of the sample at frame scan_j, in correct.txt pixels.

    Both frames are moved back to frame 0's piezo position, so what is left
    between them is the drift.  Sign: the correlation maps the scan frame onto
    the retake, i.e. it returns (retake - scan_frame); the drift the pipeline
    wants is where the sample WAS relative to the end of the scan, hence the
    negation in the caller.

    Two levels of aggregation, and both are needed.  WITHIN one crop the frame
    is cut into --bands horizontal strips, each correlated separately, and the
    surfaces are SUMMED before the peak is taken -- coherent averaging, so the
    sample peak builds up and the noise does not.  Peaking each band and taking
    a median of the peaks instead lets a band that saw nothing vote as loudly
    as one that saw the sample, which on the 20 nm scan moved the omega=0
    vertical anywhere between 0.9 and 3.6 px depending on --bands.  ACROSS
    crops the summed-surface estimates are combined by median, because a single
    crop can still land on a wrong local peak when the pair is weak: the 6 nm
    omega=90 pair reads +4.1 / +4.2 / +4.3 px at crops 1024 / 1536 / 2048 but
    +0.96 at crop 2048 with one band.  The scatter across crops is the honest
    error bar and is reported as such.
    """
    n = sc.n
    fy, fx = sc._fy, sc._fx

    def prep(j, dy, dx):
        img = sc.frame(j) - static
        ph = np.exp(2j * np.pi * (fy * dy + fx * dx))
        return np.real(np.fft.ifft2(np.fft.fft2(img) * ph))

    A = prep(retake, disp_retake[1], disp_retake[0])
    B = prep(scan_j, shifts[scan_j, 1], shifts[scan_j, 0])

    est = []
    for crop in args.crop:
        lo, hi = (n - crop) // 2, (n + crop) // 2
        a, b = A[lo:hi, lo:hi], B[lo:hi, lo:hi]
        h = crop // args.bands          # equal bands; surfaces must match shape
        total = None
        for ib in range(args.bands):
            sl = slice(ib * h, (ib + 1) * h)
            cc = ec.phase_corr(a[sl], b[sl])
            # The exact zero-lag bin is unusable and must be interpolated over.
            # Any WHITE noise shared between the two prepared frames lands
            # entirely in cc[0,0] and nowhere else, and here it is shared with
            # a minus sign, so it digs a one-pixel hole rather than adding a
            # peak.  Two channels do it.  (a) The scan frame is one of the
            # frames averaged into `static`, so subtracting the template puts
            # -1/template of that frame's photon noise into the retake, against
            # +1 of it in the frame itself.  build_template() below keeps the
            # compared frames out of the average, which kills this one at
            # source.  (b) Unavoidable: after subtracting a template that
            # averages over the whole scan, the frame at the start carries
            # +1/2 log(ref0) - 1/2 log(ref1) of flat noise and the retake at
            # the end carries exactly the negative of it.
            # On the 6 nm omega=0 pair the hole was deep enough to invert the
            # peak -- cc[0,0] = -0.025 with neighbours at +0.017 -- so the peak
            # finder took a ring lobe and reported dx = -1.41 for a pair whose
            # profile is symmetric about zero.  The sample peak is several px
            # wide, so replacing the single centre bin by its neighbours costs
            # nothing real: a genuine zero shift still reads as a local maximum
            # there, just without its spike.
            cc[0, 0] = 0.25 * (cc[0, 1] + cc[0, -1] + cc[1, 0] + cc[-1, 0])
            total = cc if total is None else total + cc
        total /= args.bands
        dy, dx, pk = local_peak(total, args.search)
        gy, gx, gpk = ec.peak(total)
        est.append(dict(crop=crop, dy=dy, dx=dx, peak=pk,
                        gy=gy, gx=gx, gpeak=gpk, cc=total))
    return est, est[len(est) // 2]['cc']


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--path', help="override the config's raw-data root")
    ap.add_argument('--pfile', help="override the config's scan prefix")
    ap.add_argument('--dist', type=int, default=1,
                    help='1-based distance plane whose retakes are measured')
    ap.add_argument('--crop', default='1024,1280,1536,1792,2048,2304',
                    type=lambda v: [int(x) for x in v.split(',')],
                    help='comma-separated central crops, in bin-0 px.  Each '
                         'gives one estimate and the median is used, so this '
                         'wants several values, not one.  Do not add crops '
                         'much above 2048: the flat-fielding near the frame '
                         'edges is poor enough that at 3072 the 20 nm omega=0 '
                         'vertical jumps to a wrong peak.')
    ap.add_argument('--bands', type=int, default=4,
                    help='horizontal bands, correlated separately and summed')
    ap.add_argument('--search', type=int, default=40,
                    help='peak is searched within this many px of zero; the '
                         'drift is physically a few px, and a wider window only '
                         'invites the residual-illumination peak')
    ap.add_argument('--template', type=int, default=32,
                    help='frames spread over the scan that build the static '
                         'illumination template')
    ap.add_argument('--nflat', type=int, default=8, help='flats/darks averaged')
    ap.add_argument('--min-peak', type=float, default=0.010,
                    help='warn if the summed correlation peak is weaker than '
                         'this; a locked-on retake gives ~0.02-0.03')
    ap.add_argument('--zero-x', action='store_true',
                    help='write zero in the horizontal column.  The horizontal '
                         'measurement itself is sound -- on the 20 nm scan it '
                         'gives +0.11 px at omega=0 against quali.mat\'s '
                         '+0.130 -- but it is the noisier of the two and the '
                         'horizontal column of the real correct_motion.txt is '
                         'not what the retakes measure (see --validate).  Use '
                         'this to keep only the vertical and leave the '
                         'horizontal to the position refinement in step 6.')
    ap.add_argument('--peter-scale', action='store_true',
                    help="write the drift in bin-2 px, reproducing Peter's "
                         'file byte-for-byte instead of what step 3 needs')
    ap.add_argument('--bin-factor', type=float, default=2.0,
                    help='binning quali.mat is expressed in; only used by '
                         '--peter-scale and --validate')
    ap.add_argument('--validate', action='store_true',
                    help='compare against quali.mat and the existing '
                         'correct_motion.txt, if the scan has them')
    ap.add_argument('--out', default='correct_motion.txt',
                    help='where to write the result.  Deliberately the CURRENT '
                         'directory by default: step 3 reads it from the raw '
                         'scan dir on eagle, and overwriting raw beamtime data '
                         'should be a deliberate copy, not a side effect.')
    ap.add_argument('--fig', default='motion_estimate.png')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(args.config, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']
    lay = Layout((args.path or cfg.get('path')).rstrip('/'),
                 args.pfile or cfg.get('pfile'))
    kdist = args.dist - 1
    dname = lay.dname(kdist)
    ntheta = lay.ntheta

    print(f'scan     : {dname}   ({lay.flavour})')
    print(f'ntheta   : {ntheta}')
    shifts = np.loadtxt(lay.shift_source(kdist))
    sc = ec.Scan(lay, args.nflat, dist=kdist, shifts=shifts)
    print(f'frame    : {sc.n} x {sc.n}   shifts: {lay.shift_source(kdist)} '
          f'{shifts.shape}')

    scales = px_per_um(lay, kdist, shifts)
    if scales is None:
        print('scale    : EDF headers carry only somega, so the retake '
              'displacement is taken from the shift table itself')
    else:
        print(f'scale    : col0 = {scales[0]:+.3f} px/um * spy,  '
              f'col1 = {scales[1]:+.3f} px/um * spz'
              f'   (voxel {1e3 / abs(scales[0]):.2f} nm)')

    retakes = find_retakes(lay, kdist, ntheta)
    if not retakes:
        raise SystemExit(
            f'no post-scan retake frames after {ntheta} in {dname}.\n'
            'Without them there is nothing to measure the drift against; '
            'leave correct_motion.txt absent and let step 6 refine positions.')
    print(f'retakes  : ' + ', '.join(f'{j} (omega={w:.0f} -> frame {js})'
                                     for j, js, w in retakes))

    # --- static illumination template ------------------------------------
    # Spread over the scan so the sample smears over its own +-300 px of random
    # displacement while the detector-fixed illumination does not.  Frames that
    # will be correlated are nudged out of the average: including one puts a
    # negative copy of its photon noise into the frame it is compared with, and
    # that digs a hole at exactly zero lag (see measure()).
    used = {js for _, js, _ in retakes} | {0, ntheta}
    tj = []
    for j in np.linspace(0, ntheta, args.template, dtype=int):
        j = int(j)
        while j in used or j in tj:
            j += 7 if j < ntheta - 7 else -7
        tj.append(j)
    print(f'template : mean of {len(tj)} frames spread over the scan, '
          f'excluding {sorted(used)}', flush=True)
    static = np.zeros((sc.n, sc.n), dtype='float32')
    for j in tj:
        static += sc.frame(j)
    static /= len(tj)

    # --- measure each retake ----------------------------------------------
    pts_j, pts_d, all_est, panels, sd = [], [], [], {}, []
    print(f'\n{"retake":>8} {"frame":>6} {"crop":>6} {"dy":>8} {"dx":>8} '
          f'{"peak":>8} | {"illum dy":>9} {"dx":>9} {"peak":>8}')
    for j, js, omega in retakes:
        disp = retake_disp(lay, kdist, shifts, scales, j)
        est, ab = measure(sc, static, shifts, j, js, disp, args)
        for e in est:
            print(f'{j:8d} {js:6d} {e["crop"]:6d} {e["dy"]:+8.2f} '
                  f'{e["dx"]:+8.2f} {e["peak"]:8.4f} | {e["gy"]:+9.1f} '
                  f'{e["gx"]:+9.1f} {e["gpeak"]:8.4f}', flush=True)
        dy = float(np.median([e['dy'] for e in est]))
        dx = float(np.median([e['dx'] for e in est]))
        pk = float(np.median([e['peak'] for e in est]))
        udy = float(np.median(np.abs([e['dy'] - dy for e in est]))) * 1.4826
        udx = float(np.median(np.abs([e['dx'] - dx for e in est]))) * 1.4826
        sd.append((udx, udy))
        print(f'{"":8} {"":6} {"median":>6} {dy:+8.2f} {dx:+8.2f} {pk:8.4f}'
              f'   +- {udy:.2f} / {udx:.2f} px across crops')
        if pk < args.min_peak:
            print(f'{"":8} WARNING: peak height {pk:.4f} is below --min-peak '
                  f'{args.min_peak}; a pair that locked onto the sample gives '
                  f'~0.02.  Trust this point only if the spread above is small.')
        if abs(est[-1]['gy'] - est[-1]['dy']) + abs(est[-1]['gx'] - est[-1]['dx']) > 1.0:
            print(f'{"":8} note: the global peak is elsewhere (dy '
                  f'{est[-1]["gy"]:+.1f}, dx {est[-1]["gx"]:+.1f}, height '
                  f'{est[-1]["gpeak"]:.4f}) -- residual illumination, which '
                  f'sits at the two frames\' displacement difference.')

        # the drift the pipeline wants is where the sample was relative to the
        # end of the scan, i.e. minus what the retake sees
        pts_j.append(js)
        pts_d.append([-dx, -dy])          # col0 = x, col1 = y
        all_est.append((j, js, omega, est, dy, dx))
        panels[js] = ab

    # frame ntheta is its own retake: zero drift, by construction
    pts_j.append(ntheta)
    pts_d.append([0.0, 0.0])
    pts_j = np.array(pts_j, dtype=float)
    pts_d = np.array(pts_d)
    sd.append((0.0, 0.0))                     # the omega=180 anchor is exact
    order = np.argsort(pts_j)
    pts_j, pts_d = pts_j[order], pts_d[order]
    sd = [sd[i] for i in order]

    # --- fit and build the file -------------------------------------------
    sdA = np.array(sd)
    if len(sdA):
        print(f'\nmeasurement uncertainty (MAD across crops): dx '
              f'{sdA[:, 0].max():.2f} px, dy {sdA[:, 1].max():.2f} px  '
              f'(a fitted amplitude below this is not measured)')
    deg = min(len(pts_j) - 1, 2)
    jj = np.arange(len(shifts), dtype=float)
    drift = np.empty((len(shifts), 2))

    def fit(vals, k):
        q = np.polyval(np.polyfit(pts_j, vals, deg), jj)
        return q - q.mean()

    for k in range(2):
        drift[:, k] = fit(pts_d[:, k], k)

    # Error band: the quadratic runs exactly through three points, so each
    # point's uncertainty moves the whole curve.  Refit with each point pushed
    # by +-its own MAD-across-crops and keep the envelope.  On the 6 nm scan the
    # omega=90 point carries most of it, which is why the band is worth drawing.
    band = np.zeros((len(shifts), 2))
    for i in range(len(pts_j)):
        u = np.array(sd[i]) if i < len(sd) else np.zeros(2)   # last point is the anchor
        for k in range(2):
            if not u[k]:
                continue
            v = pts_d[:, k].copy(); v[i] += u[k]
            band[:, k] = np.maximum(band[:, k], np.abs(fit(v, k) - drift[:, k]))
    if args.zero_x:
        drift[:, 0] = 0.0
        print('\n--zero-x: horizontal column left at zero')
    if args.peter_scale:
        drift /= args.bin_factor
        print(f'\n--peter-scale: drift divided by {args.bin_factor} '
              f'(bin-2 px, as in his file)')

    motion = shifts + drift
    print(f'\ndrift (correct_motion - correct), {len(shifts)} rows, degree {deg}:')
    for k, nm in ((0, 'col0 (x, horizontal)'), (1, 'col1 (y, vertical)  ')):
        print(f'  {nm}: min {drift[:, k].min():+7.3f}  max {drift[:, k].max():+7.3f}'
              f'  peak-to-peak {np.ptp(drift[:, k]):6.3f}  rms {drift[:, k].std():.4f} px'
              f'   +- {band[:, k].max():.3f} px from the retake error bars')

    if len(sdA):
        for k, nm, u in ((0, 'col0 (x)', sdA[:, 0].max()), (1, 'col1 (y)', sdA[:, 1].max())):
            amp = np.ptp(drift[:, k])
            if amp and amp < u:
                print(f'  WARNING: {nm} amplitude {amp:.3f} px is below the '
                      f'{u:.2f} px band spread -- treat it as unmeasured'
                      + ('' if args.zero_x else '; consider --zero-x'
                         if k == 0 else ''))

    np.savetxt(args.out, motion, fmt='%.6f')
    print(f'\nwrote {args.out}  ({motion.shape[0]} rows x 2)')
    print(f'step 3 reads it from {dname}/correct_motion.txt -- copy it there '
          f'when you want it used:\n    cp {args.out} {dname}/')

    validation = validate(args, dname, shifts, drift) if args.validate else None
    make_figure(args, dname, ntheta, shifts, drift, band, motion, pts_j,
                pts_d, all_est, panels, validation)
    print(f'wrote {args.fig}')


def validate(args, dname, shifts, drift):
    """Compare with quali.mat / correct_motion.txt when the scan has them."""
    out = {}
    qp = f'{dname}/quali.mat'
    if os.path.exists(qp):
        rows, grab = [], False
        for line in open(qp):
            if line.startswith('# name:'):
                grab = line.split(':')[1].strip() == 'corr_imagesafterscan'
                continue
            if grab and line.strip() and not line.startswith('#'):
                rows.append([float(x) for x in line.split()])
        if rows:
            out['quali'] = np.array(rows)
            print('\nquali.mat corr_imagesafterscan (bin-2 px, Peter):')
            print(out['quali'])
    mp = f'{dname}/correct_motion.txt'
    if os.path.exists(mp):
        real = np.loadtxt(mp)[:len(shifts)]
        out['delta'] = real - shifts
        print('\nreal correct_motion.txt - correct.txt, against this estimate:')
        print(f'  {"":8} {"peter rms":>10} {"mine rms":>10} {"corr":>7} '
              f'{"mine/peter":>11} {"resid /bin":>11}')
        for k, nm in ((0, 'col0 (x)'), (1, 'col1 (y)')):
            d, e = out['delta'][:, k], drift[:, k]
            if not d.std() or not e.std():
                print(f'  {nm}: one column is flat, nothing to compare')
                continue
            r = np.corrcoef(d, e)[0, 1]
            scale = np.dot(d, e) / np.dot(d, d)
            print(f'  {nm} {d.std():10.4f} {e.std():10.4f} {r:+7.4f} '
                  f'{scale:11.3f} {(e / args.bin_factor - d).std():11.4f}')
        print(f'  A column that reproduces Peter\'s shows corr ~ +1 and '
              f'mine/peter ~ {args.bin_factor} (he writes bin-{int(args.bin_factor)} px, '
              f'this writes unbinned).')
    return out or None


def make_figure(args, dname, ntheta, shifts, drift, band, motion, pts_j,
                pts_d, all_est, panels, validation):
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.42,
                          wspace=0.26, left=0.06, right=0.985, top=0.9, bottom=0.07)

    # --- row 0: the correlation surfaces the numbers come from -------------
    # These, not the frame differences, are what show whether a pair locked on:
    # a real lock is a compact bright spot near zero, and its height relative to
    # the surrounding speckle is the peak column in the table above.
    R = 24
    def show_cc(ax, cc, title, dy, dx):
        idx = np.arange(-R, R + 1)
        sub = cc[np.ix_(idx % cc.shape[0], idx % cc.shape[1])]
        ax.imshow(sub, cmap='inferno', extent=[-R, R, R, -R],
                  interpolation='nearest')
        ax.plot(dx, dy, 'o', mfc='none', mec='cyan', ms=13, mew=1.8)
        ax.axhline(0, color='w', lw=0.5, alpha=0.4)
        ax.axvline(0, color='w', lw=0.5, alpha=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('dx [px]', fontsize=8); ax.set_ylabel('dy [px]', fontsize=8)
        ax.tick_params(labelsize=7)

    shown = 0
    for j, js, omega, est, dy, dx in all_est:   # dy/dx are across-crop medians
        if shown >= 2 or js not in panels:
            continue
        show_cc(fig.add_subplot(gs[0, shown]), panels[js],
                f'retake {j} vs frame {js}  (omega={omega:.0f})\n'
                f'dy={dy:+.2f}  dx={dx:+.2f} px, height '
                f'{np.median([e["peak"] for e in est]):.4f}', dy, dx)
        shown += 1

    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    lines = [f'scan   {os.path.basename(dname)}',
             f'ntheta {ntheta}',
             '',
             'drift points (px, relative to end of scan)',
             f'{"frame":>7} {"col0 x":>9} {"col1 y":>9}']
    for jj, dd in zip(pts_j, pts_d):
        lines.append(f'{int(jj):7d} {dd[0]:+9.3f} {dd[1]:+9.3f}')
    lines += ['', 'fitted quadratic, mean removed:',
              f'  col0 x  ptp {np.ptp(drift[:, 0]):.3f} px  rms {drift[:, 0].std():.4f}',
              f'  col1 y  ptp {np.ptp(drift[:, 1]):.3f} px  rms {drift[:, 1].std():.4f}',
              '', 'random displacement for comparison:',
              f'  col0 x  ptp {np.ptp(shifts[:, 0]):.1f} px',
              f'  col1 y  ptp {np.ptp(shifts[:, 1]):.1f} px']
    ax.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', family='monospace',
            fontsize=8.5, transform=ax.transAxes)

    # --- row 1: the fit ----------------------------------------------------
    jj = np.arange(len(shifts))
    # The fit is mean-removed, so the measured points have to be shifted by the
    # same offset to sit on the curve.
    for k, nm, col in ((0, 'col 0 — horizontal (x)', 'tab:blue'),
                       (1, 'col 1 — vertical (y)', 'tab:red')):
        ax = fig.add_subplot(gs[1, k])
        off = np.polyval(np.polyfit(pts_j, pts_d[:, k],
                                    min(len(pts_j) - 1, 2)), jj).mean()
        ax.fill_between(jj, drift[:, k] - band[:, k], drift[:, k] + band[:, k],
                        color=col, alpha=0.2, lw=0,
                        label='retake error bars')
        ax.plot(jj, drift[:, k], color=col, lw=1.6, label='fitted quadratic')
        ax.plot(pts_j, pts_d[:, k] - off, 'ko', ms=7, label='measured retakes',
                zorder=5)
        if validation and 'delta' in validation:
            ax.plot(jj, validation['delta'][:, k], 'k--', lw=1.2,
                    label="Peter's file")
            ax.plot(jj, validation['delta'][:, k] * args.bin_factor, 'k:',
                    lw=1.2, label=f"Peter's x {args.bin_factor:.0f}")
        ax.axhline(0, color='0.7', lw=0.7)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel('projection index'); ax.set_ylabel('drift [px]')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    for k, c, nm in ((0, 'tab:blue', 'col0 x'), (1, 'tab:red', 'col1 y')):
        ax.fill_between(jj, drift[:, k] - band[:, k], drift[:, k] + band[:, k],
                        color=c, alpha=0.18, lw=0)
        ax.plot(jj, drift[:, k], color=c, lw=1.5, label=nm)
    ax.axhline(0, color='0.7', lw=0.7)
    ax.set_title('correct_motion − correct', fontsize=10)
    ax.set_xlabel('projection index'); ax.set_ylabel('drift [px]')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- row 2: drift against the thing it corrects ------------------------
    ax = fig.add_subplot(gs[2, 0])
    m = 200
    ax.plot(jj[:m], shifts[:m, 0], color='tab:blue', lw=1, label='col0 x')
    ax.plot(jj[:m], shifts[:m, 1], color='tab:red', lw=1, label='col1 y')
    ax.set_title(f'random displacement, first {m} frames  (±300 px)', fontsize=10)
    ax.set_xlabel('projection index'); ax.set_ylabel('correct.txt [px]')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(drift[:, 0], drift[:, 1], color='0.3', lw=1.6)
    ax.plot(drift[0, 0], drift[0, 1], 'go', ms=7, label='frame 0')
    ax.plot(drift[-1, 0], drift[-1, 1], 'rs', ms=7, label=f'frame {len(jj)-1}')
    ax.set_title('drift trajectory', fontsize=10)
    ax.set_xlabel('col0 x [px]'); ax.set_ylabel('col1 y [px]')
    ax.axis('equal'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    for (j, js, omega, est, dy, dx), c in zip(all_est, ('tab:green', 'tab:purple')):
        cc = panels[js]
        k = np.arange(-12, 13)
        ax.plot(k, [cc[0, i % cc.shape[1]] for i in k], color=c, lw=1.3,
                label=f'omega={omega:.0f}, dx cut')
        ax.plot(k, [cc[i % cc.shape[0], 0] for i in k], color=c, lw=1.3,
                ls='--', label=f'omega={omega:.0f}, dy cut')
    ax.axvline(0, color='0.7', lw=0.7)
    ax.set_title('correlation cuts through zero lag', fontsize=10)
    ax.set_xlabel('lag [px]'); ax.set_ylabel('correlation')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f'Sample drift during the scan — {os.path.basename(dname)}',
                 fontsize=13)
    fig.savefig(args.fig, dpi=110)


if __name__ == '__main__':
    main()
