"""Rotation axis from opposed pairs of PHASE-RETRIEVED projections.

WHY NOT RAW FRAMES.  The axis is a mirror symmetry: projection(theta) and
projection(theta+180) are reflections of each other about the axis column, so
correlating one against the mirrored other gives twice the offset.  That is
true of the PHASE.  It is not true of an ID16A hologram: at 4.5 nm the frames
are dominated by Fresnel fringes, which belong to the propagation geometry and
not to the object, so they do not mirror.  Correlating raw frames correlates
fringe against mirrored fringe.  Measured on AtomiumS1 FT: the mirror test on
stitched frames landed 0.35 px from ESRF's nabu value (+10.39 vs +10.74), while
focus/entropy on reconstructions was 7-12 px out.

WHERE IT SITS.  Step 4b: after step 4, before step 5's Paganin.  It needs both
halves -- step 4's amplitude-corrected pdata{k}_0, and step 5's stitch -- so it
runs a copy of step 5's `_stitch` at bin 0 followed by step 5's multiPaganin,
on just the handful of angles the pairs need (2*pairs out of ntheta).

NO CIRCULARITY, and nothing to undo in step 4.  The axis is a single constant
added to the horizontal column of EVERY plane and EVERY angle, so it slides the
stitched object sideways without changing how the planes register against each
other.  Step 4 stitched with whatever cshifts_final held; this measures what is
LEFT OVER after those same shifts, which is the correction, and step 5 reads
cshifts_final again.  Step 4's own output is detector-space and uses the shifts
only for slowly-varying amplitude matching, so it does not need redoing.

IDEMPOTENT BY CONSTRUCTION.  Because it measures a residual, a second run over
already-corrected shifts measures ~0 and adds ~0.  That is why step 4b caches
nothing: skipping on a cached value, or re-adding one, are the two ways to
double-count the axis, and measuring the residual every time does neither.

180-DEGREE SCANS.  AtomiumS1 is 0..179.955 deg, so a pair (j, ntheta-1-j) is
opposed only near j=0; by j=2 it is 0.2 deg short.  A residual rotation
decorrelates the pair, it does not translate it, so this costs peak height and
not accuracy -- but it is why `pairs` stays small and why dtheta is logged.

THE DRIFT IS STILL IN HERE.  Opposed pairs live at the two ends of the scan, so
the full sample drift separates them; no phase retrieval changes that.  What
saves it is that the drift is ~114 raw px VERTICAL and ~0 horizontal, and the
peak's horizontal lag is read AT the measured vertical lag.  The drift costs
overlap, not bias.  `max_dy` is a sanity bound on that vertical lag, nothing
more; the dy spread is logged unconditionally so a run that only just passes is
visible.
"""

import h5py
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from holotomocupy.shift import Shift

from estimate_center import phase_corr, peak

# Re-measure once with the axis folded in, so the residual is in the same log
# rather than only in the next run's.  Costs a second pass over 2*pairs angles.
VERIFY = True


# ---------------------------------------------------------------------------
# bin-0 stitch + Paganin for one angle
# ---------------------------------------------------------------------------

def build_phase_fn(fpath, ref, shifts, shrink_nd, n, nobj, ndist,
                   norm_magnifications, distances, wavelength, voxelsize,
                   delta_beta, alpha=0.01):
    """get_phase(j) -> [nobj, nobj] float32: step 5's bin-0 stitch, then Paganin.

    `ref` is pref_0 and the data read is pdata{k}_0, both written by step 4, so
    this sees exactly what step 5 will see at bin 0.  Kept as a copy of step 5's
    `_stitch` rather than shared with it: `_stitch` is closed over a bin level's
    worth of local state inside step 5's loop, and hoisting it out to serve a
    diagnostic would put a refactor of the production path in the way.
    """
    npad = n // 16
    r = np.array(shifts, dtype='float32')

    cref = cp.array(ref)
    fwhm_ref = 17.0 * (n / 2048)                       # same low-pass as step 4
    sigma_ref = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
    cref_smooth = cp.stack([ndimage.gaussian_filter(cref[k], sigma_ref)
                            for k in range(ndist)])

    cl_shift = Shift(n, nobj, n, nobj)
    v = cp.linspace(0, 1, npad, endpoint=False)
    v = v**5 * (126 - 420*v + 540*v**2 - 315*v**3 + 70*v**4)

    def multi_paganin(data, dist_eff):
        fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
        fy = cp.fft.fftfreq(data.shape[-2], d=voxelsize).astype('float32')
        fx, fy = cp.meshgrid(fx, fy)
        num = 0
        den = 0
        for j in range(data.shape[0]):
            rad = cp.fft.fft2(data[j].astype('complex64'))
            taylor = 1 + wavelength * dist_eff[j] * cp.pi * delta_beta * (fx**2 + fy**2)
            num += taylor * rad
            den += taylor**2
        num /= len(dist_eff)
        den = den / len(dist_eff) + alpha
        return cp.log(cp.real(cp.fft.ifft2(num / den))) * delta_beta * 0.5

    def get_phase(j):
        srdata = cp.zeros([ndist, nobj, nobj], dtype='float32')
        with h5py.File(fpath, 'r') as fid:
            data = cp.stack([cp.array(fid[f'/exchange/pdata{k}_0'][j]
                                      .astype('float32')) for k in range(ndist)])
        data_smooth = cp.stack([ndimage.gaussian_filter(data[k], sigma_ref)
                                for k in range(ndist)])
        rdata = data_smooth / (cref_smooth + 1e-5)

        for k in range(ndist - 1, -1, -1):
            shrink_jk = shrink_nd[j, k]
            eff_mag_jk = float(norm_magnifications[k]) / (1 + shrink_jk)
            mag = cp.array(1.0 / eff_mag_jk, dtype='float32')[None]
            tmp = cl_shift.curlySback(
                cp.log(rdata[k].astype('complex64')[None]).astype('complex64'),
                cp.array(r[j:j+1, k]), mag)[0].real
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
                denom = tmp[pady0:-pady1, padx0:-padx1].mean() + 1e-10
                tmp *= float(srdata[k+1][pady0:-pady1, padx0:-padx1].mean() / denom)
                if k == 0:
                    cs = min(nobj // 16, (nobj - pady0 - pady1) // 2,
                             (nobj - padx0 - padx1) // 2)
                    ch = cs // 2
                    ys = [pady0, nobj // 2 - ch, nobj - pady1 - cs]
                    xs = [padx0, nobj // 2 - ch, nobj - padx1 - cs]
                    prev = srdata[k + 1]
                    R = cp.array([[float(prev[y:y+cs, x:x+cs].mean() /
                                         (tmp[y:y+cs, x:x+cs].mean() + 1e-10))
                                   for x in xs] for y in ys], dtype='float32')
                    tmp *= ndimage.zoom(R, nobj / 3, order=1)[:nobj, :nobj]
                wx = cp.ones(nobj, dtype='float32')
                wy = cp.ones(nobj, dtype='float32')
                wx[:padx0] = 0
                wx[padx0:padx0+npad] = v
                wx[-padx1-npad:-padx1] = 1 - v
                wx[-padx1:] = 0
                wy[:pady0] = 0
                wy[pady0:pady0+npad] = v
                wy[-pady1-npad:-pady1] = 1 - v
                wy[-pady1:] = 0
                w = cp.outer(wy, wx)
                tmp = tmp * w + srdata[k+1] * (1 - w)
            srdata[k] = tmp

        pad8 = nobj // 8
        pj = cp.pad(srdata, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
        dist_eff = (distances * (1 + shrink_nd[j].mean(axis=-1))**2
                    / norm_magnifications**2)
        phase = multi_paganin(pj, dist_eff)
        return phase[pad8:pad8+nobj, pad8:pad8+nobj].get().astype('float32')

    return get_phase


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def measure_axis(get_phase, ntheta, nobj, theta=None, pairs=3, bands=3,
                 crop=None, max_dy=200.0, log=print):
    """Axis offset, in object px, from `pairs` opposed pairs of phase images.

    Returns a dict; writes nothing.  'center' is in the units
    rotation_center_shift has always been in, so it can be added straight to
    cshifts_final[..., 1].
    """
    crop = int(crop or nobj // 2) // 2 * 2
    # Symmetric about the grid centre, so lo + hi == nobj and the mirror centre
    # of the crop is the mirror centre of the full grid -- that is what lets the
    # raw lag be halved below without a crop-dependent correction.
    lo, hi = (nobj - crop) // 2, (nobj + crop) // 2
    edges = np.linspace(0, crop, bands + 1, dtype=int)
    rows, prof_sh = [], []

    for j in range(pairs):
        kk = ntheta - 1 - j
        dth = (180.0 * (kk - j) / (ntheta - 1) if theta is None
               else float(theta[kk] - theta[j]))
        if abs(dth - 180.0) > 0.5:
            log(f'  WARNING pair {j}/{kk} is {dth:.3f} deg apart, not 180')
        a = get_phase(j)[lo:hi, lo:hi]
        b = get_phase(kk)[lo:hi, lo:hi][:, ::-1]
        a = a - np.median(a)
        b = b - np.median(b)

        # 1-D profile match as a second opinion: phase correlation whitens, and
        # a whitened 2-D surface can lock onto a static component (see the
        # proj_bin trap).  Collapsing to a row profile keeps the amplitudes.
        pa, pb = a.mean(axis=0), b.mean(axis=0)
        pa, pb = pa - pa.mean(), pb - pb.mean()
        cc1 = np.real(np.fft.ifft(np.fft.fft(pa) * np.conj(np.fft.fft(pb))))
        t1 = int(np.argmax(cc1))
        t1 = t1 - crop if t1 > crop // 2 else t1
        prof_sh.append((t1 - 1) / 2)

        for ib in range(bands):
            r0, r1 = edges[ib], edges[ib + 1]
            dr, t, pk = peak(phase_corr(a[r0:r1], b[r0:r1]))
            # ::-1 on an even-length axis maps x -> N-1-x, so the mirror centre
            # is (N-1)/2: hence (t - 1)/2 and not t/2.
            shift = (t - 1) / 2
            rows.append(dict(j=j, k=kk, band=ib, dtheta=dth, dy=dr, t=t,
                             peak=pk, shift=shift))
            log(f'  pair {j:4d}/{kk:4d} ({dth:7.3f} deg) band {ib}: '
                f'dy={dr:+7.2f}  t={t:+8.2f}  peak={pk:.4f}  '
                f'shift={shift:+8.2f} px')

    sh = np.array([r['shift'] for r in rows])
    dy = np.array([r['dy'] for r in rows])
    log(f'dy over all {len(dy)} estimates: {dy.min():+.1f}..{dy.max():+.1f} px '
        f'(median {np.median(dy):+.1f}), max_dy={max_dy}')
    keep = np.abs(dy) <= max_dy
    if keep.sum() < 3:
        raise SystemExit(
            f'estimate_axis_paganin: only {keep.sum()} of {len(sh)} estimates '
            f'have |dy| <= {max_dy}.  dy carries the slow sample drift, which '
            f'is not undone here -- opposed frames sit at opposite ends of the '
            f'scan.  If the spread above is merely larger than max_dy, raise '
            f'it; a vertical lag far beyond the drift you expect means the '
            f'pairs are not correlating at all, and no threshold fixes that.')
    med = np.median(sh[keep])
    mad = np.median(np.abs(sh[keep] - med))
    keep &= np.abs(sh - med) <= max(3 * 1.4826 * mad, 2.0)
    n_dy = int((np.abs(dy) > max_dy).sum())
    log(f'rejected {n_dy} on |dy| > {max_dy} px, '
        f'{len(sh) - keep.sum() - n_dy} more at 3 MAD; {keep.sum()}/{len(sh)} kept')

    center = float(sh[keep].mean())
    prof = float(np.median(prof_sh))
    log(f'rotation_center_shift = {center:+.2f} +- {sh[keep].std():.2f} px '
        f'(median {np.median(sh[keep]):+.2f})')
    # Disagreement means the 2-D peak is not on the object; believe neither.
    _m = (f'1-D profile cross-check: {prof:+.2f} px (2-D {center:+.2f}, '
          f'differ by {abs(prof - center):.2f})')
    log(_m if abs(prof - center) <= 5.0 else 'WARNING ' + _m)

    return dict(center=center, std=float(sh[keep].std()), profile=prof,
                rows=rows, sh=sh, keep=keep, nkept=int(keep.sum()),
                ntotal=len(sh), crop=crop)
