#!/usr/bin/env python
"""Dose-matched comparison of a multi-distance and a single-distance scan.

The beam diverges from the focus, so the fluence in the sample falls as 1/z1^2
and a projection at the far distance costs only a fraction of one at the near
distance.  Summed over the four standard distances a multi-distance projection
costs 2.697 near-distance ones, so a single-distance scan may spend that budget
on 2.697x as many angles instead.  This script puts the two reconstructions --
same sample, same probe, same displacements, same total dose -- side by side.

    python compare_dose.py --root /data3/vnikitin/dose_study \\
        --ndist 4 --ntheta 450 --amp 16 --prb-smooth 2

--ntheta1 defaults to the dose-matched angle count rounded to the nearest
multiple of --np (matching run_dose.sh); pass it explicitly if the run used
something else.  One figure lands in --out:

  dose_ndist<N>_n<n>_ntheta<nt>_amp<a>[_prbs<s>].png   ground truth and the two
                                     reconstructions,
                        the horizontal (z = mid, tomographic) slice on the top
                        row and the error against the phantom below
  ..._zoom.png          the same slice of the two reconstructions over the
                        --zoom detail region only, one above the other, so the
                        multi- and single-distance textures can be compared at
                        1:1

Serial, no MPI -- run it after run_dose.sh.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import (read_summary, rec_dirs, read_conv,   # noqa: E402
                    last_checkpoint, read_slices)  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', default='/data3/vnikitin/dose_study')
    p.add_argument('--ndist',  type=int,   default=4,
                   help='distances in the multi-distance scan')
    p.add_argument('--ntheta', type=int,   default=450,
                   help='angles per distance in the multi-distance scan')
    p.add_argument('--ntheta1', type=int,  default=None,
                   help='angles of the single-distance scan (default: the '
                        'dose-matched count, rounded to a multiple of --np)')
    p.add_argument('--np',     type=int,   default=3,
                   help='ranks the runs were split over, for that rounding')
    p.add_argument('--amp',        type=float, default=16.0)
    p.add_argument('--prb-smooth', type=float, default=2.0)
    p.add_argument('--obj-smooth', type=float, default=C.OBJ_SMOOTH)
    p.add_argument('--n', type=int, default=None,
                   help='detector size to plot (default: whatever is on disk)')
    p.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument('--tag', default=None)
    p.add_argument('--cmap', default='gray')
    p.add_argument('--clip', type=float, default=99.8,
                   help='percentile of |ground truth| for the shared colour range')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--err-clip', type=float, default=99.5,
                   help='percentile of |error| for the error-row colour range')
    p.add_argument('--crop', type=float, default=0.10,
                   help='fraction trimmed off each side of the slices')
    p.add_argument('--zoom', default='680,450,200,200',
                   help='x0,y0,w,h of the detail region of the second figure, '
                        'in pixels of the *uncropped* slice (--crop does not '
                        'move it); "off" skips that figure')
    p.add_argument('--zoom-box', default='off', choices=['on', 'off'],
                   help='also outline the --zoom region on the reconstruction '
                        'panels of the main figure')
    p.add_argument('--conv', default='on', choices=['on', 'off'],
                   help='draw the convergence curves in the empty slot under the '
                        'ground truth (conv.csv of each reconstruction)')
    p.add_argument('--dpi', type=int, default=300,
                   help='output resolution. The panels are ~2.6 in wide and hold a '
                        'slice of ~nobj px, so 140 dpi (the old value) resolved barely '
                        'a third of it; 300 is close to 1:1 at n=1024 and 400 is 1:1 at '
                        'n=2048. Raise it further only if you intend to zoom in -- the '
                        'file grows as dpi^2.')
    return p.parse_args()


a = parse()
w = C.dose_weights(C.Z1_ALL[:a.ndist])
if a.ntheta1 is None:
    exact = C.dose_equivalent_ntheta(a.ntheta, C.Z1_ALL[:a.ndist])
    a.ntheta1 = int(round(exact / a.np)) * a.np

# (label, ndist, ntheta, relative dose) of the two scans
def plural(k):
    return f'{k} distance' + ('s' if k > 1 else '')


runs = [(plural(a.ndist), a.ndist, a.ntheta, float(w.sum()) * a.ntheta),
        (plural(1),       1,       a.ntheta1, a.ntheta1 * 1.0)]

zoom = None
if a.zoom and a.zoom.lower() != 'off':
    try:
        zx, zy, zw, zh = (int(v) for v in a.zoom.replace('x', ',').split(','))
    except ValueError:
        raise SystemExit(f'--zoom wants x0,y0,w,h in pixels -- got {a.zoom!r}')
    zoom = (zx, zy, zw, zh)


def load(ndist, ntheta):
    """(slices, summary, ground truth, detector n, rec dir or a note) of one case."""
    case = os.path.join(a.root,
                        C.dose_case_name(ndist, ntheta, a.amp, a.prb_smooth, a.obj_smooth))
    if not os.path.isdir(case):
        return None, {}, None, None, 'no data'
    got = {k: d for k, d in rec_dirs(case).items()
           if (a.n is None or k[0] == a.n) and last_checkpoint(d) is not None}
    if not got:
        return None, {}, None, None, 'not reconstructed'
    # the rec dir carries its own ntheta, and the dose-matched pair differs in
    # exactly that -- so prefer the one that matches the scan being loaded
    key = next((k for k in got if k[1] == ntheta), sorted(got)[0])
    rdir = got[key]
    # summary.txt appears only once rec.py is through its last iteration; a
    # half-converged checkpoint next to a converged one would just mislead
    if not os.path.isfile(os.path.join(rdir, 'summary.txt')):
        return None, {}, None, key[0], 'still running'
    gt = None
    data_h5 = os.path.join(case, 'data.h5')
    if os.path.isfile(data_h5):
        gt = read_slices(data_h5)
    return read_slices(last_checkpoint(rdir)), read_summary(rdir), gt, key[0], rdir


cases = []
for lbl, ndist, ntheta, dose in runs:
    sl, s, gt, dn, note = load(ndist, ntheta)
    print(f'{lbl:14s} ntheta={ntheta:5d}  '
          + (f'{note}' if sl is None else f'{note}  NRMSE={float(s.get("nrmse_obj", "nan")):.4f}'))
    cases.append(dict(lbl=lbl, ndist=ndist, ntheta=ntheta, dose=dose,
                      sl=sl, s=s, gt=gt, n=dn,
                      rdir=note if sl is not None else None))

if all(c['sl'] is None for c in cases):
    raise SystemExit('neither reconstruction is finished — nothing to plot yet')

# both scans are of the same phantom, so either data.h5 serves as the reference
gt = next((c['gt'] for c in cases if c['gt'] is not None), None)
n = next((c['n'] for c in cases if c['n']), None) or \
    next(c['sl'][0].shape[-1] for c in cases if c['sl'] is not None)

def truth_for(c):
    """Ground truth on the same grid as this case's reconstruction, or None.

    A data.h5 regenerated after its own run -- an auto-sized nobj that has since
    changed, say -- no longer describes the grid the reconstruction lives on, and
    the two legs of an old pair need not even agree with each other.  Take the
    case's own truth when it fits, the other leg's when it does not, and give up
    rather than subtract two different grids.
    """
    if c['sl'] is None:
        return None
    for g in (c['gt'], gt):
        if g is not None and g[0].shape == c['sl'][0].shape:
            return g
    print(f"!! {c['lbl']}: the reconstruction is {c['sl'][0].shape[-1]} px but the "
          f'ground truth on disk is not -- data.h5 was regenerated after the run, '
          f'so this error panel is dropped')
    return None


ref = gt if gt is not None else next(c['sl'] for c in cases if c['sl'] is not None)
lim = np.percentile(np.abs(ref[0]), a.clip)
vmin = -lim if a.vmin is None else a.vmin
vmax = lim if a.vmax is None else a.vmax


def crop(img, frac=a.crop):
    """Trim `frac` off each side — the object never reaches the field edge."""
    if frac <= 0:
        return img
    cy, cx = (int(round(frac * s)) for s in img.shape[:2])
    return img[cy:img.shape[0] - cy, cx:img.shape[1] - cx]


# the error row shares one range too, so the two columns are comparable
truths = {id(c): truth_for(c) for c in cases}
errs = [c['sl'][0] - truths[id(c)][0] for c in cases if truths[id(c)] is not None]
elim = np.percentile(np.abs(np.concatenate([e.ravel() for e in errs])), a.err_clip) if errs else 1.0

show_err = bool(errs)
ncol = 1 + len(cases) if gt is not None else len(cases)
# the second row carries the error panels; with no ground truth it exists only to
# hold the convergence curves, and the rest of it is blanked below
nrow = 2 if (show_err or a.conv == 'on') else 1
fig, ax = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.9 * nrow + 0.9),
                       squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.12))

col = 0
# the bottom-left slot has no error panel to hold, so it takes the convergence
# curves -- the one thing the still images cannot show, and what says whether a
# worse NRMSE is a worse solution or just an unfinished one
axc = ax[1, 0] if (nrow == 2 and a.conv == 'on') else None
if gt is not None:
    ax[0, col].imshow(crop(gt[0]), cmap=a.cmap, vmin=vmin, vmax=vmax)
    ax[0, col].set_title('ground truth', fontsize=9)
    if show_err and ax[1, col] is not axc:
        ax[1, col].axis('off')
    col += 1

im = ime = None
for c in cases:
    axo = ax[0, col]
    d0 = cases[0]['dose']
    ttl = (f"{c['lbl']}, {c['ndist'] * c['ntheta']} projections\n"
           f"dose = {c['dose'] / d0:.2f} ×")
    if c['sl'] is None:
        for spine in axo.spines.values():
            spine.set_visible(False)
        axo.text(0.5, 0.5, 'reconstruction\nnot finished', transform=axo.transAxes,
                 ha='center', va='center', fontsize=9, color='0.55')
    else:
        im = axo.imshow(crop(c['sl'][0]), cmap=a.cmap, vmin=vmin, vmax=vmax)
        ttl += f",  NRMSE = {float(c['s'].get('nrmse_obj', 'nan')):.3f}"
        if show_err and truths[id(c)] is not None:
            ime = ax[1, col].imshow(crop(c['sl'][0] - truths[id(c)][0]), cmap='coolwarm',
                                    vmin=-elim, vmax=elim)
        elif show_err and ax[1, col] is not axc:
            ax[1, col].axis('off')
    axo.set_title(ttl, fontsize=9)
    col += 1

if nrow == 2 and not show_err:
    for j in range(ncol):
        if ax[1, j] is not axc:
            ax[1, j].axis('off')

for r in range(nrow):
    for j in range(ncol):
        if ax[r, j] is not axc:
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
ax[0, 0].set_ylabel('horizontal\n(z = mid)', fontsize=10)
if show_err and axc is not ax[1, 0]:
    ax[1, 0].set_ylabel('error\n(rec − truth)', fontsize=10)

if axc is not None:
    drawn = 0
    for c, colour in zip(cases, ('tab:blue', 'tab:red')):
        cv = read_conv(c['rdir']) if c['rdir'] else None
        if cv is None:
            continue
        it, err, _ = cv
        m = (it >= 0) & np.isfinite(err) & (err > 0)   # iter = -1 is the initial guess
        if not m.any():
            continue
        axc.semilogy(it[m], err[m], '-o', ms=3, lw=1.3, color=colour,
                     label=f"{c['lbl']}, {c['ndist'] * c['ntheta']} proj")
        drawn += 1
    if drawn:
        axc.set_xlabel('BH iteration', fontsize=8)
        axc.set_ylabel(r'data misfit  $\||\Psi|-\sqrt{I}\|^2/N$', fontsize=8)
        axc.set_title('convergence', fontsize=9)
        axc.grid(True, which='both', lw=0.3, alpha=0.4)
        axc.tick_params(labelsize=7)
        axc.legend(fontsize=7, frameon=False)
        # square like the image panels beside it, so the row does not look ragged
        axc.set_box_aspect(1)
    else:
        axc.axis('off')
        axc = None

if im is not None:
    fig.colorbar(im, ax=ax[0].tolist(), fraction=0.012, pad=0.008,
                 label=r'Re(obj) = $-\delta$')
if ime is not None:
    fig.colorbar(ime, ax=ax[1].tolist(), fraction=0.012, pad=0.008, label='error')

meta = next((c['s'] for c in cases if c['sl'] is not None), {})
# only when it is not the phantom as it has always been -- otherwise it is one
# more constant on every figure
objs = abs(a.obj_smooth - C.OBJ_SMOOTH) > 1e-6
objs_txt = f'object $\\sigma$ = {a.obj_smooth:g} voxel, ' if objs else ''
fig.suptitle(
    f"equal-dose comparison: amp = ±{a.amp:g} px, probe $\\sigma$ = {a.prb_smooth:g} px, "
    f"{objs_txt}"
    f"n={n}, {meta.get('niter', '?')} BH iterations\n"
    f"{plural(a.ndist)} × {a.ntheta} angles "
    f"(dose {float(w.sum()):.3f} × near-distance per projection)  vs  "
    f"1 distance × {a.ntheta1} angles", fontsize=12)

if zoom is not None and a.zoom_box == 'on':
    # the panels show the cropped slice, so the box moves with the crop
    zx, zy, zw, zh = zoom
    for c, axo in zip(cases, ax[0, (1 if gt is not None else 0):]):
        if c['sl'] is None:
            continue
        ny, nx = c['sl'][0].shape[:2]
        oy, ox = (int(round(a.crop * v)) for v in (ny, nx))
        axo.add_patch(plt.Rectangle((zx - ox - 0.5, zy - oy - 0.5), zw, zh,
                                    fill=False, ec='yellow', lw=0.8))

tag = (f'_n{n}_ntheta{a.ntheta}_amp{a.amp:g}'
       + (f'_prbs{a.prb_smooth:g}' if a.prb_smooth else '')
       + (f'_objs{a.obj_smooth:g}' if objs else '')
       + (f'_{a.tag}' if a.tag else ''))
png = os.path.join(a.out, f'dose_ndist{a.ndist}{tag}.png')
fig.savefig(png, dpi=a.dpi, bbox_inches='tight')
print(f'-> {png}')

# --- the detail figure: the same region of both reconstructions, stacked ---
if zoom is not None:
    zx, zy, zw, zh = zoom
    fz, az = plt.subplots(len(cases), 1, squeeze=False,
                          figsize=(3.4, 3.4 * len(cases) + 0.9),
                          gridspec_kw=dict(hspace=0.14))
    imz = None
    for r, c in enumerate(cases):
        axz = az[r, 0]
        axz.set_xticks([]); axz.set_yticks([])
        img = None if c['sl'] is None else c['sl'][0]
        if img is not None:
            ny, nx = img.shape[:2]
            y0, x0 = max(zy, 0), max(zx, 0)
            y1, x1 = min(zy + zh, ny), min(zx + zw, nx)
            if y1 <= y0 or x1 <= x0:
                print(f"!! {c['lbl']}: the zoom region {zx},{zy},{zw},{zh} lies "
                      f'outside the {nx}x{ny} px slice -- panel dropped')
                img = None
            else:
                if (y1 - y0, x1 - x0) != (zh, zw):
                    print(f"!! {c['lbl']}: the zoom region is clipped to the "
                          f'{nx}x{ny} px slice -- showing x {x0}:{x1}, y {y0}:{y1}')
                # nearest: at this magnification every pixel is the datum
                imz = axz.imshow(img[y0:y1, x0:x1], cmap=a.cmap,
                                 vmin=vmin, vmax=vmax, interpolation='nearest')
                axz.set_title(f"{c['lbl']}, {c['ndist'] * c['ntheta']} projections"
                              + (f",  NRMSE = {float(c['s'].get('nrmse_obj', 'nan')):.3f}"
                                 if c['s'] else ''), fontsize=9)
        if img is None:
            for spine in axz.spines.values():
                spine.set_visible(False)
            axz.text(0.5, 0.5, f"{c['lbl']}\nnot available", transform=axz.transAxes,
                     ha='center', va='center', fontsize=9, color='0.55')

    if imz is not None:
        # under the panels rather than beside them: the column is one panel
        # wide, and a vertical bar would sit on top of the lower title
        fz.colorbar(imz, ax=az[:, 0].tolist(), orientation='horizontal',
                    fraction=0.045, pad=0.03, label=r'Re(obj) = $-\delta$')
    fz.suptitle(f'detail: x {zx}:{zx + zw}, y {zy}:{zy + zh} of the '
                f'horizontal slice\n'
                f'amp = ±{a.amp:g} px, probe $\\sigma$ = {a.prb_smooth:g} px, '
                f'{objs_txt}n={n}',
                fontsize=11)
    pngz = os.path.join(a.out, f'dose_ndist{a.ndist}{tag}_zoom.png')
    fz.savefig(pngz, dpi=a.dpi, bbox_inches='tight')
    print(f'-> {pngz}')
