#!/usr/bin/env python
"""Step 3 of the study — one figure comparing the reconstructions.

Reads the last checkpoint of every case that has been reconstructed and lays the
middle slices of the refractive-index decrement out side by side.  A case is a
(displacement amplitude, probe smoothness) pair; either one, or both, can be
swept:

    # displacement sweep at the standard probe
    python make_figure.py --root /data3/vnikitin/disp_study --amps "0 1 2 4 8 16 32"

    # probe-smoothness sweep at +-16 px displacements
    python make_figure.py --root /data3/vnikitin/disp_study_probe1 \\
        --amps 16 --prb-smooths "0 0.225 0.5 1 2 4 8"

Reconstructions live in <case>/rec_n<n>_ntheta<ntheta>, so one root can hold
several detector sizes / angular samplings.  Only one of them is plotted at a
time: --n / --ntheta pick it, otherwise the one covering the most cases wins.
Two figures land in --out (this script's own folder by default):

  slices_ndist<N>_n<n>_ntheta<nt>[_amp<a>][_prbs<s>].png  ground truth + one
                                       column per case,
                        horizontal (z = mid, the tomographic plane) and
                        vertical (y = mid, through the rotation axis) slices,
                        plus a |probe| row when the smoothness is being swept
  nrmse_ndist<N>_n<n>_ntheta<nt>[_amp<a>][_prbs<s>].png   masked NRMSE against the
                                       phantom, against
                        whichever of the two parameters is being swept

Serial, no MPI — run it on one rank after run_study.sh.
"""

import argparse
import glob
import hashlib
import os
import re
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import (read_summary, rec_dirs,        # noqa: E402
                    last_checkpoint, read_slices)                                                # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',  default='/data3/vnikitin/disp_study')
    p.add_argument('--ndist', type=int, default=1)
    p.add_argument('--amps',  default='0 1 2 4 8 16 32',
                   help='space- or comma-separated displacement half-widths [px]')
    p.add_argument('--prb-smooths', default=None,
                   help='space- or comma-separated probe blur sigmas [px] '
                        f'(default: only the standard probe, {C.PRB_SMOOTH:.4g} px)')
    p.add_argument('--obj-smooth', type=float, default=C.OBJ_SMOOTH,
                   help='object blur sigma [voxels] of the cases to plot; not swept, '
                        f'one value per figure (default {C.OBJ_SMOOTH:.4g} voxel)')
    p.add_argument('--n',      type=int, default=None,
                   help='detector size to plot (default: whichever covers the most cases)')
    p.add_argument('--ntheta', type=int, default=None,
                   help='number of angles to plot (default: as for --n)')
    p.add_argument('--out',   default=os.path.dirname(os.path.abspath(__file__)),
                   help='directory the PNGs are written to')
    p.add_argument('--tag',   default=None,
                   help='extra suffix for the PNG names, so two sweeps in one root '
                        'do not overwrite each other')
    p.add_argument('--cmap',  default='gray')
    p.add_argument('--probe-cmap', default='gray',
                   help='colour map for the two probe rows')
    p.add_argument('--clip',  type=float, default=99.8,
                   help='percentile of |ground truth| used for the shared colour range')
    p.add_argument('--vmin',  type=float, default=None,
                   help='fixed lower end of the object colour range (overrides --clip)')
    p.add_argument('--vmax',  type=float, default=None,
                   help='fixed upper end of the object colour range (overrides --clip)')
    p.add_argument('--slices', default='h', choices=['h', 'hv'],
                   help="object rows to draw: 'h' = the horizontal (z = mid) slice "
                        "only, 'hv' = horizontal and vertical")
    p.add_argument('--crop',  type=float, default=0.0,
                   help='fraction trimmed off each side of the object slices, '
                        'e.g. 0.15 keeps the middle 70%%')
    p.add_argument('--probe-row', default='on', choices=['auto', 'on', 'off'],
                   help="show the |probe| and arg(probe) rows "
                        "('auto' = only when the smoothness varies)")
    p.add_argument('--dpi', type=int, default=300,
                   help='output resolution. The panels are ~2.6 in wide and hold a '
                        'slice of ~nobj px, so 140 dpi (the old value) resolved barely '
                        'a third of it; 300 is close to 1:1 at n=1024 and 400 is 1:1 at '
                        'n=2048. Raise it further only if you intend to zoom in -- the '
                        'file grows as dpi^2.')
    return p.parse_args()


def numbers(txt):
    return [float(t) for t in re.split(r'[,\s]+', txt.strip()) if t]


def gt_id(data_h5):
    """Short fingerprint of the ground-truth phantom stored in a dataset.

    The phantom specification in common.py can be edited between runs, and each
    data.h5 keeps its own copy of whatever was current when it was written, so
    a sweep assembled over several sessions can silently contain two different
    ground truths — and then its NRMSE curve is not one curve.  A strided sample
    of obj_re is enough to tell them apart.
    """
    try:
        with h5py.File(data_h5, 'r') as f:
            d = f['obj_re']
            sub = np.ascontiguousarray(d[::37, ::37, ::37])
            return f'{d.shape[0]}:' + hashlib.md5(sub.tobytes()).hexdigest()[:6]
    except (OSError, KeyError):
        return None


def read_probe(data_h5):
    """(|prb|, arg prb) of the first distance of a dataset, or None."""
    try:
        with h5py.File(data_h5, 'r') as f:
            return f['prb_abs'][0], f['prb_phase'][0]
    except (OSError, KeyError):
        return None


a = parse()
out_dir = a.out
amps    = numbers(a.amps)
smooths = numbers(a.prb_smooths) if a.prb_smooths else [C.PRB_SMOOTH]
combos  = [(amp, s) for amp in amps for s in smooths]
objs = abs(a.obj_smooth - C.OBJ_SMOOTH) > 1e-6   # a non-default phantom sharpness
vary_amp = len(amps) > 1
vary_smooth = len(smooths) > 1
vary_prb = len(smooths) > 1

# --- which (n, ntheta) group to plot ----------------------------------------
groups = {}                                   # (n, ntheta) -> {(amp, s): rec_dir}
for amp, s in combos:
    case_dir = os.path.join(a.root, C.case_name(amp, a.ndist, s, a.obj_smooth))
    for key, d in rec_dirs(case_dir).items():
        if last_checkpoint(d) is not None:
            groups.setdefault(key, {})[(amp, s)] = d

if not groups:
    raise SystemExit('nothing to plot')

for (gn, gnt), got in sorted(groups.items()):
    print(f'found n={gn or "?"} ntheta={gnt or "?"}: {len(got)} case(s)')

wanted = [k for k in groups
          if (a.n is None or k[0] == a.n) and (a.ntheta is None or k[1] == a.ntheta)]
if not wanted:
    raise SystemExit(f'no reconstruction with n={a.n} ntheta={a.ntheta}')
n, ntheta = max(wanted, key=lambda k: len(groups[k]))
sel = groups[(n, ntheta)]
tag  = f'_n{n}_ntheta{ntheta}' if n else '_legacy'
# a sweep puts one amp per column, so an amp in the name would be a lie; a
# single-amp run (the brain test) needs it, or runs at different amp overwrite
# each other's figure
if not vary_amp:
    tag += f'_amp{amps[0]:g}'
if not vary_smooth:
    tag += f'_prbs{smooths[0]:g}'
if objs:
    tag += f'_objs{a.obj_smooth:g}'
if a.tag:
    tag += f'_{a.tag}'
nlbl = f'n={n}, ntheta={ntheta}' if n else 'n, ntheta unrecorded'
print(f'plotting {nlbl} ({len(sel)} cases)')


def label(amp, s, gen=None):
    parts = []
    if vary_amp or not vary_prb:
        parts.append(f'amp = ±{amp:g} px')
    if vary_prb:
        parts.append(f'probe $\\sigma$ = {s:g} px')
    if gen:
        parts[-1] += f'  [{gen}]'
    return '\n'.join(parts)


# --- collect the slices ------------------------------------------------------
cases, gt, gt_from = [], None, None
for amp, s in combos:
    rdir = sel.get((amp, s))
    data_h5 = os.path.join(a.root, C.case_name(amp, a.ndist, s, a.obj_smooth), 'data.h5')
    # summary.txt is written once rec.py is through its last iteration, so its
    # absence is what tells a finished case from one still on the GPU
    done = rdir is not None and os.path.isfile(os.path.join(rdir, 'summary.txt'))
    if not done:
        # the column stays, empty: a gap in the sweep should be visible, and a
        # half-converged checkpoint next to converged ones would just mislead
        if rdir is None and not os.path.isfile(data_h5):
            print(f'amp={amp:g} sigma={s:g}: nothing on disk, skipped')
            continue
        print(f'amp={amp:g} sigma={s:g}: '
              f'{"still running" if rdir else "not reconstructed"} — blank column')
        cases.append(dict(amp=amp, smooth=s, sl=None, s={}, cp=None,
                          prb=read_probe(data_h5), gt=gt_id(data_h5),
                          stale=False, pending=True))
        continue
    cp = last_checkpoint(rdir)
    sl = read_slices(cp)
    if gt is None:
        # same phantom in every case, so the first data.h5 of this group serves
        # as the reference — as long as it was not regenerated at another size
        g = read_slices(data_h5)
        if g[0].shape != sl[0].shape:
            print(f'amp={amp:g} sigma={s:g}: data.h5 is {g[0].shape}, checkpoint '
                  f'{sl[0].shape} — dataset regenerated at another size, '
                  f'ground truth skipped')
        else:
            gt = g
            gt_from = (amp, s)
    cases.append(dict(amp=amp, smooth=s, sl=sl, s=read_summary(rdir),
                      cp=cp, prb=read_probe(data_h5), gt=gt_id(data_h5),
                      pending=False, stale=(os.path.isfile(data_h5) and
                             os.path.getmtime(data_h5) >
                             os.path.getmtime(os.path.join(rdir, 'summary.txt'))
                             if os.path.isfile(os.path.join(rdir, 'summary.txt'))
                             else False)))
    print(f'amp={amp:g} sigma={s:g}: {cp}')

if not cases:
    raise SystemExit('nothing to plot')
if all(c['pending'] for c in cases):
    raise SystemExit('every reconstruction is still running — nothing to plot yet')

# --- ground-truth provenance -------------------------------------------------
# Cases whose data.h5 holds different phantoms are not on one curve, and cases
# whose data.h5 is newer than their own summary.txt were scored against a
# ground truth that has since been overwritten.  Both are marked rather than
# quietly averaged away.
gens = sorted({c['gt'] for c in cases if c['gt'] and not c['stale']})
gen_of = {g: chr(ord('A') + i) for i, g in enumerate(gens)}
for c in cases:
    # a stale case was scored against a phantom that no longer exists on disk,
    # so no letter can honestly be attached to it
    c['gen'] = '?' if c['stale'] else gen_of.get(c['gt'], '?')
stale = [c for c in cases if c['stale']]
if len(gens) > 1:
    print('\n!! WARNING: these cases were scored against DIFFERENT phantoms — '
          'their NRMSEs are not comparable:')
    for g in gens:
        members = [f"amp{c['amp']:g}/sigma{c['smooth']:g}" for c in cases if c['gt'] == g]
        print(f"   [{gen_of[g]}] {g}: {' '.join(members)}")
    print()
if stale:
    print('!! WARNING: data.h5 is newer than summary.txt for '
          + ', '.join(f"amp{c['amp']:g}/sigma{c['smooth']:g}" for c in stale)
          + ' — the phantom they were scored against has since been regenerated;'
            ' the slices shown are still theirs, the "ground truth" column may not be.\n')

# Shared colour range, from the ground truth (or the reconstructions, if the
# matching ground truth is gone) so every panel is comparable.
ref = gt if gt is not None else cases[0]['sl']
lim = np.percentile(np.abs(np.concatenate([ref[0].ravel(), ref[1].ravel()])), a.clip)
vmin = -lim if a.vmin is None else a.vmin
vmax = lim if a.vmax is None else a.vmax


def crop(img, frac=a.crop):
    """Trim `frac` off each side — the object never reaches the field edge."""
    if frac <= 0:
        return img
    cy, cx = (int(round(frac * n)) for n in img.shape[:2])
    return img[cy:img.shape[0] - cy, cx:img.shape[1] - cx]

show_prb = ((a.probe_row == 'on' or (a.probe_row == 'auto' and vary_prb))
            and any(c['prb'] is not None for c in cases))

# one colour range for every probe panel, so the columns really are comparable
_pa = [c['prb'][0] for c in cases if c['prb'] is not None]
_pp = [c['prb'][1] for c in cases if c['prb'] is not None]
pabs_lo, pabs_hi = (min(x.min() for x in _pa), max(x.max() for x in _pa)) if _pa else (0, 1)
pph_lo,  pph_hi  = (min(x.min() for x in _pp), max(x.max() for x in _pp)) if _pp else (0, 1)

multi_gen = len({c['gen'] for c in cases}) > 1
gt_gen = next((c['gen'] for c in cases
               if (c['amp'], c['smooth']) == gt_from), '?') if gt_from else '?'
gt_lbl = 'ground truth' + (f'  [{gt_gen}]' if multi_gen else '')
panels = ([] if gt is None else [(gt_lbl, gt, None, None)]) + \
         [(label(c['amp'], c['smooth'], c['gen'] if multi_gen else None),
           c['sl'], c['s'], c['prb']) for c in cases]

nobj_rows = 2 if a.slices == 'hv' else 1
nrow = nobj_rows + (2 if show_prb else 0)   # probe adds |prb| and arg(prb)
ncol = len(panels)
# square panels, so the rows sit right on top of each other with no dead band
fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow + 0.9),
                       gridspec_kw=dict(wspace=0.04, hspace=0.12))
ax = np.asarray(ax).reshape(nrow, ncol)

for j, (title, sl, s, prb) in enumerate(panels):
    for i in range(nobj_rows):
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if sl is None:                       # still running: leave the space empty
            for spine in ax[i, j].spines.values():
                spine.set_visible(False)
            if i == 0:
                ax[i, j].text(0.5, 0.5, 'reconstruction\nnot finished',
                              transform=ax[i, j].transAxes, ha='center',
                              va='center', fontsize=9, color='0.55')
            continue
        im = ax[i, j].imshow(crop(sl[i]), cmap=a.cmap, vmin=vmin, vmax=vmax)
    sub = '' if not s else f"\nNRMSE = {float(s.get('nrmse_obj', 'nan')):.3f}"
    ax[0, j].set_title(title + sub, fontsize=10)
    if show_prb:
        for i in (nobj_rows, nobj_rows + 1):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ra, rp = nobj_rows, nobj_rows + 1
        if prb is None:
            ax[ra, j].axis('off'); ax[rp, j].axis('off')
        else:
            pa, pp = prb
            imp = ax[ra, j].imshow(pa, cmap=a.probe_cmap, vmin=pabs_lo, vmax=pabs_hi)
            imq = ax[rp, j].imshow(pp, cmap=a.probe_cmap, vmin=pph_lo, vmax=pph_hi)
            # inside the panel, so the rows keep the same height as the object ones
            ax[ra, j].text(0.5, 0.02, f'std/mean = {pa.std()/max(pa.mean(), 1e-30):.3f}',
                           transform=ax[ra, j].transAxes, ha='center', va='bottom',
                           fontsize=8, color='w',
                           bbox=dict(fc='k', alpha=0.45, ec='none', pad=1.5))
            ax[rp, j].text(0.5, 0.02, f'p-p = {np.ptp(pp):.2f} rad',
                           transform=ax[rp, j].transAxes, ha='center', va='bottom',
                           fontsize=8, color='w',
                           bbox=dict(fc='k', alpha=0.45, ec='none', pad=1.5))

ax[0, 0].set_ylabel('horizontal\n(z = mid)', fontsize=10)
if nobj_rows > 1:
    ax[1, 0].set_ylabel('vertical\n(y = mid)', fontsize=10)
if show_prb:
    ax[nobj_rows, 0].set_ylabel('|probe|',    fontsize=10)
    ax[nobj_rows + 1, 0].set_ylabel('arg(probe)', fontsize=10)
# one colourbar per block of rows, each attached to the rows it describes
fig.colorbar(im, ax=ax[:nobj_rows].ravel().tolist(), fraction=0.012, pad=0.008,
             label=r'Re(obj) = $-\delta$')
if show_prb:
    fig.colorbar(imp, ax=ax[nobj_rows].tolist(), fraction=0.012, pad=0.008,
                 label='|probe|')
    fig.colorbar(imq, ax=ax[nobj_rows + 1].tolist(), fraction=0.012, pad=0.008,
                 label='arg(probe) [rad]')

meta = next(c['s'] for c in cases if not c['pending'])
swept = ('displacement amplitude and probe smoothness' if vary_amp and vary_prb else
         'probe smoothness' if vary_prb else 'random displacement amplitude')
objs_txt = f'object $\\sigma$={a.obj_smooth:g} voxel, ' if objs else ''
note = ('' if not multi_gen else
        '\n[A]/[B]… mark different ground-truth phantoms, [?] one that has since been '
        'overwritten — NRMSEs across them are NOT comparable')
fig.suptitle(f"single-distance reconstruction vs {swept} "
             f"(ndist={a.ndist}, {nlbl}, {objs_txt}"
             f"{meta.get('niter','?')} BH iterations){note}", fontsize=12)
png = os.path.join(out_dir, f'slices_ndist{a.ndist}{tag}.png')
fig.savefig(png, dpi=a.dpi, bbox_inches='tight')
print(f'-> {png}')

# --- NRMSE vs the swept parameter --------------------------------------------
# x is whichever parameter varies (probe smoothness wins when both do, with one
# line per amplitude); with a single series the three NRMSE components are shown.
have = [c for c in cases if 'nrmse_obj' in c['s']]
if have:
    xkey, skey = ('smooth', 'amp') if vary_prb else ('amp', 'smooth')
    xlab = ('probe smoothing $\\sigma$ [px]' if vary_prb
            else 'displacement half-width [px]')
    series = sorted({c[skey] for c in have})
    # points scored against different phantoms are drawn as separate, unjoined
    # segments — connecting them would draw a trend that does not exist
    blocks = sorted({c['gen'] for c in have})
    dash = ['-', '--', ':', '-.']
    fig2, ax2 = plt.subplots(figsize=(6.8, 4.6))
    for i, v in enumerate(series):
        for b, gen in enumerate(blocks):
            # colour separates whichever of the two actually varies
            col = f'C{b}' if len(series) == 1 else f'C{i}'
            pts = sorted([c for c in have if c[skey] == v and c['gen'] == gen],
                         key=lambda c: c[xkey])
            if not pts:
                continue
            xs = [c[xkey] for c in pts]
            if len(series) == 1 and len(blocks) == 1:
                for k, m in (('obj', 'o-'), ('delta', 's--')):
                    ax2.plot(xs, [float(c['s'][f'nrmse_{k}']) for c in pts], m,
                             label=f'NRMSE({k})')
                # beta comes out ~50x worse than delta; on a shared axis it
                # flattens the two curves the study is actually about
                if all('nrmse_beta' in c['s'] for c in pts):
                    axb = ax2.twinx()
                    axb.plot(xs, [float(c['s']['nrmse_beta']) for c in pts],
                             '^:', color='C2')
                    _bv = [float(c['s']['nrmse_beta']) for c in pts]
                    axb.set_yscale('log' if max(_bv) / min(_bv) >= 3 else 'linear')
                    axb.set_ylabel('NRMSE(beta)', color='C2')
                    axb.tick_params(axis='y', colors='C2')
                    for _ax in (axb.yaxis.set_major_formatter,
                                axb.yaxis.set_minor_formatter):
                        _ax(matplotlib.ticker.FuncFormatter(
                            lambda y, _: f'{y:g}'))
                    ax2.plot([], [], '^:', color='C2',
                             label='NRMSE(beta)  [right axis]')
            else:
                lbl = (f'amp = ±{v:g} px' if skey == 'amp'
                       else f'probe $\\sigma$ = {v:g} px')
                if len(blocks) > 1:
                    lbl += ('  [phantom overwritten]' if gen == '?'
                            else f'  [phantom {gen}]')
                ax2.plot(xs, [float(c['s']['nrmse_obj']) for c in pts],
                         marker='o', ls=dash[b % len(dash)], color=col, label=lbl)
    ax2.set_xlabel(xlab)
    ax2.set_ylabel('masked NRMSE vs phantom')
    # log only earns its keep when the curves span a real range; over a factor
    # of two or three it just throws away the tick labels
    _yv = [float(c['s'][k]) for c in have
           for k in ('nrmse_obj', 'nrmse_delta') if k in c['s']]
    ax2.set_yscale('log' if _yv and max(_yv) / min(_yv) >= 3 else 'linear')
    # sweeps that span decades (and usually include 0) are unreadable on a
    # linear x — symlog keeps the interesting small end open
    xall = sorted({c[xkey] for c in have})
    pos = [x for x in xall if x > 0]
    sym = dict(linthresh=min(pos), linscale=0.6) if pos and max(pos)/min(pos) > 20 else None
    if sym:
        ax2.set_xscale('symlog', **sym)
        ax2.set_xticks(xall)
        ax2.set_xticklabels([f'{x:g}' for x in xall], fontsize=8)
        ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # probe contrast is a pure function of sigma, so it fits on a second axis
    # (twiny, not secondary_xaxis — the latter does not inherit a symlog scale)
    ctr = {}
    if xkey == 'smooth':
        for c in have:
            if c['prb'] is not None:
                ctr[c['smooth']] = float(C.probe_contrast(c['prb'][0][None])[0])
    if len(ctr) == len(xall):
        axt = ax2.twiny()
        if sym:
            axt.set_xscale('symlog', **sym)
        axt.set_xlim(ax2.get_xlim())
        axt.set_xticks(xall)
        axt.set_xticklabels([f'{ctr[x]:.3f}' for x in xall], fontsize=7, rotation=90)
        axt.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axt.set_xlabel('probe contrast  std|prb| / mean|prb|', fontsize=9)
    fixed = ('' if vary_amp and vary_prb else
             f", probe $\\sigma$={smooths[0]:g} px" if not vary_prb else
             f", amp=±{amps[0]:g} px")
    fig2.suptitle(f'ndist={a.ndist}, {nlbl}{fixed}'
                  + (f", object $\\sigma$={a.obj_smooth:g} voxel" if objs else '')
                  + ('\nsegments scored against different phantoms are not joined'
                     if len(blocks) > 1 else ''), fontsize=11)
    ax2.grid(True, which='both', alpha=0.3)
    # ScalarFormatter rounds the sub-decade minor ticks to "0", and labelling
    # every minor tick of a multi-decade axis is unreadable -- so print plain
    # numbers, and only on the 2/3/5 minors
    def _minor_lbl(y, _):
        m = y / 10.0 ** np.floor(np.log10(y)) if y > 0 else 0
        return f'{y:g}' if round(m) in (2, 3, 5) else ''
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f'{y:g}'))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(_minor_lbl))
    ax2.tick_params(axis='y', which='minor', labelsize=7)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    png2 = os.path.join(out_dir, f'nrmse_ndist{a.ndist}{tag}.png')
    fig2.savefig(png2, dpi=a.dpi)
    print(f'-> {png2}')
