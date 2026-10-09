#!/usr/bin/env python
"""Convergence of the data misfit for each rho[prb] arm of the WT_cell1 sweep.

    python plot_conv.py [<dir holding nfp_prb*/>] [-o out.png]

Reads conv_nfp.csv (iter,err,time), written by RecNFP.error_debug on the
error_step grid -- the same numbers as the `iter=N: ...sec err=...` log lines,
but already parsed.

Two ladders are drawn, identical except for the intensity normalisation in
step0:

  nfp_prb<p>_flux   step0_flux.py, every frame scaled to mean 1   SOLID
  nfp_prb<p>        step0.py, whole stack / one global mean       dashed, faint

err is rec_nfp_mpi.min(), the data misfit F0 ALONE: no rho, no regularisation.
So it IS comparable between arms -- rho changes the trajectory, not the
functional.  It is NOT comparable with model=intensity runs (~4x larger).
Across the two ladders d itself differs, but only by a <2% per-frame gain, and
the cold-start err agrees to 0.01%, so the gap between the families is real.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = [1, 2, 4, 8, 16, 32, 64]   # 128 excluded: err@0 already saturated at 64
NITER = 1024             # a curve ending earlier is unfinished or was cancelled
ZOOM_FROM = 128          # iter at which the arms have separated and plateaued

ROOT = '/home/beams/VNIKITIN/eagle/vnikitin/20260929/PROCESSED_DATA'
OUT  = 'conv_rho_prb.png'
args = sys.argv[1:]
if '-o' in args:
    i = args.index('-o'); OUT = args[i + 1]; del args[i:i + 2]
if args:
    ROOT = args[0]


def load(subdir):
    """iter, err from conv_nfp.csv, dropping the iter=-1 cold-start row."""
    f = os.path.join(ROOT, subdir, 'scan0001', 'conv_nfp.csv')
    if not os.path.exists(f):
        return None
    a = np.genfromtxt(f, delimiter=',', skip_header=1)
    if a.ndim < 2 or len(a) < 2:
        return None
    a = a[a[:, 0] >= 0]
    return a[:, 0], a[:, 1]


fig, (ax, az) = plt.subplots(1, 2, figsize=(13.0, 5.2))
cmap = plt.get_cmap('viridis')

# (suffix, label, linestyle, linewidth, alpha, drop unfinished curves).  The
# global-mean ladder is finished, so a short curve there is a cancelled job and
# only clutters; the flux ladder is live, so partial arms are worth seeing.
FAMILIES = [('_flux', 'flux',       '-',  1.8, 1.00, False),
            ('',      'global mean', '--', 1.0, 0.45, True)]

rows = []
for suffix, fam, ls, lw, al, drop_partial in FAMILIES:
    for k, p in enumerate(ARMS):
        c = load(f'nfp_prb{p}{suffix}')
        if c is None:
            continue
        it, err = c
        partial = it[-1] < NITER
        if partial and drop_partial:
            continue
        col = cmap(k / (len(ARMS) - 1))
        tag = f' (to {int(it[-1])})' if partial else ''
        ax.semilogy(it, err, color=col, ls=ls, lw=lw, alpha=al,
                    label=f'{p}, {fam}{tag}')
        m = it >= ZOOM_FROM
        az.plot(it[m], err[m] * 1e5, color=col, ls=ls, lw=lw, alpha=al)
        rows.append((p, fam, err[0], err[-1], int(it[-1])))

# symlog, not log: iteration 0 is where the arms differ most and log has no
# room for it.  Linear below the first error_step, log above.
ax.set_xscale('symlog', linthresh=16, linscale=0.5)
ax.set_xlabel('iteration'); ax.set_ylabel('err  =  data misfit $F_0$')
ax.set_title('WT_cell1 NFP: convergence vs rho[prb]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=7, loc='upper right', ncol=2, title='rho[prb], step0',
          title_fontsize=7)

az.set_xscale('log')
az.set_xlabel('iteration'); az.set_ylabel(r'err  $\times 10^{5}$')
az.set_title(f'tail, iter $\\geq$ {ZOOM_FROM} (log x, linear y)')
az.grid(alpha=0.3, which='both')
az.axvline(NITER, color='0.8', lw=0.8)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f'wrote {OUT}')

print(f'\n{"rho[prb]":>8} {"step0":>12} {"err@0":>11} {"err@end":>11} {"last":>6}')
for p, fam, e0, e1, n in sorted(rows, key=lambda r: (r[1], r[0])):
    print(f'{p:>8} {fam:>12} {e0:>11.4e} {e1:>11.4e} {n:>6}')
