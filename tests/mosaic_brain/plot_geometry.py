#!/usr/bin/env python
"""Schematic of the synthetic mosaic scan: beamline, tile layout, sample.

Everything is drawn from config_gen.conf and tile_offsets.txt, so the figure
follows whatever those say.

Panels (b) and (c) are in the MOSAIC frame -- where each tile lands in the
composed object -- which is the negative of the shift stored in tile_offsets.txt.

Panel (c) is a top view of the ROTATING sample, so a tile column is not a strip
there: over the 180 deg scan the column at lateral offset x sweeps the annulus
|x| - fov/2 <= r <= |x| + fov/2 around the rotation axis.  Columns at the same
|x| sweep the same annulus, and the union of all of them is the reconstructed
disk.

    python plot_geometry.py config_gen.conf [-o mosaic_layout.png]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, Ellipse, Wedge,
                                FancyArrowPatch)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)
from holotomocupy.config import parse_args_gen   # noqa: E402
from mosaic_geometry import read_tile_offsets    # noqa: E402

# sample: cylinder, diameter x height in mm
SAMPLE_D = 1.0
SAMPLE_H = 1.0

DCOL = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('-o', '--out', default='mosaic_layout.png')
    opt = ap.parse_args()
    args = parse_args_gen(opt.config)

    z1    = np.array(args.z1)
    fdd   = args.focustodetectordistance
    mag   = fdd / z1
    nmag  = mag / mag[0]
    vox   = args.detector_pixelsize / mag[0]          # m, finest grid
    ndist = len(z1)

    # tile_offsets.txt holds sample shifts; the tile lands at minus that.
    names, off = read_tile_offsets(args.tile_file)
    off = off * vox * 1e3                                               # mm
    # In the composed object, x index grows with -h and y index grows with -v,
    # so on screen (y up) the tile sits at x = -h, y = +v.
    ty, tx = off[:, 0], -off[:, 1]
    fov      = args.ndet / nmag * vox * 1e3           # mm, per distance
    det_half = 0.5 * args.ndet * args.detector_pixelsize * 1e3          # mm

    y_lo, y_hi = ty.min() - fov[0] / 2, ty.max() + fov[0] / 2
    hcov = np.ptp(tx) + fov[0]
    vcov = y_hi - y_lo

    fig = plt.figure(figsize=(14.5, 9.6))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5],
                           width_ratios=[1.35, 1.0],
                           hspace=0.28, wspace=0.20,
                           left=0.06, right=0.975, top=0.885, bottom=0.105)

    # ---------------------------------------------------------------- (a) beam
    ax = fig.add_subplot(gs[0, :])
    ax.plot([0, fdd * 100], [0, det_half], color='0.75', lw=1)
    ax.plot([0, fdd * 100], [0, -det_half], color='0.75', lw=1)
    ax.fill_between([0, fdd * 100], [0, -det_half], [0, det_half],
                    color='#fff3cc', zorder=0)
    ax.plot(0, 0, marker='*', ms=17, color='#b8860b', zorder=5)
    ax.text(0, -0.9, 'focus\n(KB)', fontsize=9, ha='center', va='top',
            color='#7a5c00')

    for k in range(ndist):
        h = det_half * z1[k] / fdd
        ax.plot([z1[k] * 100] * 2, [-h, h], color=DCOL[k], lw=3, zorder=4,
                solid_capstyle='butt')
    ax.add_patch(Rectangle((fdd * 100, -det_half), 2.5, 2 * det_half,
                           fc='0.35', ec='k'))
    ax.text(fdd * 100 - 3, 0,
            f'detector\n{args.ndet}$^2$ px, {args.detector_pixelsize*1e6:.2f} $\\mu$m',
            ha='right', va='center', fontsize=9)
    ax.annotate('', xy=(fdd * 100, -det_half - 1.0), xytext=(0, -det_half - 1.0),
                arrowprops=dict(arrowstyle='<->', color='0.45'))
    ax.text(fdd * 50, -det_half - 1.35, f'focus–detector  {fdd*100:.1f} cm',
            ha='center', va='top', fontsize=9, color='0.45')
    ax.set_xlim(-7, fdd * 100 + 13)
    ax.set_ylim(-det_half - 2.6, det_half + 2.4)
    ax.set_xlabel('distance from focus [cm]')
    ax.set_ylabel('[mm]')
    ax.set_title(f'(a)  cone-beam holotomography, {ndist} sample–detector '
                 f'distances    E = {args.energy} keV', loc='left', fontsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # zoom inset on the sample region
    axi = ax.inset_axes([0.055, 0.63, 0.235, 0.35])
    zmax = z1[-1] * 100 * 1.14
    axi.plot([0, zmax], [0, det_half * zmax / (fdd * 100)], color='0.7', lw=.8)
    axi.plot([0, zmax], [0, -det_half * zmax / (fdd * 100)], color='0.7', lw=.8)
    for k in range(ndist):
        h = det_half * z1[k] / fdd
        axi.plot([z1[k] * 100] * 2, [-h, h], color=DCOL[k], lw=3,
                 solid_capstyle='butt')
        up = (k % 2 == 0)
        axi.text(z1[k] * 100, (h + 0.022) if up else (-h - 0.022),
                 f'$z_{{1,{k}}}$\n{fov[k]*1e3:.0f}', ha='center',
                 va='bottom' if up else 'top',
                 fontsize=7, color=DCOL[k], linespacing=0.95)
    axi.set_xlim(3.5, 7.9)
    axi.set_ylim(-0.34, 0.30)
    axi.set_title('sample planes — FOV [$\\mu$m]', fontsize=7.5, pad=2)
    axi.tick_params(labelsize=6.5)
    axi.set_yticks([-0.1, 0, 0.1])
    axi.set_facecolor('#fffdf5')
    ax.indicate_inset_zoom(axi, edgecolor='0.55')

    # ------------------------------------------------------ (b) mosaic, front
    ax = fig.add_subplot(gs[1, 0])
    ax.add_patch(Rectangle((-SAMPLE_D / 2, -SAMPLE_H / 2), SAMPLE_D, SAMPLE_H,
                           fc='#e9e9f4', ec='#5a5a8a', lw=1.6, zorder=1))
    ax.add_patch(Ellipse((0, SAMPLE_H / 2), SAMPLE_D, 0.13 * SAMPLE_D,
                         fc='#d6d6ea', ec='#5a5a8a', lw=1.6, zorder=2))
    ax.add_patch(Ellipse((0, -SAMPLE_H / 2), SAMPLE_D, 0.13 * SAMPLE_D,
                         fc='#e9e9f4', ec='#5a5a8a', lw=1.6, ls=':', zorder=1))
    ax.add_patch(Rectangle((-SAMPLE_D / 2, y_lo), SAMPLE_D, y_hi - y_lo,
                           fc='#ffdf85', alpha=.45, ec='none', zorder=1.5))

    for k in (ndist - 1, 0):                       # widest first, then dist 0
        for (x, y) in zip(tx, ty):
            ax.add_patch(Rectangle((x - fov[k] / 2, y - fov[k] / 2),
                                   fov[k], fov[k], fill=False,
                                   ec=DCOL[k], lw=1.1 if k else 1.7,
                                   ls=(0, (4, 2)) if k else '-', zorder=3))
    for x, y, nm in zip(tx, ty, names):
        ax.text(x, y, nm, ha='center', va='center', fontsize=11.5,
                fontweight='bold', color='#111', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='w', ec='none', alpha=.82))

    ax.plot([0, 0], [-SAMPLE_H / 2 - .14, SAMPLE_H / 2 + .14], 'k-.', lw=1, zorder=4)
    ax.text(-0.022, -SAMPLE_H / 2 - .17, 'rotation axis', fontsize=8.5,
            rotation=90, va='bottom', ha='right', color='0.25')
    ax.annotate('', xy=(-SAMPLE_D / 2, -SAMPLE_H / 2 - .15),
                xytext=(SAMPLE_D / 2, -SAMPLE_H / 2 - .15),
                arrowprops=dict(arrowstyle='<->', color='#5a5a8a'))
    ax.text(0.26, -SAMPLE_H / 2 - .175, f'{SAMPLE_D:g} mm', ha='center', va='top',
            fontsize=9, color='#5a5a8a')
    ax.annotate('', xy=(SAMPLE_D / 2 + .13, -SAMPLE_H / 2),
                xytext=(SAMPLE_D / 2 + .13, SAMPLE_H / 2),
                arrowprops=dict(arrowstyle='<->', color='#5a5a8a'))
    ax.text(SAMPLE_D / 2 + .155, 0, f'{SAMPLE_H:g} mm', fontsize=9, rotation=90,
            va='center', color='#5a5a8a')
    ax.annotate('', xy=(-SAMPLE_D / 2 - .13, y_lo), xytext=(-SAMPLE_D / 2 - .13, y_hi),
                arrowprops=dict(arrowstyle='<->', color='#c07800'))
    ax.text(-SAMPLE_D / 2 - .155, 0, f'scanned band\n{vcov:.3f} mm',
            fontsize=8.5, rotation=90, va='center', ha='right', color='#c07800')

    ax.set_xlim(-0.82, 0.82)
    ax.set_ylim(-0.86, 0.78)
    ax.set_aspect('equal')
    ax.set_xlabel('horizontal [mm]')
    ax.set_ylabel('vertical [mm]')
    ax.set_title(f'(b)  mosaic on the sample — {args.ntile_v} x {args.ntile_h} '
                 f'= {len(tx)} tiles, named "row_col"', loc='left', fontsize=11)

    # ------------------------------------------------------- (c) top view
    # The sample turns, so a column does not stay a strip: the column at
    # lateral offset x covers every sample point whose radius from the
    # rotation axis lies in [|x| - fov/2, |x| + fov/2] at some angle.  Columns
    # with the same |x| therefore sweep the SAME annulus, and the union of the
    # annuli is the disk that can be reconstructed.
    ax = fig.add_subplot(gs[1, 1])
    ax.add_patch(Circle((0, 0), SAMPLE_D / 2, fc='#e9e9f4', ec='#5a5a8a', lw=1.6,
                        zorder=1))

    xs   = np.sort(np.unique(np.round(tx, 9)))
    grp  = np.sort(np.unique(np.round(np.abs(xs), 9)))       # distinct |x|
    fill = ['#f7f4ea', '#dfdac6']       # neutral: colour means DISTANCE here
    rings = []
    for gi, xa in enumerate(grp):
        cols  = [str(c) for c, x in enumerate(xs)
                 if abs(abs(round(x, 9)) - xa) < 1e-9]
        r_out = xa + fov[0] / 2
        r_in  = max(0.0, xa - fov[0] / 2)
        rings.append((cols, r_in, r_out, xa + fov[-1] / 2))
        ax.add_patch(Wedge((0, 0), r_out, 0, 360, width=r_out - r_in,
                           fc=fill[gi % len(fill)], ec='none', alpha=.55,
                           zorder=2))
        for k in (ndist - 1, 0):                  # widest first, then dist 0
            ax.add_patch(Circle((0, 0), xa + fov[k] / 2, fill=False,
                                ec=DCOL[k], lw=1.5 if k == 0 else 1.0,
                                ls='-' if k == 0 else (0, (4, 2)),
                                alpha=.85, zorder=3))
        # label the ring on the upper-left diagonal, one step round per group;
        # the innermost ring is a small disk, so its label goes outside on a leader
        aa  = np.deg2rad(168 - 25 * gi)
        lab = ('cols ' if len(cols) > 1 else 'col ') + ','.join(cols)
        box = dict(boxstyle='round,pad=0.13', fc='w', ec='none', alpha=.85)
        if r_in:
            rm = 0.5 * (r_in + r_out)
            ax.text(rm * np.cos(aa), rm * np.sin(aa), lab, ha='center',
                    va='center', fontsize=8.5, fontweight='bold', color='#333',
                    zorder=6, bbox=box)
        else:
            rt = r_out + 0.30
            ax.annotate(lab, xy=(0.5 * r_out * np.cos(aa), 0.5 * r_out * np.sin(aa)),
                        xytext=(rt * np.cos(aa), rt * np.sin(aa)),
                        ha='center', va='center', fontsize=8.5, fontweight='bold',
                        color='#333', zorder=6, bbox=box,
                        arrowprops=dict(arrowstyle='-', color='0.45', lw=.8,
                                        shrinkA=1, shrinkB=1))

    # where the columns actually are at theta = 0, to tie (c) back to (b)
    for x in xs:
        ax.add_patch(Rectangle((x - fov[0] / 2, -SAMPLE_D / 2 - .06),
                               fov[0], SAMPLE_D + .12, fill=False,
                               ec='0.6', lw=.8, ls=(0, (1, 2)), zorder=2.5))
    for c, x in enumerate(xs):
        ax.text(x, SAMPLE_D / 2 + .10, f'{c}', ha='center', fontsize=10.5,
                fontweight='bold', color='0.45')
    ax.text(-0.92, SAMPLE_D / 2 + .10, 'column\n@ $\\theta$=0', ha='center',
            fontsize=8.5, color='0.45', linespacing=0.95)

    ax.annotate('', xy=(-SAMPLE_D / 2 - .16, 0), xytext=(-SAMPLE_D - .18, 0),
                arrowprops=dict(arrowstyle='-|>', color='#b8860b', lw=2))
    ax.text(-SAMPLE_D - .20, 0, 'beam', ha='right', va='center', fontsize=9.5,
            color='#7a5c00')
    ax.add_patch(FancyArrowPatch((0.68, -0.24), (0.24, -0.68),
                                 connectionstyle='arc3,rad=0.30',
                                 arrowstyle='-|>', mutation_scale=13,
                                 color='#444', lw=1.3))
    ax.text(0, -0.76, f'{args.ntheta} angles / {args.theta_range:g}$^\\circ$'
                      f' — each column sweeps a ring',
            ha='center', fontsize=9, color='#444')
    ax.add_patch(Circle((0, 0), SAMPLE_D / 2, fill=False, ec='#5a5a8a', lw=1.6,
                        zorder=5))                       # sample edge, on top
    ax.annotate(f'{SAMPLE_D:g} mm sample',
                xy=(-SAMPLE_D / 2 * 0.707, -SAMPLE_D / 2 * 0.707),
                xytext=(-1.06, -0.60), fontsize=8.5, color='#5a5a8a',
                ha='center', va='center', zorder=6,
                arrowprops=dict(arrowstyle='-', color='#5a5a8a', lw=.8,
                                shrinkA=1, shrinkB=1))
    ax.plot(0, 0, 'k+', ms=9, zorder=6)
    ax.set_xlim(-1.42, 0.80)
    ax.set_ylim(-0.86, 0.78)
    ax.set_aspect('equal')
    ax.set_xlabel('horizontal [mm]')
    ax.set_ylabel('beam direction [mm]')
    ax.set_title('(c)  top view — swept by the rotation', loc='left', fontsize=11)

    fig.legend(handles=[
        plt.Line2D([], [], color=DCOL[0], lw=2,
                   label=f'FOV of $z_{{1,0}}$: {fov[0]*1e3:.0f} $\\mu$m '
                         f'— the region every distance sees'),
        plt.Line2D([], [], color=DCOL[-1], lw=1.4, ls=(0, (4, 2)),
                   label=f'FOV of $z_{{1,{ndist-1}}}$: {fov[-1]*1e3:.0f} $\\mu$m '
                         f'— widest, lowest magnification'),
        plt.Rectangle((0, 0), 1, 1, fc='#ffdf85', alpha=.6, ec='none',
                      label=f'scanned band: middle {vcov:.3f} mm of the '
                            f'{SAMPLE_H:g} mm cylinder')],
        loc='lower center', ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f'Synthetic mosaic holotomography scan — {args.ntile_v}×{args.ntile_h}'
        f' = {len(tx)} tiles × {ndist} distances = {len(tx)*ndist} frames per angle,'
        f' {args.ntheta} angles,  {vox*1e9:.0f} nm voxel\n'
        f'covers {hcov:.3f} mm (h) × {vcov:.3f} mm (v) of a '
        f'{SAMPLE_D:g} mm × {SAMPLE_H:g} mm cylinder — full width, middle band',
        fontsize=12.5, y=0.985)

    fig.savefig(opt.out, dpi=140)
    print(f'wrote {opt.out}')
    print(f'  horizontal coverage {hcov:.4f} mm  (sample {SAMPLE_D} mm)')
    print(f'  vertical   coverage {vcov:.4f} mm  (sample {SAMPLE_H} mm)')
    for k in range(ndist):
        print(f'  dist {k}: norm mag {nmag[k]:.4f}   FOV {fov[k]*1e3:7.1f} um'
              f'   overlap {(fov[k]-args.tile_step_h*vox*1e3)*1e3:6.1f} um')
    print('  rings swept by the rotation (dist 0 FOV, widest FOV outer radius):')
    for cols, r_in, r_out, r_wide in rings:
        print(f'    col {",".join(cols):<4s}  r {r_in*1e3:6.1f} - {r_out*1e3:6.1f} um'
              f'   (widest dist out to {r_wide*1e3:6.1f} um)')


if __name__ == '__main__':
    main()
