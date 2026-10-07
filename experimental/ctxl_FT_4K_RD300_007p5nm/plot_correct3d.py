#!/usr/bin/env python
"""Plot correct_correct3D.txt -- horizontal and vertical -- in one panel.

    python plot_correct3d.py [-o out.png] [--v] [--bin 2] [--voxel 7.5]

correct_correct3D.txt is the only shift input nabu itself applies (the +-300 px
random walk and the drift are already baked into Peter's <pfile>_rec_ EDFs by
correct_motion.txt).  One row per projection plus a trailing duplicate row
(ntheta+1 = 12001), two columns, in 2x2-BINNED detector pixels.

COLUMN ORDER.  Column 0 is HORIZONTAL, column 1 is VERTICAL.  Three independent
checks agree, on all four 20260829 scans:
  - correct_correct3D_v.txt -- the vertical-only variant -- zeroes column 0;
  - column 1 has a mean of ~0.00 while column 0 carries a positive offset
    (ctxl_HT 0.821 binned = the +1.64 raw px the rotation-axis work measured
    as the HORIZONTAL term);
  - steps15.py:670 reads it as `[:ntheta, ::-1]`, and holotomocupy shifts are
    (y, x), so the file must be (x, y).
../AtomiumS1/plot_correct3d.py's docstring states the opposite and is wrong;
its panels are mislabelled.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DROP = ('/home/beams/VNIKITIN/eagle/vnikitin/20260829/ctxl/'
        'ctxl_FT_4K_RD300_007p5nm_0003_')

p = argparse.ArgumentParser()
p.add_argument('-o', '--out', default=os.path.join(HERE, 'correct3d.png'))
p.add_argument('-f', '--file', default=None, help='override the .txt path')
p.add_argument('--v', action='store_true', help='plot the _v (vertical-only) variant')
p.add_argument('--bin', type=int, default=2, help='correct3d_bin (config: 2)')
p.add_argument('--voxel', type=float, default=7.5, help='unbinned voxel size, nm')
p.add_argument('--ntheta', type=int, default=12000)
a = p.parse_args()

path = a.file or (DROP + ('/correct_correct3D_v.txt' if a.v else '/correct_correct3D.txt'))
d = np.loadtxt(path)
nrow = len(d)
d = d[:a.ntheta]                      # drop the trailing duplicate row
h, v = d[:, 0], d[:, 1]               # horizontal, vertical -- see the docstring
j = np.arange(len(d))

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.plot(j, h, lw=0.9, color='#c1121f', label=f'horizontal  (col 0)   mean {h.mean():+.3f}  ptp {np.ptp(h):.2f}')
ax.plot(j, v, lw=0.9, color='#0353a4', label=f'vertical    (col 1)   mean {v.mean():+.3f}  ptp {np.ptp(v):.2f}')
ax.axhline(0, color='k', lw=0.6, alpha=0.4)

ax.set_xlabel(f'projection index   (0 .. {a.ntheta - 1}, omega 0 .. 180 deg)')
ax.set_ylabel(f'shift  [{a.bin}x{a.bin}-binned detector px, as stored]')
ax.set_xlim(0, len(d))
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc='best', fontsize=9, framealpha=0.92)
ax.set_title(f'{os.path.basename(path)}   --   {nrow} rows '
             f'({a.ntheta} plotted + {nrow - a.ntheta} trailing duplicate)', fontsize=10)

# right axis: raw (unbinned) detector px, which is what our configs carry
r = ax.secondary_yaxis('right', functions=(lambda y: y * a.bin, lambda y: y / a.bin))
r.set_ylabel(f'raw detector px  (x{a.bin})')

lo, hi = ax.get_ylim()
ax.text(0.995, 0.02,
        f'{(hi - lo) * a.bin * a.voxel:.0f} nm full scale   '
        f'({a.voxel} nm voxels)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, alpha=0.7)

fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(f'{path}\n  {nrow} rows, plotted {len(d)}')
print(f'  horizontal (col 0): mean {h.mean():+8.4f}  min {h.min():+8.4f}  max {h.max():+8.4f}  ptp {np.ptp(h):7.4f}  binned px')
print(f'  vertical   (col 1): mean {v.mean():+8.4f}  min {v.min():+8.4f}  max {v.max():+8.4f}  ptp {np.ptp(v):7.4f}  binned px')
print(f'  -> raw px: horizontal ptp {np.ptp(h) * a.bin:.3f}, vertical ptp {np.ptp(v) * a.bin:.3f}'
      f'   ({np.ptp(h) * a.bin * a.voxel:.0f} / {np.ptp(v) * a.bin * a.voxel:.0f} nm)')
print(f'wrote {a.out}')
