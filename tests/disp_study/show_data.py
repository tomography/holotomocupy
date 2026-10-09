#!/usr/bin/env python
"""What one angle of a scan actually is: the illumination and the frames.

Every figure of this study shows a reconstruction; this one shows the input.
For each distance of a dataset it draws the probe that illuminates the sample
-- amplitude and phase, as `gen_data.py` wrote it into data.h5 -- and, under
them, the intensity recorded at one angle.  The distances differ only in how
far the sample sits from the focus, so the columns are the same wavefield seen
at four propagation lengths, and the fringes coarsen from left to right.

The probe's own speckle is far stronger than the sample's contrast, so a fourth
row divides the frame by the flat field (`/ref` of the same file) -- that is the
row the sample is actually visible in.  `--flat off` drops it and leaves the
figure at exactly the probe and the raw frame.

    python show_data.py --root /data3/vnikitin/dose_study_phantom \\
        --ndist 4 --ntheta 900 --amp 1 --prb-smooth 1 --tag phantom

One figure lands in --out:

  data_ndist<N>_n<n>_ntheta<nt>_amp<a>[_prbs<s>][_<tag>].png

Serial, no MPI, and it reads data.h5 only -- no reconstruction needed, so it
runs as soon as gen_data.py is done.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', default='/data3/vnikitin/dose_study_phantom')
    p.add_argument('--ndist',  type=int, default=4,
                   help='distances of the dataset to open')
    p.add_argument('--ntheta', type=int, default=900,
                   help='angles of the dataset to open (part of its directory name)')
    p.add_argument('--amp',        type=float, default=1.0)
    p.add_argument('--prb-smooth', type=float, default=1.0)
    p.add_argument('--obj-smooth', type=float, default=C.OBJ_SMOOTH)
    p.add_argument('--theta', type=int, default=0,
                   help='index of the angle whose frames are drawn')
    p.add_argument('--flat', default='on', choices=['on', 'off'],
                   help='add the flat-field-corrected row, frame / ref')
    p.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument('--tag', default=None)
    p.add_argument('--cmap', default='gray')
    p.add_argument('--clip', type=float, default=99.8,
                   help='percentile bounding each row\'s shared colour range')
    p.add_argument('--dpi',  type=int, default=300)
    return p.parse_args()


a = parse()
case = os.path.join(a.root,
                    C.dose_case_name(a.ndist, a.ntheta, a.amp, a.prb_smooth, a.obj_smooth))
# only shown when it is not the phantom as it has always been
objs = abs(a.obj_smooth - C.OBJ_SMOOTH) > 1e-6
h5 = os.path.join(case, 'data.h5')
if not os.path.isfile(h5):
    raise SystemExit(f'{h5}: no such dataset -- run gen_data.py first')

with h5py.File(h5, 'r') as f:
    meta   = dict(f.attrs)
    ndist  = int(meta['ndist'])
    ntheta = int(meta['ntheta'])
    n      = int(meta['n'])
    if not -ntheta <= a.theta < ntheta:
        raise SystemExit(f'--theta {a.theta}: the scan has {ntheta} angles')
    it     = a.theta % ntheta
    z1     = f['z1'][:]
    theta  = float(f['theta'][it])
    pos    = f['pos'][it]                       # [ndist, 2] true displacement, px
    prb    = f['prb_abs'][:] * np.exp(1j * f['prb_phase'][:])
    # /data is already the intensity the detector counted; /ref is still an
    # amplitude (the probe-fit term compares |D prb| against it), so square only
    # that one -- the flat-field correction below is a ratio of intensities
    frames = f['data'][:, it].astype('float32')
    flat   = f['ref'][:].astype('float32') ** 2 if a.flat == 'on' else None

contrast = C.probe_contrast(prb)
w = C.dose_weights(z1)
print(f'{h5}  n={n}  ndist={ndist}  ntheta={ntheta}')
print(f'angle {it} of {ntheta}: theta = {np.rad2deg(theta):.3f} deg'
      if abs(theta) <= 2 * np.pi else f'angle {it} of {ntheta}: theta = {theta:.3f}')
for j in range(ndist):
    print(f'  z1 = {z1[j] * 1e3:6.3f} mm  dose x{w[j]:.3f}  '
          f'std|prb|/mean|prb| = {contrast[j]:.4f}  '
          f'shift = ({pos[j, 0]:+.2f}, {pos[j, 1]:+.2f}) px  '
          f'I in [{frames[j].min():.3f}, {frames[j].max():.3f}]')

rows = [(r'$|{\rm probe}|$',          np.abs(prb)),
        (r'$\arg({\rm probe})$ [rad]', np.angle(prb)),
        (f'intensity, angle {it}',     frames)]
if flat is not None:
    # the flat field is noiseless here, so the ratio needs no regularisation
    # beyond keeping a division by zero out of the figure
    rows.append(('intensity / flat field', frames / np.maximum(flat, 1e-6)))

fig, ax = plt.subplots(len(rows), ndist, squeeze=False,
                       figsize=(2.9 * ndist + 0.9, 2.9 * len(rows) + 0.9),
                       gridspec_kw=dict(hspace=0.06, wspace=0.04))

for r, (label, vol) in enumerate(rows):
    # one range per row: the columns are the same quantity at four distances and
    # are only worth showing side by side if they are on the same scale
    lo, hi = np.percentile(vol, [100.0 - a.clip, a.clip])
    for j in range(ndist):
        axj = ax[r, j]
        axj.set_xticks([]); axj.set_yticks([])
        im = axj.imshow(vol[j], cmap=a.cmap, vmin=lo, vmax=hi)
        if r == 0:
            axj.set_title(f'$z_1$ = {z1[j] * 1e3:.2f} mm,  dose $\\times${w[j]:.3f}',
                          fontsize=10)
            axj.text(0.03, 0.03, f'std/mean = {contrast[j]:.3f}',
                     transform=axj.transAxes, fontsize=7, color='yellow')
        if r == 2:
            axj.text(0.03, 0.03,
                     f'shift = ({pos[j, 0]:+.2f}, {pos[j, 1]:+.2f}) px',
                     transform=axj.transAxes, fontsize=7, color='yellow')
    ax[r, 0].set_ylabel(label, fontsize=10)
    fig.colorbar(im, ax=ax[r, :].tolist(), fraction=0.02, pad=0.01)

fig.suptitle(
    f'{os.path.basename(case)}:  n = {n} px,  amp = ±{a.amp:g} px,  '
    f'probe $\\sigma$ = {a.prb_smooth:g} px'
    + (f', object $\\sigma$ = {a.obj_smooth:g} voxel' if objs else '')
    + (f',  {int(meta["photons"]):g} photons' if meta.get('photons') else ',  noiseless'),
    fontsize=12)

tag = (f'_n{n}_ntheta{ntheta}_amp{a.amp:g}'
       + (f'_prbs{a.prb_smooth:g}' if a.prb_smooth else '')
       + (f'_objs{a.obj_smooth:g}' if objs else '')
       + (f'_{a.tag}' if a.tag else ''))
png = os.path.join(a.out, f'data_ndist{ndist}{tag}.png')
fig.savefig(png, dpi=a.dpi, bbox_inches='tight')
print(f'-> {png}')
