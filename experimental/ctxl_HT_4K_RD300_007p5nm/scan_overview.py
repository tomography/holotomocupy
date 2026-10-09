#!/usr/bin/env python
"""
One-page summary of what this acquisition actually is.

    python scan_overview.py config_steps15.conf [--fig scan_overview.png]

Reads the raw scan -- the .info sidecars, the NXtomo (or 2025 bliss HDF5)
metadata, the random-displacement tables and a handful of EDF frames -- and
draws everything worth knowing before starting a reconstruction:

  * the derived geometry per distance (energy, z1/z2, magnification, voxel size,
    propagation distance, field of view, angular sampling, exposure, data
    volume), next to the values this folder's configs were filled in with, so a
    mismatch is visible;
  * one hologram per distance at the same angle, which is the picture of what
    a multi-distance HT scan buys: the same object at four fringe regimes;
  * three holograms across the 180 deg range at the first distance;
  * the flat field, one frame with the scan-mean illumination subtracted (the
    only view in which the sample is actually visible), and the end-of-scan /
    start-of-scan flat ratio, which is the honest picture of how much the
    illumination moved during the scan -- this is what limits any registration
    on this data;
  * the random displacement pattern, in both the x-y plane and against frame
    index, which is the whole point of an "RD300" scan, together with the
    object-plane amplitude that sets nobj;
  * the slow stage drift, correct_motion.txt minus the commanded displacement,
    which is what step 3 actually adds on top of the random shifts;
  * mean transmission against frame index, sampled across the scan: a flat line
    means the beam and the sample held up, a trend means drift or damage.

Deliberately standalone -- numpy / h5py / fabio / matplotlib, no cupy and no MPI
-- so it runs on a login node before steps15 has ever been started.
"""

import argparse
import configparser
import os

import fabio
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from esrf_layout import Layout


def binned(img, k):
    """Block-mean down by k. A 4096 px frame drawn into ~300 screen px aliases
    into pure speckle otherwise, hiding the sample completely."""
    ny, nx = img.shape[0] // k * k, img.shape[1] // k * k
    return img[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def imshow(ax, img, title, cmap='gray', pct=(1, 99), k=4):
    img = binned(img, k) if k > 1 else img
    lo, hi = np.percentile(img, pct)
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])


def rd(f):
    return fabio.open(f).data.astype('float32')


class Plane:
    """Dark/flat-corrected access to one distance, built once and reused."""

    def __init__(self, lay, k, nflat, ntheta):
        self.lay, self.k, self.ntheta = lay, k, ntheta
        self.dark = np.mean([rd(f) for f in lay.darks(k, nflat)], axis=0)
        self.ref0 = np.mean([rd(f) for f in lay.refs(k, 0, nflat)], axis=0) - self.dark
        r1 = lay.refs(k, ntheta, nflat)
        self.ref1 = (np.mean([rd(f) for f in r1], axis=0) - self.dark) if r1 else None

    def trans(self, j):
        return (rd(self.lay.proj(self.k, j)) - self.dark) / (self.ref0 + 1e-3)

    def holo(self, j):
        return -np.log(np.clip(self.trans(j), 1e-3, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--path', help='override path= (e.g. a local mount of eagle)')
    ap.add_argument('--pfile', help='override pfile=')
    ap.add_argument('--nflat', type=int, default=8, help='flats/darks averaged')
    ap.add_argument('--probe', type=int, default=40,
                    help='frames sampled for the transmission trace')
    ap.add_argument('--sample', default='',
                    help='free-text sample description for the header block')
    ap.add_argument('--proposal', default='',
                    help='free-text proposal / beamtime label')
    ap.add_argument('--fig', default='scan_overview.png')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(args.config, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']

    lay = Layout((args.path or cfg.get('path')).rstrip('/'),
                 args.pfile or cfg.get('pfile'))
    geo = lay.geometry()
    ndist, ntheta = lay.ndist, lay.ntheta
    info = lay.info(0)
    pfile = lay.pfile

    energy = geo['energy']
    det_px = geo['detector_pixelsize']
    f2d = geo['focustodetectordistance']
    z1, z2 = geo['z1'], geo['z2']
    mag, nmag = geo['magnifications'], geo['norm_magnifications']
    voxel = geo['voxelsizes']
    d_eff = z1 * z2 / f2d                      # true propagation distance
    wavelen = 12.398419e-10 / energy           # m

    scan_range = float(info.get('ScanRange', -180))
    expo, latency = lay.exposure(0)
    n0, n1 = fabio.open(lay.refs(0, 0)[0]).data.shape
    nedf = lay.nproj_files(0)
    fov = n0 * voxel[0]
    # Fresnel number of one detector pixel at the effective (parallel-beam)
    # propagation distance -- how many object pixels a point spreads over, i.e.
    # how holographic rather than absorption-like these frames are.
    fringe = np.sqrt(wavelen * d_eff) / voxel

    # --- displacements ------------------------------------------------------
    shifts = np.array([np.loadtxt(lay.shift_source(k), dtype='float32')
                       for k in range(ndist)])                # (ndist, rows, 2)
    sh = shifts[:, :ntheta]
    # Commanded in each plane's own detector pixels; in the object frame they
    # are 1/norm_mag times larger, and that is what nobj has to hold.
    obj_amp = np.abs(sh / nmag[:, None, None]).max()

    ref_dist = cfg.getint('ref_dist', fallback=0)
    mpath = f'{lay.dname(ref_dist)}/correct_motion.txt'
    drift = None
    if os.path.exists(mpath):
        raw = np.loadtxt(mpath, dtype='float32')
        m = min(len(raw), len(shifts[ref_dist]))
        drift = raw[:m] - shifts[ref_dist][:m]

    # --- images -------------------------------------------------------------
    planes = [Plane(lay, k, args.nflat, ntheta) for k in range(ndist)]
    p0 = planes[0]
    shown = [0, ntheta // 2, ntheta - 1]
    holos = [p0.holo(j) for j in shown]
    dist_holos = [p.holo(0) for p in planes]

    probe_j = np.unique(np.linspace(0, ntheta - 1, args.probe).astype(int))
    trans = np.empty(len(probe_j))
    # Running mean of -log frames over the whole scan.  The sample is at a
    # different one of the +-300 px random positions in every frame, so it
    # smears away; what survives is the detector-fixed illumination left over
    # after the flat division.  Subtracting it is what makes the sample visible
    # here, and it is the same step estimate_center.py needs before any
    # correlation on this data means anything.
    illum = np.zeros_like(p0.ref0, dtype='float64')
    for i, j in enumerate(probe_j):
        t = p0.trans(int(j))
        trans[i] = t.mean()
        illum += -np.log(np.clip(t, 1e-3, None))
    illum /= len(probe_j)

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(19.5, 11.0))
    gs = GridSpec(3, 5, figure=fig, width_ratios=[1.22, 1, 1, 1, 1],
                  hspace=0.30, wspace=0.20,
                  left=0.010, right=0.990, top=0.928, bottom=0.055)

    raw_gb = nedf * ndist * n0 * n1 * 2 / 1024**3
    scan_h = (ntheta * ndist * (expo + (latency or 0)) / 3600.0) if expo else None
    aux = sorted({f for k in range(ndist)
                  for f in ('correct.txt', 'correct_motion.txt', 'quali.mat',
                            'shrink_list.mat', 'angles_file.txt')
                  if os.path.exists(f'{lay.dname(k)}/{f}')}
                 | {f for f in ('rhapp.mat', 'correct_correct3D.txt',
                                'reference_motion.mat')
                    if os.path.exists(f'{lay.path}/{pfile}_/{f}')})

    def col(v, fmt='{:.3f}'):
        return '  '.join(fmt.format(x) for x in v)

    txt = f"""$\\bf{{{pfile.replace('_', chr(92) + '_')}}}$

  sample          {args.sample or '(see --sample)'}
  proposal        {args.proposal or '(see --proposal)'}
  acquired        {info.get('Date', '?')}
  layout          {lay.flavour}

$\\bf{{beam}}$
  energy          {energy:.2f} keV   ($\\lambda$ = {wavelen * 1e10:.4f} $\\AA$)
  detector pixel  {det_px * 1e6:.4f} um   (optic {info.get('Optic_used', '?')})
  focus->detector {f2d * 1e3:.4f} mm

$\\bf{{geometry,\\ per\\ distance}}$
  z1 [mm]      {col(z1 * 1e3, '{:.4f}')}
  z2 [mm]      {col(z2 * 1e3, '{:.2f}')}
  mag [x]      {col(mag, '{:.1f}')}
  norm mag     {col(nmag, '{:.4f}')}
  voxel [nm]   {col(voxel * 1e9, '{:.3f}')}
  prop d [mm]  {col(d_eff * 1e3, '{:.3f}')}
  fringe [px]  {col(fringe, '{:.0f}')}
  $\\bf{{voxel\\ size}}$    $\\bf{{{voxel[0] * 1e9:.3f}\\ nm}}$  (plane 1)
  field of view   {fov * 1e6:.2f} um  ({n0} px)

$\\bf{{sampling}}$
  distances       {ndist}
  projections     {ntheta} over {abs(scan_range):.0f} deg
  angular step    {abs(scan_range) / ntheta:.4f} deg
  Nyquist needs   {int(np.ceil(np.pi * n0 / 2))} for {n0} px
  exposure        {f'{expo:.3f} s' if expo else '?'}"""
    if scan_h:
        txt += f'\n  scan duration   {scan_h:.2f} h  (all distances)'
    txt += f"""
  flats / darks   {lay.nref} / {lay.ndark} per distance
  EDF frames      {nedf} x {ndist}   ({raw_gb:.0f} GiB raw)

$\\bf{{random\\ displacement}}$
  x   {sh[..., 0].min():+7.1f} .. {sh[..., 0].max():+7.1f} px (detector)
  y   {sh[..., 1].min():+7.1f} .. {sh[..., 1].max():+7.1f} px (detector)
  max |r| in the object frame  {obj_amp:.0f} px
  = +-{obj_amp * voxel[0] * 1e6:.2f} um of sample motion
  needs nobj >= {int(np.ceil((n0 / nmag[-1] + 2 * obj_amp) / 64)) * 64}"""
    if drift is not None:
        txt += f"""

$\\bf{{drift\\ (correct\\_motion - random)}}$
  x   {drift[:, 0].min():+7.3f} .. {drift[:, 0].max():+7.3f} px
  y   {drift[:, 1].min():+7.3f} .. {drift[:, 1].max():+7.3f} px
  at plane {ref_dist + 1}, the reference plane"""
    txt += f"""

$\\bf{{aux\\ files\\ present}}$
  {', '.join(aux) if aux else 'none'}

$\\bf{{this\\ folder's\\ configs}}$
  n = {cfg.get('n', '?')}   nobj = {cfg.get('nobj', '?')}
  ref_dist = {cfg.get('ref_dist', '?')}
  rotation_center_shift = {cfg.get('rotation_center_shift', '?')} px
  paganin (delta/beta)  = {cfg.get('paganin', '?')}
"""
    axt = fig.add_subplot(gs[:, 0]); axt.axis('off')
    axt.text(0.0, 1.0, txt, va='top', ha='left', fontsize=7.6,
             family='monospace', linespacing=1.30)

    # row 0: the same frame at every distance
    for k in range(min(ndist, 4)):
        imshow(fig.add_subplot(gs[0, k + 1]), dist_holos[k],
               f'distance {k + 1}   frame 0\n'
               f'{voxel[k] * 1e9:.2f} nm, prop {d_eff[k] * 1e3:.2f} mm, '
               f'fringe {fringe[k]:.0f} px')

    # row 1: angular coverage at distance 1, plus what the flats look like
    for i, (j, h) in enumerate(zip(shown, holos)):
        imshow(fig.add_subplot(gs[1, i + 1]), h,
               f'dist 1  frame {j}  ($\\theta$ = {abs(scan_range) * j / ntheta:.1f}$\\degree$)\n'
               f'$-\\log(I/I_0)$, sample at '
               f'({sh[0, j, 1]:+.0f}, {sh[0, j, 0]:+.0f}) px')
    imshow(fig.add_subplot(gs[1, 4]), holos[0] - illum,
           f'frame 0 minus the mean of {len(probe_j)} frames\n'
           f'= the sample, residual illumination removed')

    # row 2: displacement, drift, transmission, flat stability
    ax = fig.add_subplot(gs[2, 1])
    s = ax.scatter(sh[0, :, 0], sh[0, :, 1], c=np.arange(ntheta), s=2,
                   cmap='viridis', linewidths=0)
    ax.set_xlabel('x displacement [px]'); ax.set_ylabel('y displacement [px]')
    ax.set_title('random displacement, distance 1', fontsize=8.5)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
    cb = fig.colorbar(s, ax=ax, orientation='horizontal', fraction=.05, pad=.19)
    cb.ax.tick_params(labelsize=7); cb.set_label('frame', size=8)

    ax = fig.add_subplot(gs[2, 2])
    w = min(150, ntheta)
    ax.plot(sh[0, :w, 0], '.-', lw=.6, ms=2.5, label='x')
    ax.plot(sh[0, :w, 1], '.-', lw=.6, ms=2.5, label='y')
    ax.set_xlabel('frame'); ax.set_ylabel('displacement [px]')
    ax.set_title(f'displacement vs frame (first {w})\n'
                 f'consecutive frames jump the full range', fontsize=8.5)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, 3])
    if drift is not None:
        ax.plot(drift[:, 0], lw=1.1, label=f'x, ptp {np.ptp(drift[:, 0]):.2f} px')
        ax.plot(drift[:, 1], lw=1.1, label=f'y, ptp {np.ptp(drift[:, 1]):.2f} px')
        ax.set_xlabel('frame'); ax.set_ylabel('drift [detector px]')
        ax.set_title(f'correct_motion.txt - random shift\n'
                     f'(plane {ref_dist + 1}; what step 3 adds)', fontsize=8.5)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    else:
        ax.axis('off')
        ax.text(.5, .5, 'no correct_motion.txt', ha='center', va='center')

    ax = fig.add_subplot(gs[2, 4])
    ax.plot(probe_j, trans, 'o-', ms=3, lw=.8)
    ax.set_xlabel('frame'); ax.set_ylabel(r'mean $I/I_0$')
    ax.set_title(f'mean transmission, dist 1 ({len(probe_j)} frames)\n'
                 f'{trans.mean():.4f} $\\pm$ {trans.std():.4f}'
                 + (f'   |   flat end/start {np.mean(p0.ref1 / (p0.ref0 + 1e-3)):.3f}'
                    f' $\\pm$ {np.std(p0.ref1 / (p0.ref0 + 1e-3)):.3f}'
                    if p0.ref1 is not None else ''),
                 fontsize=8.5)
    ax.grid(alpha=.3)

    fig.suptitle(f'{pfile}   --   ESRF ID16A, {info.get("Date", "")}   --   '
                 f'{voxel[0] * 1e9:.2f} nm voxels, {ntheta} projections over '
                 f'{abs(scan_range):.0f}$\\degree$, {ndist} distances',
                 fontsize=12)
    fig.savefig(args.fig, dpi=110)
    print(f'figure -> {args.fig}')

    # --- same numbers on stdout, for the terminal ---------------------------
    print(f'\n{pfile}   {lay.flavour}   ndist={ndist} ntheta={ntheta} '
          f'n={n0}x{n1} nref={lay.nref} ndark={lay.ndark}')
    print(f'energy {energy} keV   det px {det_px * 1e6:.4f} um   '
          f'focus->detector {f2d * 1e3:.4f} mm')
    print(f'{"k":>2} {"z1 [mm]":>10} {"z2 [mm]":>10} {"mag":>9} {"norm mag":>9} '
          f'{"voxel [nm]":>11} {"prop [mm]":>10} {"fringe px":>10}')
    for k in range(ndist):
        print(f'{k + 1:>2} {z1[k] * 1e3:10.4f} {z2[k] * 1e3:10.3f} {mag[k]:9.3f} '
              f'{nmag[k]:9.5f} {voxel[k] * 1e9:11.4f} {d_eff[k] * 1e3:10.4f} '
              f'{fringe[k]:10.1f}')
    for m in lay.info_check(geo):
        print(f'WARNING geometry vs .info: {m}')
    print(f'\nrandom displacement, detector px per plane:')
    for k in range(ndist):
        print(f'  plane {k + 1}: x {sh[k, :, 0].min():+8.2f} .. {sh[k, :, 0].max():+8.2f}'
              f'   y {sh[k, :, 1].min():+8.2f} .. {sh[k, :, 1].max():+8.2f}'
              f'   -> object frame |r| <= {np.abs(sh[k] / nmag[k]).max():7.1f} px')
    print(f'  object-frame footprint  {n0 / nmag[-1]:.0f} px + 2 x {obj_amp:.0f} px'
          f'  -> nobj >= {int(np.ceil((n0 / nmag[-1] + 2 * obj_amp) / 64)) * 64}')
    if drift is not None:
        print(f'\ndrift (correct_motion.txt - random, plane {ref_dist + 1}), '
              f'{len(drift)} rows:')
        for i, nm_ in ((0, 'x'), (1, 'y')):
            d = drift[:, i]
            print(f'  {nm_}: {d.min():+8.4f} .. {d.max():+8.4f}   '
                  f'ptp {np.ptp(d):.4f}   rms {d.std():.4f} px')
        # Two different questions, and only the second one is interesting.
        # The second difference between CONSECUTIVE rows is tiny for any
        # smooth curve sampled 4000 times, so it only shows that ESRF fitted
        # something analytic rather than following the retakes frame by frame.
        # Whether that analytic thing is a straight line is a separate test:
        # fit a ramp over the whole scan and look at what is left.
        m = len(drift) - 3
        curv = np.abs(np.diff(drift[:m], 2, axis=0)).max()
        print(f'  max |2nd difference| between consecutive rows = {curv:.2e} px'
              f'  ({"analytic, not per-frame" if curv < 1e-4 else "per-frame noise"})')
        t = np.arange(m)
        V = np.column_stack([np.ones(m), t])
        for i, nm_ in ((0, 'x'), (1, 'y')):
            res = drift[:m, i] - V @ np.linalg.lstsq(V, drift[:m, i], rcond=None)[0]
            print(f'  {nm_}: residual after removing a straight ramp = '
                  f'{np.ptp(res):.4f} px ptp, {res.std():.4f} rms  '
                  f'({"a ramp" if np.ptp(res) < 0.05 else "curved, higher order than linear"})')
    print(f'\naux files: {", ".join(aux) if aux else "none"}')


if __name__ == '__main__':
    main()
