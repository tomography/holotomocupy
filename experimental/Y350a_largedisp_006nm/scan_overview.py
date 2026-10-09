#!/usr/bin/env python
"""
One-page summary of what this acquisition actually is.

    python scan_overview.py config_steps15.conf [--fig scan_overview.png]

Reads the raw scan -- the .info file, the ICAT HDF5 metadata, correct.txt and a
handful of EDF frames -- and draws everything worth knowing before starting a
reconstruction:

  * the derived geometry (energy, distances, magnification, voxel size, field of
    view, angular sampling, exposure, dose proxy, data volume), next to the
    values this folder's configs were filled in with, so a mismatch is visible;
  * three holograms across the 180 deg range, showing what the sample looks like
    and how strong the Fresnel fringes are at this propagation distance;
  * the flat field, one frame with the scan-mean illumination subtracted (the
    only view in which the sample is actually visible), and the end-of-scan /
    start-of-scan flat ratio,
    which is the honest picture of how much the illumination moved during the
    27 minutes of the scan -- this is what limits any registration on this data;
  * the +-300 px random displacement pattern from correct.txt, in both the x-y
    plane and against frame index, which is the whole point of an "FT large
    random displacement" scan;
  * mean transmission against frame index, sampled across the scan: a flat line
    means the beam and the sample held up, a trend means drift or radiation
    damage.

Deliberately standalone -- numpy / h5py / fabio / matplotlib, no cupy and no MPI
-- so it runs on a login node before steps15 has ever been started.
"""

import argparse
import configparser
import glob
import os

import fabio
import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from show_geometry import (read_energy, read_sx0, read_sx,
                           read_detector_pixelsize, read_focustodetectordistance)


def read_info(path):
    """Parse the ESRF <prefix>.info sidecar into a dict of strings."""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip()
    return out


def binned(img, k):
    """Block-mean down by k. A 4096 px frame drawn into ~300 screen px aliases
    into pure speckle otherwise, hiding the sample completely."""
    ny, nx = img.shape[0] // k * k, img.shape[1] // k * k
    return img[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def imshow(ax, img, title, cmap='gray', pct=(1, 99), k=4):
    img = binned(img, k) if k > 1 else img
    lo, hi = np.percentile(img, pct)
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--path', help='override path= (e.g. a local mount of eagle)')
    ap.add_argument('--pfile', help='override pfile=')
    ap.add_argument('--nflat', type=int, default=8, help='flats/darks averaged')
    ap.add_argument('--probe', type=int, default=40,
                    help='frames sampled for the transmission trace')
    ap.add_argument('--fig', default='scan_overview.png')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(args.config, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']
    path = (args.path or cfg.get('path')).rstrip('/')
    pfile = args.pfile or cfg.get('pfile')
    dname = f'{path}/{pfile}_1_'

    # --- geometry, exactly as steps15 derives it ---------------------------
    h5file = sorted(glob.glob(f'{dname}/*.h5'))[0]
    energy = read_energy(h5file)                       # keV
    det_px = read_detector_pixelsize(h5file)           # m
    z1_2 = read_focustodetectordistance(h5file)        # m
    sx0 = read_sx0(h5file)
    z1 = read_sx(h5file) - sx0
    z2 = z1_2 - z1
    mag = z1_2 / z1
    voxel = abs(det_px / mag)
    wavelen = 12.398419e-10 / energy                   # m
    ndist = len(sorted(glob.glob(f'{path}/{pfile}_[0-9]_/')))

    info = read_info(f'{dname}/{pfile}_1_.info')
    ntheta = int(info['TOMO_N'])
    scan_range = float(info['ScanRange'])
    expo = float(info['Count_time'])
    latency = float(info['Latency_time'])

    n0, n1 = fabio.open(f'{dname}/ref0000_0000.edf').data.shape
    nref = len(glob.glob(f'{dname}/ref[0-9]*_0000.edf'))
    ndark = len(glob.glob(f'{dname}/darkend[0-9]*.edf'))
    nedf = len(glob.glob(f'{dname}/{pfile}_1_[0-9]*.edf'))
    fov = n0 * voxel

    # Fresnel number of one detector pixel at the effective (parallel-beam)
    # propagation distance -- how many pixels a point spreads over, i.e. how
    # holographic rather than absorption-like these frames are.
    d_eff = z1 * z2 / z1_2
    fringe = np.sqrt(wavelen * d_eff) / voxel

    # --- images ------------------------------------------------------------
    def rd(f):
        return fabio.open(f).data.astype('float32')

    dark = np.mean([rd(f) for f in sorted(glob.glob(f'{dname}/darkend*.edf'))[:args.nflat]], axis=0)
    ref0 = np.mean([rd(f) for f in sorted(glob.glob(f'{dname}/ref*_0000.edf'))[:args.nflat]], axis=0) - dark
    ref1_f = sorted(glob.glob(f'{dname}/ref*_{ntheta:04d}.edf'))[:args.nflat]
    ref1 = (np.mean([rd(f) for f in ref1_f], axis=0) - dark) if ref1_f else None

    def holo(j):
        img = (rd(f'{dname}/{pfile}_1_{j:04d}.edf') - dark) / (ref0 + 1e-3)
        return -np.log(np.clip(img, 1e-3, None))

    shown = [0, ntheta // 2, ntheta - 1]
    holos = [holo(j) for j in shown]

    # --- displacements and transmission trace ------------------------------
    shifts = np.loadtxt(f'{dname}/correct.txt', dtype='float32')[:ntheta]
    probe_j = np.unique(np.linspace(0, ntheta - 1, args.probe).astype(int))
    trans = np.empty(len(probe_j))
    # Running mean of -log frames over the whole scan.  The sample is at a
    # different one of the +-300 px random positions in every frame, so it
    # smears away; what survives is the detector-fixed illumination left over
    # after the flat division.  Subtracting it is what makes the sample visible
    # here, and it is the same step estimate_center.py needs before any
    # correlation on this data means anything.
    illum = np.zeros_like(ref0, dtype='float64')
    for i, j in enumerate(probe_j):
        img = (rd(f'{dname}/{pfile}_1_{j:04d}.edf') - dark) / (ref0 + 1e-3)
        trans[i] = img.mean()
        illum += -np.log(np.clip(img, 1e-3, None))
    illum /= len(probe_j)

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(17.5, 10.5))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1.15, 1, 1, 1],
                  hspace=0.28, wspace=0.22,
                  left=0.012, right=0.988, top=0.925, bottom=0.055)

    scan_h = ntheta * (expo + latency) / 3600.0
    raw_gb = nedf * n0 * n1 * 2 / 1024**3
    aux = [f for f in ('correct.txt', 'correct_motion.txt', 'rhapp.mat',
                       'shrink_list.mat', 'quali.mat')
           if os.path.exists(f'{dname}/{f}')]

    txt = f"""$\\bf{{{pfile.replace('_', chr(92) + '_')}}}$

  sample          Y350a (stained brain tissue)
  proposal        ls3231 @ ESRF ID16A
  acquired        {info.get('Date', '?')}
  scan type       {info.get('Scan_Type', '?')},  CCD {info.get('CCD_Mode', '?')}

$\\bf{{beam\\ and\\ optics}}$
  energy          {energy:.2f} keV   ($\\lambda$ = {wavelen * 1e10:.4f} $\\AA$)
  ring current    {info.get('SrCurrent', '?')} mA

$\\bf{{geometry}}$
  focus->detector {z1_2 * 1e3:.3f} mm
  focus->sample   z1 = {z1 * 1e3:.4f} mm
  sample->det     z2 = {z2 * 1e3:.3f} mm
  magnification   {mag:.2f}x
  eff. propagation {d_eff * 1e3:.4f} mm
  detector pixel  {det_px * 1e6:.4f} um
  $\\bf{{voxel\\ size}}$    $\\bf{{{voxel * 1e9:.3f}\\ nm}}$
  field of view   {fov * 1e6:.2f} um  ({n0} px)
  1st Fresnel zone {fringe:.0f} px  ($\\sqrt{{\\lambda d}}$ / voxel)
                  -> strongly holographic

$\\bf{{sampling}}$
  distances       {ndist}
  projections     {ntheta} over {abs(scan_range):.0f} deg
  angular step    {abs(scan_range) / ntheta:.4f} deg
  Nyquist needs   {int(np.ceil(np.pi * n0 / 2))} for {n0} px
  exposure        {expo:.2f} s  (+{latency:.2f} s latency)
  scan duration   {scan_h:.2f} h
  flats / darks   {nref} / {ndark}
  EDF frames      {nedf}   ({raw_gb:.0f} GiB raw)

$\\bf{{displacement\\ (correct.txt)}}$
  x   {shifts[:, 0].min():+7.1f} .. {shifts[:, 0].max():+7.1f} px
  y   {shifts[:, 1].min():+7.1f} .. {shifts[:, 1].max():+7.1f} px
  = +-{abs(shifts).max() * voxel * 1e6:.2f} um of sample motion

$\\bf{{aux\\ files\\ present}}$
  {', '.join(aux) if aux else 'none'}

$\\bf{{this\\ folder's\\ configs}}$
  n = {cfg.get('n', '?')}   nobj = {cfg.get('nobj', '?')}
  rotation_center_shift = {cfg.get('rotation_center_shift', '?')} px
  paganin (delta/beta)  = {cfg.get('paganin', '?')}
  no shrinkage correction (rho[tp] = 0)
"""
    axt = fig.add_subplot(gs[:, 0]); axt.axis('off')
    axt.text(0.0, 1.0, txt, va='top', ha='left', fontsize=8.1,
             family='monospace', linespacing=1.35)

    for i, (j, h) in enumerate(zip(shown, holos)):
        ax = fig.add_subplot(gs[0, i + 1])
        imshow(ax, h, f'hologram  frame {j}  ($\\theta$ = {abs(scan_range) * j / ntheta:.1f}$\\degree$)\n'
                      f'$-\\log(I/I_0)$, sample at '
                      f'({shifts[j, 1]:+.0f}, {shifts[j, 0]:+.0f}) px')

    imshow(fig.add_subplot(gs[1, 1]), ref0,
           f'flat field (mean of {args.nflat}, dark subtracted)\n'
           f'{ndark} darks averaged and subtracted throughout')
    imshow(fig.add_subplot(gs[1, 2]), holos[0] - illum,
           f'frame 0 minus the mean of {len(probe_j)} frames\n'
           f'= the sample, with the residual illumination removed')
    ax = fig.add_subplot(gs[1, 3])
    if ref1 is not None:
        r = ref1 / (ref0 + 1e-3)
        imshow(ax, r, f'flat drift over the scan\nend / start   '
                      f'(1 $\\pm$ {np.std(r):.3f}, mean {np.mean(r):.3f})',
               cmap='coolwarm', pct=(2, 98))
    else:
        ax.axis('off')
        ax.text(.5, .5, 'no end-of-scan flats', ha='center', va='center')

    ax = fig.add_subplot(gs[2, 1])
    s = ax.scatter(shifts[:, 0], shifts[:, 1], c=np.arange(ntheta), s=2,
                   cmap='viridis', linewidths=0)
    ax.set_xlabel('x displacement [px]'); ax.set_ylabel('y displacement [px]')
    ax.set_title('random displacement pattern', fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
    cb = fig.colorbar(s, ax=ax, orientation='horizontal', fraction=.05, pad=.19)
    cb.ax.tick_params(labelsize=7); cb.set_label('frame', size=8)

    ax = fig.add_subplot(gs[2, 2])
    w = min(200, ntheta)
    ax.plot(shifts[:w, 0], '.-', lw=.6, ms=2.5, label='x')
    ax.plot(shifts[:w, 1], '.-', lw=.6, ms=2.5, label='y')
    ax.set_xlabel('frame'); ax.set_ylabel('displacement [px]')
    ax.set_title(f'displacement vs frame (first {w})\n'
                 f'consecutive frames jump the full range', fontsize=9)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.plot(probe_j, trans, 'o-', ms=3, lw=.8)
    ax.set_xlabel('frame'); ax.set_ylabel(r'mean $I/I_0$')
    ax.set_title(f'mean transmission ({len(probe_j)} frames sampled)\n'
                 f'{trans.mean():.4f} $\\pm$ {trans.std():.4f}', fontsize=9)
    ax.grid(alpha=.3)

    fig.suptitle(f'{pfile}   --   ESRF ID16A, {info.get("Date", "")}   --   '
                 f'{voxel * 1e9:.2f} nm voxels, {ntheta} projections over '
                 f'{abs(scan_range):.0f}$\\degree$, {ndist} distance',
                 fontsize=12)
    fig.savefig(args.fig, dpi=110)
    print(f'figure -> {args.fig}')


if __name__ == '__main__':
    main()
