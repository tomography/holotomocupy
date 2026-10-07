#!/usr/bin/env python
"""
Step 7 — per-angle drift refinement by entropy autofocus (one GPU, no MPI).

    python step7.py config_step6_bin2.conf [options]

Takes the step-6 checkpoint, re-projects a slab of it, and searches for the
per-angle shift that minimises the entropy of the FBP.  Writes
`correct_correct3D_extra.txt` NEXT TO THE CONFIG (--out overrides), which is
exactly where step 6 looks for it; step 6 adds it to the positions it reads,
so the next run is a step-6 rerun and nothing else.

CAVEAT.  The volume this re-projects was reconstructed with the shifts already
in place, so on a converged reconstruction part of what comes out is the
metric's own bias -- 1.7 px vertical / 10.7 px horizontal ptp on the ctxl FT
deg-7 null run.  Run `tests/find_shifts_extra` on the same geometry and
compare before shipping an answer.
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import cupy as cp

from holotomocupy.autofocus import (EXTRA_NAME, Focus, ShiftFourier,
                                    cylinder_mask, fit_drift, grey_range,
                                    project_chunked, rigid_part, scan_theta,
                                    summary_png, write_correct3d_extra)
from holotomocupy.config import parse_args
from holotomocupy.tomo import Tomo


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', help='the step-6 config of the run to refine')
    p.add_argument('--iter', type=int, default=0,
                   help='checkpoint to read (0 = the highest one present)')
    p.add_argument('--ntheta', type=int, default=None,
                   help='angles to search at, on an even stride '
                        '(default: every angle in the scan)')
    p.add_argument('--nslice', type=int, default=64,
                   help='centred z slices; every evaluation reconstructs all '
                        'of them, so this is the cost knob')
    p.add_argument('--bin', type=int, default=0,
                   help='extra x/y binning LEVEL on top of the checkpoint, '
                        'factor 2**bin as everywhere else: 0 = none (default), '
                        '1 = 2x2, 2 = 4x4')
    p.add_argument('--deg', type=int, default=7, help='Legendre degree')
    p.add_argument('--maxfev', type=int, default=400,
                   help='Nelder-Mead budget PER degree')
    p.add_argument('--zchunk', type=int, default=32,
                   help='slices reconstructed at once; sets peak memory, not '
                        'the answer')
    p.add_argument('--out', default=None,
                   help='where the files land (default: next to the config, which is where step 6 looks)')
    p.add_argument('--dry-run', action='store_true',
                   help='print the geometry and the units, then stop')
    return p.parse_args()


def latest_checkpoint(path_out, want):
    d = os.path.join(path_out, 'checkpoints')
    if want:
        p = os.path.join(d, f'checkpoint_{want:04d}.h5')
        if not os.path.exists(p):
            raise SystemExit(f'{p}: no such checkpoint')
        return p, want
    found = sorted(glob.glob(os.path.join(d, 'checkpoint_[0-9]*.h5')))
    if not found:
        raise SystemExit(f'{d}: no checkpoints')
    return found[-1], int(os.path.basename(found[-1])[11:15])


def read_slab(ckpt, nb, nbz, nslice):
    """A centred z slab of obj_re, binned nbz in z and nb in x/y.

    nbz = nb*tomo_upsample makes the search grid isotropic: the object's z
    voxel is one projection-plane pixel while x/y is tomo_upsample of them.
    """
    with h5py.File(ckpt, 'r') as f:
        ds = f['obj_re']
        nzobj, nobj = ds.shape[0], ds.shape[1]
        ny = nobj // nb
        nz = min(nslice, nzobj // nbz)
        z0 = (nzobj // nbz - nz) // 2
        out = np.empty((nz, ny, ny), dtype='float32')
        for i in range(nz):
            raw = np.asarray(ds[(z0 + i) * nbz:(z0 + i + 1) * nbz,
                                :ny * nb, :ny * nb], dtype='float32')
            out[i] = raw.mean(axis=0).reshape(ny, nb, ny, nb).mean(axis=(1, 3))
    return out


def checkpoint_shape(ckpt):
    with h5py.File(ckpt, 'r') as f:
        return f['obj_re'].shape


def main():
    a = parse()
    # default --out is the config's own directory, which is where step 6
    # looks for correct_correct3D_extra.txt
    if a.out is None:
        a.out = os.path.dirname(os.path.abspath(a.config)) or '.'
    cfg = parse_args(a.config)
    os.makedirs(a.out, exist_ok=True)

    ckpt, it = latest_checkpoint(cfg.path_out, a.iter)
    up = cfg.tomo_upsample
    # The ladder shares path_out, so the newest checkpoint is not necessarily
    # at this config's bin.  Take the grid from the checkpoint itself.
    nzck, nck = checkpoint_shape(ckpt)[:2]
    # No implicit downsampling: the search grid IS the checkpoint grid unless
    # --bin asks otherwise.  nbz keeps it isotropic -- the object's z voxel is
    # one projection-plane pixel while x/y is tomo_upsample of them.
    nb = 2 ** max(0, a.bin)
    n = nck // nb
    nbz = nb * up
    nzc = min(a.zchunk or a.nslice, a.nslice)

    # One search pixel in the units of Peter's file.  The projection plane at
    # bin 0 is nobj*2**bin*tomo_upsample raw detector px wide and the search
    # grid covers the same field of view; correct3d_bin is the binning the
    # file itself is written in.
    nobj0 = cfg.nobj * 2**cfg.bin
    raw_per_px = nobj0 * up / n
    scale = raw_per_px / cfg.correct3d_bin

    with h5py.File(cfg.in_file, 'r') as f:
        ntheta_scan = len(f['/exchange/theta'])
    if a.ntheta is None:              # default: use them all
        a.ntheta = ntheta_scan
    theta_deg, theta = scan_theta(cfg.in_file, a.ntheta)

    print(f'config   {a.config}')
    print(f'  object {cfg.nzobj} x {cfg.nobj}^2 at bin {cfg.bin}, '
          f'tomo_upsample {up}  ({nobj0} x/y at bin 0)')
    print(f'  ckpt   {ckpt}  (iteration {it}, {nzck} x {nck}^2)')
    print(f'search   {a.ntheta} of {ntheta_scan} angles, grid {n} '
          f'(bin {nb} in x/y, {nbz} in z), {a.nslice} slices, '
          f'{nzc} per chunk, degree {a.deg}')
    print(f'angles   {theta_deg[0]:.4f}..{theta_deg[-1]:.4f} deg from '
          f'{cfg.in_file}::/exchange/theta, negated as step 5 does')
    print(f'units    1 search px = {raw_per_px:g} raw detector px = '
          f'{scale:g} file px at correct3d_bin={cfg.correct3d_bin}')
    if a.dry_run:
        return

    t0 = time.time()
    obj = read_slab(ckpt, nb, nbz, a.nslice)
    nz = obj.shape[0]
    print(f'read     {obj.shape} in {time.time()-t0:.1f} s')

    cl = Tomo(n, nzc, theta, -1.0)                 # sized by the CHUNK
    sh = ShiftFourier(nz, n)                       # the shift is global in z
    sel = cylinder_mask(n, nzc)
    t0 = time.time()
    d = project_chunked(cl, obj, nzc)
    cp.cuda.Stream.null.synchronize()
    del obj
    print(f'project  {tuple(d.shape)} in {time.time()-t0:.1f} s')

    t0 = time.time()
    lo, hi = grey_range(cl, d, sel, nzc)
    print(f'score    entropy, 256 bins in [{lo:.4g}, {hi:.4g}] from the '
          f'uncorrected FBP ({time.time()-t0:.1f} s)')

    fc = Focus(cl, sh, d, theta, sel, lo, hi, nzc)
    q0, mid0 = fc.sweep(np.zeros((a.ntheta, 2), 'float32'), want_mid=True)
    print(f'         entropy at no correction {q0:.5f}')

    t0 = time.time()
    print(f'Nelder-Mead from zero, degree 1 to {a.deg}, '
          f'<= {a.maxfev} evaluations per degree:')
    fc.trace.clear()
    coef, rungs = fit_drift(fc, a.deg, maxfev=a.maxfev)
    s = fc.s_of(np.linalg.lstsq(fc.V, coef.ravel(), rcond=None)[0])
    q1, mid1 = fc.sweep(s, want_mid=True)
    print(f'  {len(fc.trace)} evaluations, {fc.nit} iterations, '
          f'{time.time()-t0:.1f} s;  entropy {q1:.5f}, '
          f'{q1 - q0:+.5f} against no correction (negative is better)')
    print(f'  found    y ptp {np.ptp(s[:, 0]):.3f}  x ptp {np.ptp(s[:, 1]):.3f} '
          f'search px')
    print('  of which a rigid object shift of ' + '/'.join(
        f'{v:+.2f}' for v in rigid_part(s, fc.G)[2])
        + ' px in z/y/x, which no tomogram can see')

    tag = os.path.join(a.out, f'step7_it{it}_n{n}_th{a.ntheta}_nz{nz}')
    np.savetxt(f'{tag}_shifts.csv', np.column_stack([theta_deg, s]),
               delimiter=',', header='theta_deg,sy,sx', comments='')
    np.savetxt(f'{tag}_trace.csv', np.array(fc.trace), delimiter=',',
               header='entropy,identifiable_px,' + ','.join(
                   f'c{k}{ax}' for k in range(a.deg + 1) for ax in 'yx'),
               comments='')
    summary_png(f'{tag}.png', fc, theta_deg, s, (mid0, mid1, None),
                title=f'step 7, {os.path.basename(cfg.path_out)} '
                      f'iteration {it}: {len(fc.trace)} evaluations')
    extra = os.path.join(a.out, EXTRA_NAME)
    s_out = write_correct3d_extra(extra, coef, a.ntheta, ntheta_scan, scale)
    print(f'wrote    {os.path.abspath(extra)}  ({ntheta_scan + 1} rows, '
          f'columns horizontal,vertical)')
    print(f'         file px: y ptp {np.ptp(s_out[:, 0]):.3f}  '
          f'x ptp {np.ptp(s_out[:, 1]):.3f}')
    print(f'         {tag}_shifts.csv  {tag}_trace.csv  {tag}.png')
    print('next     rerun step 6; it reads the file from this directory')


if __name__ == '__main__':
    sys.exit(main())
