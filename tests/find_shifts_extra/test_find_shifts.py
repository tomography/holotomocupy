#!/usr/bin/env python
"""Does the autofocus find a drift that is really there?

    ./run.sh test_find_shifts.py                  # ~30 s on one GPU
    ./run.sh test_find_shifts.py --nslice 32 --ntheta 192 --deg 5

Injects a known Legendre drift into the projections of a textured phantom,
runs `holotomocupy.autofocus.fit_drift`, and checks that most of the
IDENTIFIABLE part of it comes back.  Identifiable means: minus the rigid
object translation, which no reference-free score can see -- the gauge.

PASS means the identifiable error dropped by at least --min-removed and the
entropy went down.  The truth is used only to judge; the objective is one
float that a real scan can compute.
"""
import argparse
import os
import sys
import time

import numpy as np
import cupy as cp

from holotomocupy.autofocus import (Focus, ShiftFourier, cylinder_mask,
                                    fit_drift, grey_range, legendre,
                                    project_chunked, rigid_part, summary_png)
from holotomocupy.tomo import Tomo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phantom import phantom                                    # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=192, help='object grid')
    p.add_argument('--nslice', type=int, default=16, help='centred z slices')
    p.add_argument('--ntheta', type=int, default=0, help='angles (0 = 3n/2)')
    p.add_argument('--deg', type=int, default=5, help='Legendre degree to fit')
    p.add_argument('--motion-y', default='0,2.0,1.2,-1.8,0.7,-1.5',
                   help='Legendre coefficients (px) of the vertical drift')
    p.add_argument('--motion-x', default='0,2.5,1.2,-1.8,0.7,1.0',
                   help='Legendre coefficients (px) of the horizontal drift')
    p.add_argument('--maxfev', type=int, default=300,
                   help='Nelder-Mead budget PER degree')
    p.add_argument('--zchunk', type=int, default=32, help='slices per FBP')
    p.add_argument('--min-removed', type=float, default=0.80,
                   help='fraction of the identifiable drift that must go')
    p.add_argument('--png', default='', help='write a summary figure here')
    return p.parse_args()


def main():
    a = parse()
    n, nz = a.n, a.nslice
    a.ntheta = a.ntheta or 3 * n // 2
    nzc = min(a.zchunk or nz, nz)

    # The pipeline reconstructs at theta = -theta_raw/180*pi; the search runs
    # in the same geometry, here and in step7.py, so an answer transfers.
    theta_deg = np.linspace(0.0, 180.0, a.ntheta, endpoint=False)
    theta = (-np.radians(theta_deg)).astype('float32')

    cy = [float(v) for v in a.motion_y.split(',') if v.strip()]
    cx = [float(v) for v in a.motion_x.split(',') if v.strip()]
    deg_true = max(len(cy), len(cx)) - 1
    c_true = np.zeros((deg_true + 1, 2))
    c_true[:len(cy), 0], c_true[:len(cx), 1] = cy, cx
    s_true = (legendre(a.ntheta, deg_true) @ c_true).astype('float32')

    cl = Tomo(n, nzc, theta, -1.0)
    sh = ShiftFourier(nz, n)
    sel = cylinder_mask(n, nzc)

    t0 = time.time()
    obj = phantom(nz, n)
    d = sh(project_chunked(cl, obj, nzc), s_true)
    cp.cuda.Stream.null.synchronize()
    del obj
    lo, hi = grey_range(cl, d, sel, nzc)
    fc = Focus(cl, sh, d, theta, sel, lo, hi, nzc, s_true=s_true)
    gpart, idn, t3 = rigid_part(s_true, fc.G)
    print(f'geometry   n={n} nz={nz} ntheta={a.ntheta}, '
          f'data in {time.time()-t0:.1f} s')
    print(f'injected   degree {deg_true}, rms {s_true.std():.3f} px; a rigid '
          f'shift of ' + '/'.join(f'{v:+.2f}' for v in t3)
          + f' px in z/y/x ({np.linalg.norm(gpart)/np.linalg.norm(s_true):.0%} '
            f'of it) is unidentifiable')

    e0 = fc.err(np.zeros_like(s_true))
    q0, mid0 = fc.sweep(np.zeros_like(s_true), want_mid=True)
    qt, midt = fc.sweep(s_true, want_mid=True)
    print(f'entropy    {q0:.5f} at no correction, {qt:.5f} at the true shifts')

    fc.trace.clear()
    t0 = time.time()
    coef, _ = fit_drift(fc, a.deg, maxfev=a.maxfev)
    s = fc.s_of(np.linalg.lstsq(fc.V, coef.ravel(), rcond=None)[0])
    q1, mid1 = fc.sweep(s, want_mid=True)
    e1 = fc.err(s)
    removed = 1 - e1 / max(e0, 1e-12)
    print(f'           {len(fc.trace)} evaluations in {time.time()-t0:.1f} s')
    print(f'result     identifiable error {e1:.4f} px, from {e0:.4f} '
          f'-- {removed:.1%} removed; entropy {q1:.5f}')

    if a.png:
        summary_png(a.png, fc, theta_deg, s, (mid0, mid1, midt),
                    title=f'find_shifts: n={n} nz={nz} ntheta={a.ntheta}')
        print(f'wrote      {os.path.abspath(a.png)}')

    ok = removed >= a.min_removed and q1 < q0
    print(f'{"PASS" if ok else "FAIL"}       '
          f'removed {removed:.1%} (need {a.min_removed:.0%}), '
          f'entropy {q1 - q0:+.5f} (need < 0)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
