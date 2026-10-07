#!/usr/bin/env python
"""How much drift does the autofocus invent when there is none?

    ./run.sh test_find_shifts_null.py             # ~30 s on one GPU
    ./run.sh test_find_shifts_null.py --nslice 64 --ntheta 384

Same search as test_find_shifts.py, on projections with NO motion in them.
Whatever comes out is the metric's own bias: the entropy minimum does not sit
exactly at zero shift, and the search will happily walk to it.

On this phantom the bias is ~0.02 px, because the data are exactly consistent
with the model.  On a real reconstruction it is far larger -- 0.49 px was
measured on the ctxl FT volume -- so this is a regression guard on the method,
not a bound for the pipeline.  Read it that way: a step7 answer of the same
order as a null run on the same geometry is noise.

PASS means the invented drift stayed under --max-invented.
"""
import argparse
import os
import sys
import time

import numpy as np
import cupy as cp

from holotomocupy.autofocus import (Focus, ShiftFourier, cylinder_mask,
                                    fit_drift, grey_range, project_chunked,
                                    rigid_part, summary_png)
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
    p.add_argument('--maxfev', type=int, default=300,
                   help='Nelder-Mead budget PER degree')
    p.add_argument('--zchunk', type=int, default=32, help='slices per FBP')
    p.add_argument('--max-invented', type=float, default=0.1,
                   help='px rms of identifiable drift the search may invent')
    p.add_argument('--png', default='', help='write a summary figure here')
    return p.parse_args()


def main():
    a = parse()
    n, nz = a.n, a.nslice
    a.ntheta = a.ntheta or 3 * n // 2
    nzc = min(a.zchunk or nz, nz)

    theta_deg = np.linspace(0.0, 180.0, a.ntheta, endpoint=False)
    theta = (-np.radians(theta_deg)).astype('float32')
    s_zero = np.zeros((a.ntheta, 2), dtype='float32')

    cl = Tomo(n, nzc, theta, -1.0)
    sh = ShiftFourier(nz, n)
    sel = cylinder_mask(n, nzc)

    t0 = time.time()
    obj = phantom(nz, n)
    d = project_chunked(cl, obj, nzc)
    cp.cuda.Stream.null.synchronize()
    del obj
    lo, hi = grey_range(cl, d, sel, nzc)
    fc = Focus(cl, sh, d, theta, sel, lo, hi, nzc, s_true=s_zero)
    print(f'geometry   n={n} nz={nz} ntheta={a.ntheta}, no motion, '
          f'data in {time.time()-t0:.1f} s')

    q0, mid0 = fc.sweep(s_zero, want_mid=True)
    fc.trace.clear()
    t0 = time.time()
    coef, _ = fit_drift(fc, a.deg, maxfev=a.maxfev)
    s = fc.s_of(np.linalg.lstsq(fc.V, coef.ravel(), rcond=None)[0])
    q1, mid1 = fc.sweep(s, want_mid=True)
    bias = fc.err(s)
    print(f'           {len(fc.trace)} evaluations in {time.time()-t0:.1f} s')
    print(f'invented   {bias:.4f} px rms identifiable; '
          f'y ptp {np.ptp(s[:, 0]):.3f}  x ptp {np.ptp(s[:, 1]):.3f} px')
    print('           plus a rigid ' + '/'.join(
        f'{v:+.2f}' for v in rigid_part(s, fc.G)[2])
        + ' px in z/y/x, which is free and costs nothing')
    print(f'entropy    {q0:.5f} -> {q1:.5f} ({q1 - q0:+.5f}), bought with '
          f'drift that was not there')

    if a.png:
        summary_png(a.png, fc, theta_deg, s, (mid0, mid1, None),
                    title=f'find_shifts null: n={n} nz={nz} ntheta={a.ntheta}')
        print(f'wrote      {os.path.abspath(a.png)}')

    ok = bias <= a.max_invented
    print(f'{"PASS" if ok else "FAIL"}       invented {bias:.4f} px '
          f'(allowed {a.max_invented:g})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
