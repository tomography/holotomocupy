#!/usr/bin/env python
"""A square shifted and scaled: Shift (B-spline) against ShiftFFT (chirp-z).

    ./run.sh test_square_mag.py
    ./run.sh test_square_mag.py --n 128 --smooth 0 --half 24

The magnifications are the ones a multi-distance scan actually runs at,
m[k] = z1[k]/z1[0] off the shipped configs -- ctxl_HT and AtomiumS1_HT are
[1, 1.043, 1.214, 1.571], the 4-distance demo goes to 1.921.  m = 1 is the
control: there the chirp-z degenerates to a plain Fourier shift.

Input grid 3N/2, output grid N, so the square is resampled onto a smaller
frame exactly as curlyS does inside the solver.

WHAT TO EXPECT.  The two interpolate at the SAME sample positions, so they
agree in the interior of a band-limited input.  They differ in two places,
both by design:

  * sharp edges -- the FFT path rings (Gibbs), the B-spline does not.
    --smooth 1.5 band-limits the square and the ringing goes away.
  * outside the input grid -- at m > 1 the output reaches past the input
    (m*N/2 > 3N/4 once m > 1.5), where Shift drops the tap and leaves zero
    while ShiftFFT wraps around.  The printed numbers cover the interior
    only; the figure shows the whole frame.

Writes test_square_mag.png next to this file.
"""
import argparse
import os

import numpy as np
import cupy as cp
import scipy.ndimage as snd

from holotomocupy.shift import Shift
from holotomocupy.shift_fft import ShiftFFT

# z1 per distance (mm) from the configs; m[k] = z1[k]/z1[0].
Z1 = {'ctxl_HT / AtomiumS1_HT': [6.164, 6.428, 7.486, 9.682],
      'demo 4-distance': [5.110, 5.464, 6.879, 9.817]}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=128, help='output grid; input is 3n/2')
    p.add_argument('--half', type=int, default=24, help='half side of the square')
    p.add_argument('--smooth', type=float, default=1.5,
                   help='Gaussian sigma on the square (0 = sharp, rings)')
    p.add_argument('--margin', type=int, default=8,
                   help='pixels trimmed from each edge before the numbers')
    p.add_argument('--shift', default='2.5,-1.5', help='r = "ry,rx" in pixels')
    return p.parse_args()


def square(npsi, half, smooth):
    """Four quadrants at 0.25/0.5/0.75/1.0 so internal edges show too."""
    c = npsi // 2
    img = np.zeros((npsi, npsi), dtype='float32')
    for ys, xs, v in ((slice(c - half, c), slice(c - half, c), 0.25),
                      (slice(c - half, c), slice(c, c + half), 0.50),
                      (slice(c, c + half), slice(c - half, c), 0.75),
                      (slice(c, c + half), slice(c, c + half), 1.00)):
        img[ys, xs] = v
    if smooth > 0:
        img = snd.gaussian_filter(img, smooth, mode='constant', cval=0.0)
    return (img + 0j).astype('complex64')


def main():
    a = parse()
    n = a.n
    npsi = 3 * n // 2
    ry, rx = (float(v) for v in a.shift.split(','))

    ms = sorted({round(z / zs[0], 4) for zs in Z1.values() for z in zs})
    print(f'input {npsi}x{npsi} -> output {n}x{n},  r = ({ry:+.1f}, {rx:+.1f}),  '
          f'smooth {a.smooth}')
    for lab, zs in Z1.items():
        print(f'  {lab}: m = {[round(z / zs[0], 3) for z in zs]}')

    img = square(npsi, a.half, a.smooth)
    psi = cp.asarray(img)[cp.newaxis]
    cub, fft = Shift(n, npsi, n, npsi), ShiftFFT(n, npsi, n, npsi)
    r = cp.asarray([[ry, rx]], dtype='float32')

    outs, mg = [], a.margin
    print(f'\n{"m":>7s} {"max|cubic-fft|":>15s} {"rms":>10s} {"of cubic ptp":>13s} '
          f'{"reach/grid":>11s}')
    for mv in ms:
        m = cp.full((1, 2), mv, dtype='float32')
        oc = cp.asnumpy(cub.curlyS(psi, r, m)[0]).real
        of = cp.asnumpy(fft.curlyS(psi, r, m)[0]).real
        outs.append((mv, oc, of))
        d = (oc - of)[mg:-mg, mg:-mg]
        ptp = np.ptp(oc[mg:-mg, mg:-mg])
        # how far the output reaches into the input grid, as a fraction of it
        reach = mv * (n / 2) / (npsi / 2)
        print(f'{mv:7.4f} {np.abs(d).max():15.3e} {np.sqrt((d**2).mean()):10.3e} '
              f'{np.abs(d).max() / max(ptp, 1e-12):12.2%} {reach:10.2f}x'
              + ('  <- past the input grid' if reach > 1 else ''))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('\n(matplotlib unavailable; no figure)')
        return
    nc = len(outs)
    fig, ax = plt.subplots(4, nc, figsize=(3.1 * nc, 12.4), squeeze=False)
    vmin, vmax = -0.2, 1.2
    for k, (mv, oc, of) in enumerate(outs):
        ax[0][k].imshow(img.real, cmap='gray', vmin=vmin, vmax=vmax)
        ax[0][k].set_title(f'input {npsi}$^2$\nm={mv:.3f}', fontsize=10)
        ax[1][k].imshow(oc, cmap='gray', vmin=vmin, vmax=vmax)
        ax[1][k].set_title(f'Shift (B-spline) {n}$^2$', fontsize=10)
        ax[2][k].imshow(of, cmap='gray', vmin=vmin, vmax=vmax)
        ax[2][k].set_title(f'ShiftFFT (chirp-z)', fontsize=10)
        d = oc - of
        lim = max(float(np.abs(d).max()), 1e-9)
        im = ax[3][k].imshow(d, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax[3][k].set_title(f'difference  $\\pm${lim:.1e}', fontsize=10)
        fig.colorbar(im, ax=ax[3][k], fraction=0.046)
        for row in range(4):
            ax[row][k].set_xticks([])
            ax[row][k].set_yticks([])
    fig.suptitle(f'square shifted by ({ry:+.1f}, {rx:+.1f}) and scaled: '
                 f'B-spline vs chirp-z, smooth={a.smooth}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'test_square_mag.png')
    fig.savefig(out, dpi=110)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
