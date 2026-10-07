#!/usr/bin/env python
"""The actual blur kernel for each psf_sigma in the Y350c sweep, on 9x9 px.

Not an idealised Gaussian: this calls psf.psf_taps, so it is the DISCRETE
Gaussian w_t = exp(-s^2) I_|t|(s^2) the solver really convolves with, and the
2-D kernel is the outer product because psf_blur is separable.

9x9 is a window, not the kernel: psf_taps truncates at 4 sigma, so above
sigma ~1 px the kernel is wider than 9 and the panel is a crop.  The printed
"in 9x9" column says how much of the kernel's mass that crop holds.

    PYTHONPATH=<repo>/src python tests/psf/plot_psf_kernels.py
    ... --bin 2        # the per-level sigmas instead of the raw ones
"""
import argparse
import os

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from holotomocupy.psf import psf_taps

W = 9


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bin', type=int, default=0,
                   help='show sigma at this binning level (raw/2**bin); 0 = raw')
    p.add_argument('--sigmas', default='',
                   help='comma-separated RAW sigmas; default is the '
                        'sweep itself (0,0.5,1,1.2,1.5,2,...,5)')
    p.add_argument('--out', default=None)
    return p.parse_args()


def kernel2d(sigma, w=W):
    """(w, w) kernel as the solver applies it, plus mass inside the window."""
    _, t = psf_taps(sigma)
    if t is None:                                   # sigma = 0: identity
        k = np.zeros((w, w), 'float64')
        k[w // 2, w // 2] = 1.0
        return k, 1.0, 0.0
    t = cp.asnumpy(t).astype('float64')
    r = len(t) // 2
    x = np.arange(len(t)) - r
    realised = float(np.sqrt((t * x**2).sum() / t.sum()))
    c = w // 2
    lo, hi = r - c, r + c + 1                       # crop to the window
    if lo < 0:                                      # kernel narrower than w
        tw = np.zeros(w); tw[c - r:c + r + 1] = t
    else:
        tw = t[lo:hi]
    frac = float(tw.sum() ** 2)                     # separable -> square it
    k = np.outer(tw, tw)
    return k / k.sum(), frac, realised


def main():
    a = parse()
    # the default IS the sweep: one entry per polaris_run_psf*.sh arm
    SWEEP = [0, 0.5, 1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    raw = (np.array([float(v) for v in a.sigmas.split(',') if v.strip()])
           if a.sigmas else np.array(SWEEP, dtype='float64'))
    sig = raw / 2**a.bin
    lab = 'raw detector px' if a.bin == 0 else f'binned px at bin {a.bin}'

    print(f'sigma in {lab};  kernel on a {W}x{W} window')
    print(f'{"raw":>5s} {"sigma":>7s} {"taps":>5s} {"realised":>9s} {"in 9x9":>8s} {"peak":>8s}')
    ks = []
    for r_, s in zip(raw, sig):
        k, frac, realised = kernel2d(s)
        _, t = psf_taps(s)
        ks.append((r_, s, k))
        print(f'{r_:5.1f} {s:7.4f} {0 if t is None else len(t):5d} '
              f'{realised:9.4f} {frac:7.1%} {k.max():8.4f}')

    n = len(ks)
    fig = plt.figure(figsize=(1.45 * n, 4.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.05], hspace=0.35)
    for j, (r_, s, k) in enumerate(ks):
        axk = fig.add_subplot(gs[0, j])
        axk.imshow(k, cmap='gray', interpolation='nearest')
        axk.set_title(f'$\\sigma_{{raw}}$={r_:g}\n$\\sigma$={s:g} px', fontsize=8)
        axk.set_xticks([]); axk.set_yticks([])
    # one overlay instead of 11 panels: on a shared linear axis the sigma=0
    # spike (peak 1.0) flattens every other profile into the baseline.
    axp = fig.add_subplot(gs[1, :])
    # grayscale, capped below white so the lightest line stays visible
    cmap = plt.get_cmap('gray')
    x = np.arange(W) - W // 2
    for i, (r_, s, k) in enumerate(ks):
        axp.plot(x, k[W // 2], 'o-', ms=3, lw=1.2,
                 color=cmap(0.72 * i / max(1, len(ks) - 1)), label=f'{r_:g}')
    axp.set_yscale('log')
    axp.set_ylim(1e-4, 1.5)
    axp.set_xlabel('detector px from centre')
    axp.set_ylabel('kernel, central row (log)')
    axp.grid(alpha=0.3)
    axp.legend(title='$\\sigma_{raw}$', ncol=6, fontsize=7, title_fontsize=7,
               loc='lower center')
    fig.suptitle(f'Detector PSF kernel on {W}x{W} px - discrete Gaussian, '
                 f'sigma in {lab}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f'psf_kernels_bin{a.bin}.png')
    fig.savefig(out, dpi=130)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
